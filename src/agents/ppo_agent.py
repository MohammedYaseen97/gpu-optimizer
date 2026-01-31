"""PPO agent implementation for `SchedulerEnv`."""

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List
import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from src.environment.scheduler_env import SchedulerEnv
from src.agents.base_agent import BaseAgent
from src.agents.policy_net import PolicyNetwork
from src.agents.value_net import ValueNetwork


@dataclass
class PPOConfig:
    """Hyperparameters for PPO."""

    total_timesteps: int = 100_000
    num_steps: int = 256            # rollout length per update
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True


class PPOAgent(BaseAgent):
    """On-policy PPO agent wrapping the policy/value networks and env."""

    def __init__(self, env: SchedulerEnv, config: PPOConfig):
        super().__init__(env)
        self.config = config

        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.n
        self.mask_dim = env.max_queue_size + 1

        self.policy = PolicyNetwork(state_dim=self.obs_dim, action_dim=self.act_dim)
        self.value_fn = ValueNetwork(state_dim=self.obs_dim)

        # Single optimizer over both networks for now (you can split later)
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value_fn.parameters()),
            lr=config.learning_rate,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)
        self.value_fn.to(self.device)

    def select_action(self, observation) -> int:
        """Sample an action from the current policy for a single observation."""
        obs_t = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        obs_t = obs_t.unsqueeze(0)  # (1, obs_dim)

        with torch.no_grad():
            logits = self.policy(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample()

        return int(action.item())

    def run_episode(self, max_steps: Optional[int] = None) -> float:
        """Run one evaluation episode using the current policy and action mask."""
        obs, info = self.env.reset()
        total_reward = 0.0

        limit = int(max_steps) if max_steps is not None else int(self.env.max_episode_steps)
        for _ in range(limit):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            mask_t = torch.as_tensor(info["action_mask"], dtype=torch.bool, device=self.device)

            with torch.no_grad():
                logits = self.policy(obs_t)
                masked_logits = logits.clone()
                masked_logits[:, ~mask_t] = -float("inf")
                dist = Categorical(logits=masked_logits)
                action = int(dist.sample().item())

            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        return total_reward

    # ---- Rollout collection -------------------------------------------------

    def _collect_rollout(self) -> Dict[str, torch.Tensor]:
        """Collect one on-policy rollout of length `config.num_steps`.

        Also tracks any completed episodes encountered during collection so
        training can log episodic returns (more meaningful than raw loss curves).
        """

        # ---------------- utility function to get action, logprob, value --

        def get_action_and_value(
            obs: torch.Tensor, action_mask: torch.Tensor
        ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            obs         : (obs_dim,) tensor on device
            action_mask : (act_dim,) bool tensor on device
            """
            obs = obs.unsqueeze(0)  # (1, obs_dim)

            with torch.no_grad():  # no grads during data collection
                logits = self.policy(obs)          # (1, act_dim)
                value = self.value_fn(obs).squeeze(0)  # (1,1)->(1,) or (1,) -> treat as (1,)

                masked_logits = logits.clone()
                masked_logits[:, ~action_mask] = -float("inf")

                dist = Categorical(logits=masked_logits)
                action = dist.sample()             # (1,)

            return int(action.item()), dist.log_prob(action.squeeze(0)), dist.entropy().squeeze(0), value.squeeze(0)

        # -------- running the policy rollout loop -------------------------

        config = self.config

        # only need to store tensors on the device
        obs = torch.zeros((config.num_steps, self.obs_dim), device=self.device)        # (T, obs_dim)
        actions = torch.zeros(config.num_steps, dtype=torch.long, device=self.device)  # (T,)
        logprobs = torch.zeros(config.num_steps, device=self.device)                   # (T,)
        rewards = torch.zeros(config.num_steps, device=self.device)                    # (T,)
        dones = torch.zeros(config.num_steps, device=self.device)                      # (T,)
        values = torch.zeros(config.num_steps, device=self.device)                     # (T,)
        action_masks = torch.ones(
            (config.num_steps, self.mask_dim), dtype=torch.bool, device=self.device
        )  # (T, act_dim)

        next_obs, next_info = self.env.reset()
        next_done = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        ep_returns: List[float] = []
        ep_lengths: List[int] = []
        running_return = 0.0
        running_len = 0

        for step in range(config.num_steps):
            next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
            obs[step] = next_obs_t

            action_mask_t = torch.as_tensor(
                next_info["action_mask"], dtype=torch.bool, device=self.device
            )
            action_masks[step] = action_mask_t

            action, logprob, _, value = get_action_and_value(next_obs_t, action_mask_t)

            actions[step] = action
            logprobs[step] = logprob
            values[step] = value
            dones[step] = next_done

            next_obs, reward, terminated, truncated, next_info = self.env.step(action)
            rewards[step] = torch.tensor(reward, dtype=torch.float32, device=self.device)
            next_done = torch.tensor(
                float(np.logical_or(terminated, truncated)), dtype=torch.float32, device=self.device
            )

            running_return += float(reward)
            running_len += 1

            if next_done.item():
                ep_returns.append(running_return)
                ep_lengths.append(running_len)
                running_return = 0.0
                running_len = 0
                next_obs, next_info = self.env.reset()

        # Bootstrap value for last state
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        action_mask_t = torch.as_tensor(next_info["action_mask"], dtype=torch.bool, device=self.device)
        _, _, _, next_value = get_action_and_value(next_obs_t, action_mask_t)

        return {
            "obs": obs,
            "action_masks": action_masks,
            "actions": actions,
            "logprobs": logprobs,
            "rewards": rewards,
            "dones": dones,
            "values": values,
            "last_value": next_value,
            "last_done": next_done,
            "ep_returns": ep_returns,
            "ep_lengths": ep_lengths,
        }
        
    # ---- Advantage and return computation -----------------------------------
    
    def _compute_returns_and_advantages(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute GAE advantages and returns for a rollout batch."""
        
        rewards, dones, values = (batch[k] for k in ("rewards", "dones", "values"))
        last_value, last_done = batch["last_value"], batch["last_done"]
        
        advantages = torch.zeros_like(rewards).to(self.device)
        lastgaelam = 0
        for t in reversed(range(self.config.num_steps)):
            if t == self.config.num_steps - 1:
                nextnonterminal = 1 - last_done
                nextvalues = last_value
            else:
                nextnonterminal = 1 - dones[t+1]
                nextvalues = values[t+1]
            delta_t = rewards[t] + self.config.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta_t + self.config.gamma * self.config.gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values
            
        return {
            "advantages": advantages,
            "returns": returns
        }            

    # ---- PPO update step ----------------------------------------------------

    def _update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform PPO updates on a collected batch and return scalar losses."""
        
        obs, action_masks, actions, old_logprobs, advantages, returns = (
            batch[k]
            for k in ("obs", "action_masks", "actions", "logprobs", "advantages", "returns")
        )

        batch_size = self.config.num_steps
        minibatch_size = batch_size // self.config.num_minibatches

        b_inds = np.arange(batch_size)
        for epoch in range(self.config.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                # Mini-batch tensors
                mb_obs = obs[mb_inds]  # (B, obs_dim)
                mb_masks = action_masks[mb_inds]  # (B, act_dim)
                mb_actions = actions[mb_inds].long()  # (B,)
                mb_old_logprobs = old_logprobs[mb_inds]

                # Compute new logprobs, entropy, values for minibatch
                logits = self.policy(mb_obs)
                values_mb = self.value_fn(mb_obs).squeeze(-1)  # (B,)

                masked_logits = logits.clone()
                masked_logits[~mb_masks] = -float("inf")
                dist = Categorical(logits=masked_logits)

                new_logprobs = dist.log_prob(mb_actions)
                new_entropy = dist.entropy()

                logratio = new_logprobs - mb_old_logprobs
                ratio = torch.exp(logratio)
                
                mb_advantages = advantages[mb_inds]
                if self.config.normalize_advantages:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )
                
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                v_loss_unclipped = (values_mb - returns[mb_inds]) ** 2
                v_clipped = torch.clamp(
                    values_mb - returns[mb_inds],
                    -self.config.clip_coef,
                    self.config.clip_coef,
                )
                v_loss_clipped = (v_clipped - returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
                
                # Entropy loss
                entropy_loss = new_entropy.mean()
                loss = pg_loss - self.config.ent_coef * entropy_loss + v_loss * self.config.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value_fn.parameters()),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()
                
        return {
            "pg_loss": pg_loss.item(),
            "v_loss": v_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "loss": loss.item()
        }
        
    # ---- High-level training loop ------------------------------------------

    def train(
        self,
        log_csv_path: Optional[str] = None,
        eval_interval_updates: int = 0,
        eval_episodes: int = 0,
    ) -> None:
        """Main training loop: rollouts → advantages → PPO updates with logging.

        If `log_csv_path` is provided, appends per-update stats including
        mean episodic return observed during rollout collection.

        If `eval_interval_updates > 0`, runs evaluation every N updates and logs
        `eval_mean_return` (this is usually the most interpretable learning curve
        for long episodes where rollouts may not contain full episode terminations).
        """
        total_timesteps = 0
        update_idx = 0

        csv_file = None
        csv_writer = None
        if log_csv_path is not None:
            os.makedirs(os.path.dirname(log_csv_path) or ".", exist_ok=True)
            file_exists = os.path.exists(log_csv_path)
            csv_file = open(log_csv_path, "a", newline="")
            csv_writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "update",
                    "timesteps",
                    "mean_ep_return",
                    "mean_ep_length",
                    "num_episodes",
                    "eval_mean_return",
                    "loss",
                    "pg_loss",
                    "v_loss",
                    "entropy",
                ],
            )
            if not file_exists:
                csv_writer.writeheader()

        while total_timesteps < self.config.total_timesteps:
            rollout_batch = self._collect_rollout()
            ret_adv = self._compute_returns_and_advantages(rollout_batch)
            batch = {**rollout_batch, **ret_adv}
            stats = self._update(batch)

            total_timesteps += self.config.num_steps
            update_idx += 1

            ep_returns = rollout_batch.get("ep_returns", [])
            ep_lengths = rollout_batch.get("ep_lengths", [])
            mean_ep_return = float(np.mean(ep_returns)) if len(ep_returns) > 0 else float("nan")
            mean_ep_length = float(np.mean(ep_lengths)) if len(ep_lengths) > 0 else float("nan")

            # Simple logging so you can see training behaviour
            print(
                f"[PPO] update={update_idx}  timesteps={total_timesteps}  "
                f"loss={stats['loss']:.3f}  pg={stats['pg_loss']:.3f}  "
                f"v={stats['v_loss']:.3f}  ent={stats['entropy_loss']:.3f}"
            )

            if csv_writer is not None:
                eval_mean_return = float("nan")
                if eval_interval_updates and eval_episodes and (update_idx % eval_interval_updates == 0):
                    eval_rewards = [self.run_episode() for _ in range(int(eval_episodes))]
                    eval_mean_return = float(np.mean(eval_rewards))

                csv_writer.writerow(
                    {
                        "update": update_idx,
                        "timesteps": total_timesteps,
                        "mean_ep_return": mean_ep_return,
                        "mean_ep_length": mean_ep_length,
                        "num_episodes": len(ep_returns),
                        "eval_mean_return": eval_mean_return,
                        "loss": stats["loss"],
                        "pg_loss": stats["pg_loss"],
                        "v_loss": stats["v_loss"],
                        "entropy": stats["entropy_loss"],
                    }
                )
                csv_file.flush()

        if csv_file is not None:
            csv_file.close()


