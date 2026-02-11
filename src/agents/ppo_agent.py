"""PPO agent implementation for `SchedulerEnv`.

Kept intentionally small and readable:
- collect rollout
- compute GAE advantages / returns
- PPO update (clipped objective)
- optional CSV logging + periodic evaluation
"""

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
    # LR scheduling:
    # - constant: lr = learning_rate
    # - linear:   lr = learning_rate * (1 - progress)
    # - exp:      lr = learning_rate * (lr_exp_end_frac ** progress)
    #   (so at the end, lr ~= learning_rate * lr_exp_end_frac)
    lr_schedule: str = "linear"  # {"constant","linear","exp"}
    lr_exp_end_frac: float = 0.2
    # Clip per-minibatch change in entropy (new - old) to stabilize updates.
    # Set <= 0 to disable.
    entropy_jump_clip: float = 0.5
    # Optional early stopping based on approximate KL divergence.
    # If set (e.g. 0.02), stop PPO epochs early when KL grows too large.
    target_kl: Optional[float] = None
    # Periodic evaluation: number of fixed seed blocks to run.
    # Each block uses a disjoint contiguous seed range of length `eval_episodes`.
    eval_num_blocks: int = 3
    # Seeding:
    # - Training should see diverse episodes, but be reproducible.
    # - Periodic eval should use a FIXED seed set so the curve is comparable.
    seed: int = 1
    train_seed_base: int = 1_000_000
    eval_seed_base: int = 2_000_000


class PPOAgent(BaseAgent):
    """On-policy PPO agent wrapping the policy/value networks and env."""

    def __init__(self, env: SchedulerEnv, config: PPOConfig):
        super().__init__(env)
        self.config = config

        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.n

        self.policy = PolicyNetwork(state_dim=self.obs_dim, action_dim=self.act_dim)
        self.value_fn = ValueNetwork(state_dim=self.obs_dim)

        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value_fn.parameters()),
            lr=config.learning_rate,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)
        self.value_fn.to(self.device)
        self._train_episode_idx = 0
        self._eval_env: Optional[SchedulerEnv] = None
        if hasattr(env, "_init_kwargs"):
            try:
                # Avoid perturbing training env state during periodic eval.
                self._eval_env = SchedulerEnv(**getattr(env, "_init_kwargs"))
            except Exception:
                self._eval_env = None

    def _reset_train_env(self):
        """Reset env for training with a deterministic, rolling seed schedule."""
        seed = int(self.config.train_seed_base) + int(self.config.seed) + int(self._train_episode_idx)
        self._train_episode_idx += 1
        return self.env.reset(seed=seed)

    @staticmethod
    def _mask_logits(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply a hard feasibility mask to logits.

        `action_mask` is boolean with True meaning "allowed".
        We set invalid logits to a very negative value (instead of -inf) to avoid
        potential NaNs on some backends.
        """
        if action_mask.dtype != torch.bool:
            action_mask = action_mask.bool()
        if action_mask.ndim == 1:
            action_mask = action_mask.unsqueeze(0)
        return logits.masked_fill(~action_mask, -1e9)

    def select_action(self, observation, action_mask: Optional[np.ndarray] = None) -> int:
        """Sample an action from the current policy for a single observation."""
        obs_t = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.policy(obs_t)
            if action_mask is not None:
                mask_t = torch.as_tensor(action_mask, device=self.device).bool()
                logits = self._mask_logits(logits, mask_t)
            dist = Categorical(logits=logits)
            return int(dist.sample().item())

    def run_episode(
        self,
        max_steps: Optional[int] = None,
        seed: Optional[int] = None,
        deterministic: bool = True,
    ) -> float:
        """Run one evaluation episode using the current policy (with hard masking)."""
        obs, info = self.env.reset(seed=seed)
        total_reward = 0.0

        limit = int(max_steps) if max_steps is not None else int(self.env.max_episode_steps)
        for _ in range(limit):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            mask = info.get("action_mask", None)
            mask_t = None if mask is None else torch.as_tensor(mask, device=self.device).bool()

            with torch.no_grad():
                logits = self.policy(obs_t)  # (1, act_dim)
                if mask_t is not None:
                    logits = self._mask_logits(logits, mask_t)
                if deterministic:
                    action = int(torch.argmax(logits, dim=-1).item())
                else:
                    dist = Categorical(logits=logits)
                    action = int(dist.sample().item())

            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        return total_reward

    def run_episode_with_stats(
        self,
        max_steps: Optional[int] = None,
        seed: Optional[int] = None,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """Run one episode and return reward + invalid/no-op rates."""
        obs, info = self.env.reset(seed=seed)
        total_reward = 0.0
        invalid_steps = 0
        noop_steps = 0
        steps = 0

        limit = int(max_steps) if max_steps is not None else int(self.env.max_episode_steps)
        for _ in range(limit):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            mask = info.get("action_mask", None)
            mask_t = None if mask is None else torch.as_tensor(mask, device=self.device).bool()
            with torch.no_grad():
                logits = self.policy(obs_t)
                if mask_t is not None:
                    logits = self._mask_logits(logits, mask_t)
                if deterministic:
                    action = int(torch.argmax(logits, dim=-1).item())
                else:
                    dist = Categorical(logits=logits)
                    action = int(dist.sample().item())

            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            steps += 1
            if action == 0:
                noop_steps += 1
            if bool(info.get("invalid_action", False)):
                invalid_steps += 1

            if terminated or truncated:
                break

        denom = float(max(1, steps))
        return {
            "total_reward": float(total_reward),
            "steps": float(steps),
            "invalid_frac": float(invalid_steps / denom),
            "noop_frac": float(noop_steps / denom),
        }

    def _run_episode_with_stats_on_env(
        self,
        env: SchedulerEnv,
        *,
        max_steps: Optional[int] = None,
        seed: Optional[int] = None,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """Same as `run_episode_with_stats`, but on a specific env instance."""
        obs, info = env.reset(seed=seed)
        total_reward = 0.0
        invalid_steps = 0
        noop_steps = 0
        steps = 0

        limit = int(max_steps) if max_steps is not None else int(env.max_episode_steps)
        for _ in range(limit):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            mask = info.get("action_mask", None)
            mask_t = None if mask is None else torch.as_tensor(mask, device=self.device).bool()
            with torch.no_grad():
                logits = self.policy(obs_t)
                if mask_t is not None:
                    logits = self._mask_logits(logits, mask_t)
                if deterministic:
                    action = int(torch.argmax(logits, dim=-1).item())
                else:
                    dist = Categorical(logits=logits)
                    action = int(dist.sample().item())

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            if action == 0:
                noop_steps += 1
            if bool(info.get("invalid_action", False)):
                invalid_steps += 1

            if terminated or truncated:
                break

        denom = float(max(1, steps))
        return {
            "total_reward": float(total_reward),
            "steps": float(steps),
            "invalid_frac": float(invalid_steps / denom),
            "noop_frac": float(noop_steps / denom),
        }

    # ---- High-level training loop (ppo.py-style) ----------------------------

    def train(
        self,
        log_csv_path: Optional[str] = None,
        eval_interval_updates: int = 0,
        eval_episodes: int = 0,
        eval_deterministic: bool = True,
    ) -> None:
        """Main training loop: rollouts → advantages → PPO updates with logging.

        If `log_csv_path` is provided, appends per-update stats including
        mean episodic return observed during rollout collection.

        If `eval_interval_updates > 0`, runs evaluation every N updates and logs
        `eval_mean_return` (this is usually the most interpretable learning curve
        for long episodes where rollouts may not contain full episode terminations).
        """
        cfg = self.config
        device = self.device

        num_steps = int(cfg.num_steps)
        total_updates = int(np.ceil(float(cfg.total_timesteps) / float(num_steps)))
        minibatch_size = int(num_steps // int(cfg.num_minibatches))
        if minibatch_size <= 0:
            raise ValueError("num_minibatches is too large for num_steps")

        # Rollout storage (re-used each update; ppo.py style)
        obs = torch.zeros((num_steps, self.obs_dim), dtype=torch.float32, device=device)
        actions = torch.zeros((num_steps,), dtype=torch.long, device=device)
        old_logprobs = torch.zeros((num_steps,), dtype=torch.float32, device=device)
        old_entropies = torch.zeros((num_steps,), dtype=torch.float32, device=device)
        rewards = torch.zeros((num_steps,), dtype=torch.float32, device=device)
        dones = torch.zeros((num_steps,), dtype=torch.float32, device=device)
        values = torch.zeros((num_steps,), dtype=torch.float32, device=device)
        action_masks = torch.zeros((num_steps, self.act_dim), dtype=torch.bool, device=device)
        invalids = torch.zeros((num_steps,), dtype=torch.float32, device=device)
        noops = torch.zeros((num_steps,), dtype=torch.float32, device=device)

        # Logging setup
        csv_file = None
        csv_writer = None
        eval_num_blocks = int(max(1, cfg.eval_num_blocks))
        eval_block_fields = [f"eval_block{b}_return" for b in range(eval_num_blocks)]
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
                    "eval_min_block_return",
                    "eval_max_block_return",
                    *eval_block_fields,
                    "invalid_frac",
                    "noop_frac",
                    "eval_invalid_frac",
                    "eval_noop_frac",
                    "lr",
                    "ent_coef",
                    # Backwards-compatible single KL column (matches older CSVs)
                    # while also logging richer diagnostics.
                    "approx_kl",
                    "approx_kl_last",
                    "approx_kl_mean",
                    "approx_kl_max",
                    "early_stop",
                    "loss",
                    "pg_loss",
                    "v_loss",
                    "entropy",
                ],
            )
            if not file_exists:
                csv_writer.writeheader()

        lr_initial = float(cfg.learning_rate)
        ent_coef_now = float(cfg.ent_coef)

        # Track best evaluation points (for your “best not in first half” question)
        best_eval_mean = (float("-inf"), -1, -1)  # (value, update, timesteps)
        best_eval_minblock = (float("-inf"), -1, -1)

        for update in range(1, total_updates + 1):
            # Match the previous implementation: each rollout starts from a fresh,
            # deterministically seeded training reset (no carry-over state between updates).
            next_obs_np, next_info = self._reset_train_env()
            next_done = torch.tensor(0.0, dtype=torch.float32, device=device)

            timesteps = int(update * num_steps)
            progress = float((update - 1) * num_steps) / float(max(1, cfg.total_timesteps))
            sched = str(cfg.lr_schedule).lower().strip()
            if sched == "constant":
                lr_now = lr_initial
            elif sched == "linear":
                lr_now = lr_initial * float(np.clip(1.0 - progress, 0.0, 1.0))
            elif sched == "exp":
                end_frac = float(np.clip(float(cfg.lr_exp_end_frac), 1e-6, 1.0))
                lr_now = lr_initial * (end_frac ** progress)
            else:
                raise ValueError(f"Unknown lr_schedule={cfg.lr_schedule!r}. Use constant|linear|exp.")
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr_now

            # ---- Rollout collection ---------------------------------------
            ep_returns: List[float] = []
            ep_lengths: List[int] = []
            running_return = 0.0
            running_len = 0

            for step in range(num_steps):
                obs_t = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
                obs[step] = obs_t
                dones[step] = next_done

                mask_np = next_info.get("action_mask", None)
                if mask_np is None:
                    mask_t = torch.ones((self.act_dim,), dtype=torch.bool, device=device)
                else:
                    mask_t = torch.as_tensor(mask_np, device=device).bool()
                action_masks[step] = mask_t

                with torch.no_grad():
                    logits = self.policy(obs_t.unsqueeze(0))  # (1, act_dim)
                    logits = self._mask_logits(logits, mask_t)
                    dist = Categorical(logits=logits)
                    action_t = dist.sample()  # (1,)
                    old_logprob_t = dist.log_prob(action_t)  # (1,)
                    old_entropy_t = dist.entropy()  # (1,)
                    value_t = self.value_fn(obs_t.unsqueeze(0)).squeeze(-1)  # (1,)

                action = int(action_t.item())
                actions[step] = action
                old_logprobs[step] = old_logprob_t.squeeze(0)
                old_entropies[step] = old_entropy_t.squeeze(0)
                values[step] = value_t.squeeze(0)

                next_obs_np, reward, terminated, truncated, next_info = self.env.step(action)
                done_flag = bool(np.logical_or(terminated, truncated))
                next_done = torch.tensor(1.0 if done_flag else 0.0, dtype=torch.float32, device=device)

                rewards[step] = float(reward)
                invalids[step] = float(bool(next_info.get("invalid_action", False)))
                noops[step] = float(action == 0)

                running_return += float(reward)
                running_len += 1
                if done_flag:
                    ep_returns.append(running_return)
                    ep_lengths.append(running_len)
                    running_return = 0.0
                    running_len = 0
                    next_obs_np, next_info = self._reset_train_env()
                    next_done = torch.tensor(0.0, dtype=torch.float32, device=device)

            # Bootstrap value for last state (GAE)
            with torch.no_grad():
                next_obs_t = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
                next_value = self.value_fn(next_obs_t.unsqueeze(0)).squeeze(0).squeeze(-1)

            # ---- GAE advantages + returns ----------------------------------
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = torch.tensor(0.0, dtype=torch.float32, device=device)
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + cfg.gamma * nextvalues * nextnonterminal - values[t]
                lastgaelam = delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + values
            if cfg.normalize_advantages:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # ---- PPO update (minibatch SGD) --------------------------------
            b_inds = np.arange(num_steps)
            approx_kls: List[float] = []
            approx_kl_last = float("nan")
            approx_kl_max = float("-inf")
            early_stop = False

            for epoch in range(int(cfg.update_epochs)):
                np.random.shuffle(b_inds)
                for start in range(0, num_steps, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]

                    mb_obs = obs[mb_inds]
                    mb_actions = actions[mb_inds]
                    mb_old_logprobs = old_logprobs[mb_inds]
                    mb_old_entropies = old_entropies[mb_inds]
                    mb_adv = advantages[mb_inds]
                    mb_returns = returns[mb_inds]
                    mb_old_values = values[mb_inds]
                    mb_masks = action_masks[mb_inds]

                    logits = self.policy(mb_obs)
                    logits = self._mask_logits(logits, mb_masks)
                    dist = Categorical(logits=logits)

                    new_logprobs = dist.log_prob(mb_actions)
                    new_entropy = dist.entropy()
                    new_values = self.value_fn(mb_obs).squeeze(-1)

                    logratio = new_logprobs - mb_old_logprobs
                    ratio = torch.exp(logratio)

                    with torch.no_grad():
                        approx_kl_t = ((ratio - 1.0) - logratio).mean()
                        approx_kl_last = float(approx_kl_t.item())
                        approx_kls.append(approx_kl_last)
                        approx_kl_max = max(approx_kl_max, approx_kl_last)

                    # Policy loss (clipped ratio)
                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss (clipped)
                    v_loss_unclipped = (new_values - mb_returns) ** 2
                    v_pred_clipped = mb_old_values + torch.clamp(
                        new_values - mb_old_values, -cfg.clip_coef, cfg.clip_coef
                    )
                    v_loss_clipped = (v_pred_clipped - mb_returns) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                    # Entropy bonus (with optional “jump” clamp)
                    if float(cfg.entropy_jump_clip) > 0.0:
                        clip = float(cfg.entropy_jump_clip)
                        new_entropy = mb_old_entropies + torch.clamp(new_entropy - mb_old_entropies, -clip, clip)
                    entropy_loss = new_entropy.mean()

                    loss = pg_loss - ent_coef_now * entropy_loss + cfg.vf_coef * v_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(self.policy.parameters()) + list(self.value_fn.parameters()),
                        cfg.max_grad_norm,
                    )
                    self.optimizer.step()

                    # Target-KL early stop (use max KL across minibatches)
                    if cfg.target_kl is not None and approx_kl_last > float(cfg.target_kl):
                        early_stop = True
                        break
                if early_stop:
                    break

            approx_kl_mean = float(np.mean(approx_kls)) if len(approx_kls) else float("nan")
            if approx_kl_max == float("-inf"):
                approx_kl_max = float("nan")

            mean_ep_return = float(np.mean(ep_returns)) if len(ep_returns) else float("nan")
            mean_ep_length = float(np.mean(ep_lengths)) if len(ep_lengths) else float("nan")
            invalid_frac = float(invalids.mean().item())
            noop_frac = float(noops.mean().item())

            print(
                f"[PPO] update={update}  timesteps={timesteps}  "
                f"loss={loss.item():.3f}  pg={pg_loss.item():.3f}  "
                f"v={v_loss.item():.3f}  ent={entropy_loss.item():.3f}  "
                f"kl_last={approx_kl_last:.4f}  kl_mean={approx_kl_mean:.4f}  kl_max={approx_kl_max:.4f}  "
                f"stop={int(early_stop)}  inv={invalid_frac:.3f}  noop={noop_frac:.3f}",
                flush=True,
            )

            # ---- Periodic evaluation --------------------------------------
            eval_mean_return = float("nan")
            eval_min_block_return = float("nan")
            eval_max_block_return = float("nan")
            eval_invalid_frac = float("nan")
            eval_noop_frac = float("nan")
            eval_block_returns: list[float] = []

            if csv_writer is not None and eval_interval_updates and eval_episodes and (update % int(eval_interval_updates) == 0):
                base = int(cfg.eval_seed_base) + int(cfg.seed)
                num_blocks = int(max(1, cfg.eval_num_blocks))
                block_size = int(eval_episodes)
                eval_env = self._eval_env if self._eval_env is not None else self.env
                all_stats = []
                for b in range(num_blocks):
                    block_base = base + b * block_size
                    block_stats = [
                        self._run_episode_with_stats_on_env(
                            eval_env,
                            seed=block_base + ep,
                            deterministic=bool(eval_deterministic),
                        )
                        for ep in range(block_size)
                    ]
                    all_stats.extend(block_stats)
                    eval_block_returns.append(float(np.mean([s["total_reward"] for s in block_stats])))

                eval_mean_return = float(np.mean(eval_block_returns))
                eval_min_block_return = float(np.min(eval_block_returns))
                eval_max_block_return = float(np.max(eval_block_returns))
                eval_invalid_frac = float(np.mean([s["invalid_frac"] for s in all_stats]))
                eval_noop_frac = float(np.mean([s["noop_frac"] for s in all_stats]))

                # Track best eval points.
                if eval_mean_return > best_eval_mean[0]:
                    best_eval_mean = (eval_mean_return, update, timesteps)
                if eval_min_block_return > best_eval_minblock[0]:
                    best_eval_minblock = (eval_min_block_return, update, timesteps)

            if csv_writer is not None:
                row = {
                    "update": update,
                    "timesteps": timesteps,
                    "mean_ep_return": mean_ep_return,
                    "mean_ep_length": mean_ep_length,
                    "num_episodes": len(ep_returns),
                    "eval_mean_return": eval_mean_return,
                    "eval_min_block_return": eval_min_block_return,
                    "eval_max_block_return": eval_max_block_return,
                    "invalid_frac": invalid_frac,
                    "noop_frac": noop_frac,
                    "eval_invalid_frac": eval_invalid_frac,
                    "eval_noop_frac": eval_noop_frac,
                    "lr": lr_now,
                    "ent_coef": ent_coef_now,
                    "approx_kl": approx_kl_last,
                    "approx_kl_last": approx_kl_last,
                    "approx_kl_mean": approx_kl_mean,
                    "approx_kl_max": approx_kl_max,
                    "early_stop": float(1.0 if early_stop else 0.0),
                    "loss": float(loss.item()),
                    "pg_loss": float(pg_loss.item()),
                    "v_loss": float(v_loss.item()),
                    "entropy": float(entropy_loss.item()),
                }
                for b, v in enumerate(eval_block_returns):
                    row[f"eval_block{b}_return"] = float(v)
                csv_writer.writerow(row)
                csv_file.flush()

        if csv_file is not None:
            csv_file.close()

        if best_eval_mean[1] != -1:
            print(
                f"[PPO] Best eval_mean_return={best_eval_mean[0]:.3f} at update={best_eval_mean[1]} timesteps={best_eval_mean[2]}",
                flush=True,
            )
        if best_eval_minblock[1] != -1:
            print(
                f"[PPO] Best eval_min_block_return={best_eval_minblock[0]:.3f} at update={best_eval_minblock[1]} timesteps={best_eval_minblock[2]}",
                flush=True,
            )


