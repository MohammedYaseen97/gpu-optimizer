"""
Evaluate baseline schedulers on the GPU scheduling environment.

Goal:
- Run FIFO, SJF, and Priority schedulers for a number of episodes
- Collect simple metrics (total reward, jobs completed, etc.)
- Print a small comparison table

You will fill in the details of how each scheduler chooses actions.
"""

import argparse
from typing import Type, Dict, Any, Optional, Tuple

import numpy as np

from src.environment.scheduler_env import SchedulerEnv
from src.agents.baselines.fifo import FIFOScheduler
from src.agents.baselines.sjf import SJFScheduler
from src.agents.baselines.rand_feas_no0 import RandomFeasibleNoNoopScheduler


# Keep baseline eval aligned with `scripts/train_ppo.py` defaults, without CLI knobs.
EVAL_SEED_BASE = 2_000_000
REWARD_KWARGS = {
    "progress_scale": 1.0,
    "time_pressure_scale": 1.0,
    "noop_penalty_scale": 0.05,
    "early_finish_bonus_scale": 1.0,
    "deadline_miss_penalty_scale": 1.0,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Experiment-level
    p.add_argument("--seed", type=int, default=1)

    # Env config
    p.add_argument("--num-gpus", type=int, default=8)
    p.add_argument("--max-queue-size", type=int, default=50)
    p.add_argument("--max-episode-steps", type=int, default=2000)
    p.add_argument("--max-duration", type=float, default=10000.0)
    p.add_argument("--jobs-per-episode", type=int, default=400)
    p.add_argument("--arrival-mode", type=str, default="bursty", choices=["all_at_zero", "bursty"])
    p.add_argument("--arrival-span", type=float, default=None)
    p.add_argument("--horizon-factor", type=float, default=1.3)

    # Eval config (match PPO multi-block eval)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--eval-num-blocks", type=int, default=3)
    return p.parse_args()


def evaluate_agent(
    agent_cls: Type,
    env_kwargs: Dict[str, Any],
    *,
    seed_base: int,
    num_episodes: int,
) -> Dict[str, float]:
    """
    Run multiple episodes with a given agent class and aggregate simple stats.

    Parameters
    ----------
    agent_cls : Type
        Class of the agent (e.g., FIFOScheduler).
    num_episodes : int
        Number of episodes to average over.

    Returns
    -------
    stats : dict
        Dictionary with at least:
        - 'avg_total_reward'
    """
    # Ensure each baseline sees the same sequence of episode workloads.
    # NOTE: `SchedulerEnv.reset(seed=...)` seeds workload generation, so we seed
    # per-episode for determinism and fair comparison.
    env = SchedulerEnv(**env_kwargs)
    agent = agent_cls(env)

    rewards = []
    invalid_fracs = []
    noop_fracs = []
    for ep in range(int(num_episodes)):
        obs, info = env.reset(seed=int(seed_base) + ep)
        total_reward = 0.0
        invalid_steps = 0
        noop_steps = 0
        steps = 0

        for _t in range(int(env.max_episode_steps)):
            action = int(agent.select_action(obs))
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
        rewards.append(total_reward)
        invalid_fracs.append(float(invalid_steps / denom))
        noop_fracs.append(float(noop_steps / denom))

    return {
        "avg_total_reward": float(np.mean(rewards)),
        "avg_invalid_frac": float(np.mean(invalid_fracs)),
        "avg_noop_frac": float(np.mean(noop_fracs)),
    }


def main() -> None:
    args = parse_args()

    env_kwargs: Dict[str, Any] = {
        "num_gpus": args.num_gpus,
        "max_queue_size": args.max_queue_size,
        "max_episode_steps": args.max_episode_steps,
        "max_duration": args.max_duration,
        "jobs_per_episode": args.jobs_per_episode,
        "arrival_mode": args.arrival_mode,
        "arrival_span": args.arrival_span,
        "horizon_factor": args.horizon_factor,
        # Reward (fixed; keep apples-to-apples with PPO runner)
        **REWARD_KWARGS,
    }

    agents = {
        "FIFO": FIFOScheduler,
        "SJF": SJFScheduler,
        "RAND_FEAS_NO0": RandomFeasibleNoNoopScheduler,
    }

    # Fixed-seed blocks for robustness, same idea as PPO's multi-block eval.
    base = int(EVAL_SEED_BASE) + int(args.seed)
    num_blocks = int(max(1, args.eval_num_blocks))
    block_size = int(args.eval_episodes)

    print("Evaluating baseline schedulers...\n")
    for name, cls in agents.items():
        block_returns = []
        block_invalids = []
        block_noops = []
        for b in range(num_blocks):
            block_base = base + b * block_size
            stats = evaluate_agent(cls, env_kwargs=env_kwargs, seed_base=block_base, num_episodes=block_size)
            block_returns.append(stats["avg_total_reward"])
            block_invalids.append(stats["avg_invalid_frac"])
            block_noops.append(stats["avg_noop_frac"])

        mean_ret = float(np.mean(block_returns))
        min_ret = float(np.min(block_returns))
        max_ret = float(np.max(block_returns))
        mean_inv = float(np.mean(block_invalids))
        mean_noop = float(np.mean(block_noops))
        print(
            f"{name:12s} | mean={mean_ret:8.3f}  min={min_ret:8.3f}  max={max_ret:8.3f} | "
            f"invalid={mean_inv:.3f}  noop={mean_noop:.3f}"
        )


if __name__ == "__main__":
    main()


