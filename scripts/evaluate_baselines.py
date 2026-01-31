"""
Evaluate baseline schedulers on the GPU scheduling environment.

Goal:
- Run FIFO, SJF, and Priority schedulers for a number of episodes
- Collect simple metrics (total reward, jobs completed, etc.)
- Print a small comparison table

You will fill in the details of how each scheduler chooses actions.
"""

import argparse
from typing import Type, Dict, Any

import numpy as np

from src.environment.scheduler_env import SchedulerEnv
from src.agents.baselines.fifo import FIFOScheduler
from src.agents.baselines.sjf import SJFScheduler
from src.agents.baselines.priority import PriorityScheduler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--num-episodes", type=int, default=5)
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
    return p.parse_args()


def evaluate_agent(agent_cls: Type, env_kwargs: Dict[str, Any], seed: int, num_episodes: int) -> Dict[str, float]:
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
    np.random.seed(seed)
    env = SchedulerEnv(**env_kwargs)
    agent = agent_cls(env)

    rewards = []
    for _ in range(num_episodes):
        total_reward = agent.run_episode()
        rewards.append(total_reward)

    return {
        "avg_total_reward": float(np.mean(rewards)),
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
    }

    agents = {
        "FIFO": FIFOScheduler,
        "SJF": SJFScheduler,
        "PRIORITY": PriorityScheduler,
    }

    print("Evaluating baseline schedulers...\n")
    for name, cls in agents.items():
        stats = evaluate_agent(cls, env_kwargs=env_kwargs, seed=args.seed, num_episodes=args.num_episodes)
        print(f"{name:10s} | avg_total_reward = {stats['avg_total_reward']:.3f}")


if __name__ == "__main__":
    main()


