"""
Evaluate baseline schedulers on the GPU scheduling environment.

Goal:
- Run FIFO, SJF, and Priority schedulers for a number of episodes
- Collect simple metrics (total reward, jobs completed, etc.)
- Print a small comparison table

You will fill in the details of how each scheduler chooses actions.
"""

from typing import Type, Dict

import numpy as np

from src.environment.scheduler_env import SchedulerEnv
from src.agents.baselines.fifo import FIFOScheduler
from src.agents.baselines.sjf import SJFScheduler
from src.agents.baselines.priority import PriorityScheduler


def evaluate_agent(agent_cls: Type, num_episodes: int = 5) -> Dict[str, float]:
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
    env = SchedulerEnv()
    agent = agent_cls(env)

    rewards = []
    for _ in range(num_episodes):
        total_reward = agent.run_episode()
        rewards.append(total_reward)

    return {
        "avg_total_reward": float(np.mean(rewards)),
    }


def main() -> None:
    agents = {
        "FIFO": FIFOScheduler,
        "SJF": SJFScheduler,
        "PRIORITY": PriorityScheduler,
    }

    print("Evaluating baseline schedulers...\n")
    for name, cls in agents.items():
        stats = evaluate_agent(cls, num_episodes=5)
        print(f"{name:10s} | avg_total_reward = {stats['avg_total_reward']:.3f}")


if __name__ == "__main__":
    main()


