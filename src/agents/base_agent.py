"""
Base agent interface.

This defines the methods that both baselines and RL agents should expose
so that evaluation scripts can treat them uniformly.
"""

from abc import ABC, abstractmethod
from typing import Any

from src.environment.scheduler_env import SchedulerEnv


class BaseAgent(ABC):
    """
    Abstract base class for agents that act in `SchedulerEnv`.
    """

    def __init__(self, env: SchedulerEnv):
        """
        Store a reference to the environment.
        """
        self.env = env

    @abstractmethod
    def select_action(self, observation) -> int:
        """
        Given an observation, return an action (int).

        Contract:
        - Action must be in env.action_space (Discrete)
        - 0 = no-op, i>0 = job index + 1 (per your MDP spec)
        """
        raise NotImplementedError

    def update(self, *args: Any, **kwargs: Any) -> None:
        """
        Placeholder for learning updates (used in Week 3 for PPO).

        Baseline agents can leave this as a no-op.
        """
        return None

    def run_episode(self, max_steps: int = 100) -> float:
        """
        Convenience method to run one full episode with this agent.

        Returns the total reward obtained.
        """
        obs, info = self.env.reset()
        total_reward = 0.0

        for _ in range(max_steps):
            action = self.select_action(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        return total_reward


