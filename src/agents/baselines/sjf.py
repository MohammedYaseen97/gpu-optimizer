"""
SJF (Shortest Job First) baseline scheduler.

Idea:
- Among all schedulable jobs in the queue, pick the one with the smallest
  estimated_duration.
"""

from typing import Optional

from src.environment.scheduler_env import SchedulerEnv


class SJFScheduler:
    """
    Shortest-Job-First baseline scheduler.
    """

    def __init__(self, env: SchedulerEnv):
        """
        Parameters
        ----------
        env : SchedulerEnv
            The environment this scheduler will act in.
        """
        self.env = env

    def select_action(self, observation) -> int:
        """
        Choose an action given the current observation.

        Strategy (obs-only, to match RL agents):
        - Treat `observation` as your only source of job information:
          * It contains a fixed-size window over the queue, in order.
          * Each job occupies a fixed feature slice; padding is zeros.
        - Reconstruct per-job feature slices from `observation`.
        - Among jobs that appear to be real (e.g. non-zero duration), and that
          are schedulable under current idle GPU capacity (you can still consult
          `env.simulator.cluster` for idle count), pick the one with minimum
          `estimated_duration`.
        - Return its index+1 in the queue window.
        - If no schedulable job exists, return 0 (no-op).
        """
        job_array = observation[:-2]
        idle_ratio = observation[-2]  # scalar in [0,1]

        assert len(job_array) % 4 == 0
        len_jobs = len(job_array) // 4

        min_duration = float("inf")
        sjf_index = 0  # 0 = no-op

        for i in range(len_jobs):
            base = 4 * i

            duration = job_array[base + 0]
            required_gpus_ratio = job_array[base + 2]

            # Skip padded / empty slots
            if duration <= 0.0:
                continue

            # Check capacity: required_gpus_ratio <= idle_ratio
            if required_gpus_ratio > idle_ratio:
                continue

            # SJF: pick smallest normalized duration
            if duration < min_duration:
                sjf_index = i + 1
                min_duration = duration

        return sjf_index


    def run_episode(self, max_steps: int = 100) -> float:
        """
        Convenience helper to run one full episode with this scheduler.

        Mirrors the pattern used in `FIFOScheduler`.
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


