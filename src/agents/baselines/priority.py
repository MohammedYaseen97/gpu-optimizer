"""
Priority-based baseline scheduler.

Idea:
- Among schedulable jobs, prefer higher-priority ones.
- Break ties using submission_time (earlier submitted first).
"""

from typing import Optional

from src.environment.scheduler_env import SchedulerEnv


class PriorityScheduler:
    """
    Priority-based baseline scheduler.
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

        Strategy (obs-only baseline):
        - Do NOT read the full job queue; instead:
          * Parse `observation` into per-job feature slices (fixed size).
          * Identify valid jobs (e.g. non-zero duration or priority).
        - Using only features present in the observation:
          * Among schedulable jobs (you may still use cluster idle GPU count),
            pick the one with highest `priority`.
          * If there is a tie, pick the one that appears earlier in the window
            (proxy for earlier submission_time).
        - Return its index+1 in the queue window.
        - If none schedulable, return 0 (no-op).
        """
        # Observation may include extra features; job window is always the first
        # (max_queue_size * 4) entries. The last 2 entries remain cluster ratios.
        job_feats = observation[: self.env.max_queue_size * 4]
        idle_ratio = observation[-2]  # scalar in [0,1]

        assert len(job_feats) % 4 == 0
        len_jobs = len(job_feats) // 4

        max_priority = 0.0  # normalized
        priority_index = 0  # 0 = no-op
        max_waiting_time = 0.0  # normalised

        for i in range(len_jobs):
            base = 4 * i

            duration = job_feats[base + 0]
            priority = job_feats[base + 1]
            required_gpus_ratio = job_feats[base + 2]
            waiting_time = job_feats[base + 3]

            # Skip padded / empty slots
            if duration <= 0.0:
                continue

            # Check capacity: required_gpus_ratio <= idle_ratio
            if required_gpus_ratio > idle_ratio:
                continue

            # priority indexing
            if priority > max_priority or (
                priority == max_priority and waiting_time > max_waiting_time
            ):
                priority_index = i + 1
                max_priority = priority
                max_waiting_time = waiting_time

        return priority_index

    def run_episode(self, max_steps: Optional[int] = None) -> float:
        """
        Convenience helper to run one full episode with this scheduler.

        Same pattern as in FIFO/SJF baselines.
        """
        obs, info = self.env.reset()
        total_reward = 0.0

        limit = int(max_steps) if max_steps is not None else int(self.env.max_episode_steps)
        for _ in range(limit):
            action = self.select_action(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        return total_reward

