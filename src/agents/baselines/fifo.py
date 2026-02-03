"""
FIFO (First-In-First-Out) baseline scheduler.

Idea:
- Maintain a simple queue of jobs in order of submission_time.
- Always schedule the oldest waiting job that fits on the cluster.

This module defines a minimal interface you can reuse for other baselines.
"""

from typing import Optional

from src.environment.scheduler_env import SchedulerEnv


class FIFOScheduler:
    """
    FIFO baseline scheduler.

    Usage pattern:
    - Wrap a `SchedulerEnv` instance
    - On each decision step, choose which job index to schedule (or no-op)
    - Does NOT learn; just implements a fixed heuristic
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

        Contract:
        - Returns an integer in [0, env.max_queue_size]
        - 0 = no-op, i>0 = schedule job at index (i-1) in the queue window

        IMPORTANT: For fair comparison with RL agents, this baseline should
        make decisions **only from the observation**, not by peeking into
        env internals.

        Hint:
        - Observation encodes up to `max_queue_size` jobs in order (oldest first)
          plus cluster features at the end. You know:
          * number of per-job features (see MDP spec)
          * that padding entries are zeros when there are fewer jobs.
        - To implement FIFO:
          * Decode the window from `observation` into per-job feature slices.
          * Identify which indices correspond to real jobs (e.g. non-zero duration).
          * Among those, pick the earliest job that is schedulable under the
            current cluster capacity (you may still look at `env.simulator.cluster`
            for idle GPU count, but NOT at the full job queue).
          * Return (index + 1) for that job, or 0 if none is feasible.
        """
        # Observation may include extra features; job window is always the first
        # (max_queue_size * 4) entries. The last 2 entries remain cluster ratios.
        job_feats = observation[: self.env.max_queue_size * 4]
        idle_ratio = observation[-2]  # scalar in [0,1]

        assert len(job_feats) % 4 == 0
        len_jobs = len(job_feats) // 4

        # Start below any valid waiting time so FIFO picks a job even at t=0
        # (when all waiting_time values are exactly 0.0).
        max_waiting_time = -1.0  # normalized
        fifo_index = 0  # 0 = no-op

        for i in range(len_jobs):
            base = 4 * i

            duration = job_feats[base + 0]
            required_gpus_ratio = job_feats[base + 2]
            waiting_time = job_feats[base + 3]

            # Skip padded / empty slots
            if duration <= 0.0:
                continue

            # Check capacity: required_gpus_ratio <= idle_ratio
            if required_gpus_ratio > idle_ratio:
                continue

            # FIFO: pick job with largest waiting time (oldest)
            if waiting_time > max_waiting_time:
                fifo_index = i + 1
                max_waiting_time = waiting_time

        return fifo_index
        

    def run_episode(self, max_steps: Optional[int] = None) -> float:
        """
        Convenience helper to run one full episode with this scheduler.

        Returns
        -------
        total_reward : float
            Sum of rewards obtained in this episode.

        You can use this in Week 2 evaluation scripts.
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


