"""
Metrics utilities for GPU scheduling experiments.

This module centralizes metric computation so both baselines and RL agents
can use the same definitions.

Design goal:
- Take raw logs / simulator outputs and compute:
  * GPU utilization
  * Job wait times
  * Throughput
  * Fairness (e.g. Jain's index)
"""

from dataclasses import dataclass, field
from typing import Dict, List
import math


@dataclass
class EpisodeMetrics:
    """
    Container for per-episode metrics.

    You can:
    - Incrementally update fields during an episode
    - Call helper methods to compute derived quantities
    """

    # Raw counts
    jobs_completed: int = 0
    total_jobs: int = 0

    # Time-related
    total_wait_time: float = 0.0
    episode_duration: float = 0.0  # simulated time from start to end

    # GPU usage (aggregate over cluster)
    gpu_busy_time: float = 0.0     # sum over GPUs of busy durations
    total_gpus: int = 0

    # Optional per-user stats for fairness (user_id -> jobs_completed)
    per_user_completed: Dict[str, int] = field(default_factory=dict)

    def avg_wait_time(self) -> float:
        """Average job wait time."""
        if self.jobs_completed == 0:
            return 0.0
        return self.total_wait_time / self.jobs_completed

    def throughput(self) -> float:
        """Jobs completed per unit time."""
        if self.episode_duration <= 0.0:
            return 0.0
        return self.jobs_completed / self.episode_duration

    def gpu_utilization(self) -> float:
        """
        Fraction of total GPU time that was busy.

        Definition:
        gpu_util = gpu_busy_time / (total_gpus * episode_duration)
        """
        if self.total_gpus <= 0 or self.episode_duration <= 0.0:
            return 0.0
        return self.gpu_busy_time / (self.total_gpus * self.episode_duration)

    def jains_fairness_index(self) -> float:
        """
        Jain's fairness index over per-user completion counts.

        J(x) = ( (sum_i x_i)^2 ) / ( n * sum_i x_i^2 )
        where x_i is metric per user (e.g., jobs_completed).
        """
        values = list(self.per_user_completed.values())
        n = len(values)
        if n == 0:
            return 1.0  # degenerate but "perfectly fair"

        s1 = sum(values)
        s2 = sum(v * v for v in values)
        if s2 == 0.0:
            return 1.0
        return (s1 * s1) / (n * s2)


