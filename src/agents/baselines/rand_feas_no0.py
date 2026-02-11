"""
Random feasible (non-noop) baseline scheduler.

Idea:
- If at least one job in the queue window is feasible, pick one uniformly at random.
- Never choose no-op (action 0) when a feasible job action exists.
- If no job action is feasible, fall back to no-op.

This baseline is useful as a sanity check for "always keep GPUs busy" behaviour
without any prioritization heuristic.
"""

import numpy as np

from src.environment.scheduler_env import SchedulerEnv


class RandomFeasibleNoNoopScheduler:
    def __init__(self, env: SchedulerEnv):
        self.env = env

    def select_action(self, observation) -> int:
        # Reconstruct feasibility from observation only:
        # - required_gpus_ratio is feature index 2 of each job slice
        # - idle_ratio is observation[-2]
        job_feats = observation[: self.env.max_queue_size * 4]
        idle_ratio = float(observation[-2])

        assert len(job_feats) % 4 == 0
        n = len(job_feats) // 4

        feasible_actions = []
        for i in range(n):
            base = 4 * i
            duration = float(job_feats[base + 0])
            req_ratio = float(job_feats[base + 2])
            if duration <= 0.0:
                continue
            if req_ratio > idle_ratio:
                continue
            feasible_actions.append(i + 1)  # action index

        if not feasible_actions:
            return 0
        return int(np.random.choice(feasible_actions))

