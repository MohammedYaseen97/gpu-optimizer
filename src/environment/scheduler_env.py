"""
Gymnasium environment for GPU job scheduling.

This environment is the MDP wrapper around the `Simulator` and `Cluster`.
It is responsible for:
- Exposing observations and actions exactly as specified in `docs/mdp_spec.md`
- Translating agent actions into simulator calls
- Computing rewards and episode termination signals
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional

from ..simulation.simulator import Simulator
from ..environment.cluster import Cluster
from ..environment.job import Job


class SchedulerEnv(gym.Env):
    """
    GPU scheduling environment following the Gymnasium API.

    High-level contract (see `docs/mdp_spec.md`):
    - Observation = flattened state vector (queue window + cluster features)
    - Action     = discrete choice (0 = no-op, i>0 = choose job index)
    - Reward     = combination of dense, completion, priority, and episode terms
    """

    def __init__(
        self,
        num_gpus: int = 8,
        max_queue_size: int = 50,
        max_episode_steps: int = 2000,
        max_duration: float = 10000.0,
        max_priority: int = 10,
        **kwargs: Any,
    ):
        """
        Wire up the environment according to the MDP spec.
        """
        super().__init__()
        self.cluster = Cluster(num_gpus)
        self.simulator = Simulator(self.cluster, max_time=max_duration)
        
        self.num_gpus = num_gpus
        self.max_queue_size = max_queue_size
        self.max_episode_steps = max_episode_steps
        self.max_suration = max_duration
        self.max_priority = max_priority
        
        self.state_dim = max_queue_size * 5 + 2
        self.observation_space = spaces.Box(low=0.0, high 1.0, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.max_queue_size + 1)
        
        self.step_count = 0
        self.total_jobs_in_episode = 0

        return


    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Start a fresh episode.

        Responsibilities:
        - Re-initialize simulator / cluster state
        - Set internal counters (e.g. step index) back to zero
        - Optionally generate an initial workload
        - Return a valid observation in `observation_space` and an `info` dict
          (commonly containing an action mask)
        """
        self.cluster.reset()
        self.simulator.reset()
        
        self.step_count = 0
        self.total_jobs_in_episode = 0
        
        return observation_space.sample(), {"action_mask": self._get_action_mask()}
      

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Apply one agent action and advance the underlying simulator.

        Semantics (fixed by spec):
        - `action == 0`     → no-op
        - `1 <= action <= N`→ attempt to schedule job at index (action-1) in queue window

        This method should:
        - Interpret the action w.r.t. the current job queue
        - Delegate actual scheduling and time progression to the `Simulator`
        - Use your reward function to compute a scalar reward
        - Determine `terminated` (natural end) vs `truncated` (time limit)
        - Produce the next observation and an updated action mask in `info`
        """
        pass

    def _get_observation(self) -> np.ndarray:
        """
        Build the current observation vector.

        Must match the structure described in section 2 of the spec:
        - A fixed-size window over the job queue (up to `max_queue_size` jobs)
        - Per-job features and cluster features, all normalized into [0, 1]

        The exact indexing and padding strategy is up to you, but the final
        shape and feature ordering should follow your design in the spec.
        """
        pass

    def _get_action_mask(self) -> np.ndarray:
        """
        Indicate which discrete actions are currently feasible.

        Contract:
        - Shape: (max_queue_size + 1,) boolean
        - `True`  → action is allowed
        - `False` → action should never be sampled (e.g. no job / not enough GPUs)

        At a minimum:
        - No-op (0) should always be valid
        - Indices referring to non-existent jobs or impossible allocations
          must be marked invalid
        """
        pass

    def _calculate_reward(
        self,
        job_completed: Optional[Job] = None,
        episode_ended: bool = False,
    ) -> float:
        """
        Compute the scalar reward for the current step.

        The components and their intended roles are described in section 4
        of the MDP spec (dense term, completion term, priority term,
        and optional episode bonus).

        Design this so that `step()` only needs to tell you "what happened"
        (e.g. which job finished, whether the episode just ended) and the
        arithmetic and weighting live here.
        """
        pass

    def render(self) -> None:
        """
        Optional visualization hook.

        You can start with a simple textual dump of the queue / cluster state
        and later replace it with richer plots if you want.
        """
        pass
