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
from ..simulation.event import Event, EventType
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
        self.simulator = Simulator(Cluster(num_gpus), max_time=max_duration)
        
        self.num_gpus = num_gpus
        self.max_queue_size = max_queue_size
        self.max_episode_steps = max_episode_steps
        self.max_duration = max_duration
        self.max_priority = max_priority
        
        self.state_dim = max_queue_size * 4 + 2
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.state_dim,),
            dtype=np.float32,
        )
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
        - Generate an initial workload (job arrival events)
        - Return a valid observation in `observation_space` and an `info` dict
          (commonly containing an action mask)
        """
        # Optional seeding hook for reproducibility
        if seed is not None:
            np.random.seed(seed)

        self.simulator.reset()

        self.step_count = 0
        self.total_jobs_in_episode = 0

        # Simple initial workload: fixed number of jobs arriving at t=0
        num_initial_jobs = 20
        for j in range(num_initial_jobs):
            submission_time = 0.0
            estimated_duration = np.random.uniform(
                low=self.max_duration * 0.05, high=self.max_duration * 0.2
            )
            priority = np.random.randint(1, self.max_priority + 1)
            required_gpus = np.random.randint(1, min(4, self.num_gpus) + 1)

            job = Job(
                job_id=f"job_{j}",
                submission_time=submission_time,
                required_gpus=required_gpus,
                estimated_duration=estimated_duration,
                priority=priority,
            )

            event = Event(
                event_type=EventType.JOB_ARRIVAL,
                timestamp=submission_time,
                data={"job": job},
            )
            self.simulator.schedule_event(event)
            self.total_jobs_in_episode += 1

        return self._get_observation(), {"action_mask": self._get_action_mask()}
      

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
        if action > 0:
            job_queue = self.simulator.get_job_queue()
            job = job_queue[action - 1]
            self.simulator.schedule_job(job)
            self.total_jobs_in_episode += 1

        continue_sim, completed_jobs = self.simulator.step()
        job_completed = completed_jobs[0] if completed_jobs else None

        # Check if simulator naturally ran out of events
        simulator_done = not continue_sim

        reward = self._calculate_reward(
            job_completed=job_completed, episode_ended=simulator_done
        )

        self.step_count += 1

        # Natural termination: no waiting jobs and no busy GPUs
        job_queue_empty = len(self.simulator.get_job_queue()) == 0
        cluster_state = self.simulator.cluster.get_state()
        no_busy_gpus = cluster_state["busy_count"] == 0
        terminated = job_queue_empty and no_busy_gpus

        # Truncation: episode cut short by limits (time/steps), not by natural end
        time_limit_hit = self.simulator.current_time >= self.max_duration
        step_limit_hit = self.step_count >= self.max_episode_steps
        truncated = (time_limit_hit or step_limit_hit) and not terminated

        obs = self._get_observation()
        info: Dict[str, Any] = {"action_mask": self._get_action_mask()}

        return obs, reward, terminated, truncated, info
        

    def _get_observation(self) -> np.ndarray:
        """
        Build the current observation vector.

        Must match the structure described in section 2 of the spec:
        - A fixed-size window over the job queue (up to `max_queue_size` jobs)
        - Per-job features and cluster features, all normalized into [0, 1]

        The exact indexing and padding strategy is up to you, but the final
        shape and feature ordering should follow your design in the spec.
        """
        obs = np.zeros(self.state_dim, dtype=np.float32)

        job_queue = self.simulator.get_job_queue()
        num_jobs = min(len(job_queue), self.max_queue_size)

        for i in range(num_jobs):
            job = job_queue[i]
            base = i * 4

            obs[base + 0] = job.estimated_duration / self.max_duration
            obs[base + 1] = job.priority / self.max_priority
            obs[base + 2] = job.required_gpus / self.num_gpus
            obs[base + 3] = (self.simulator.current_time - job.submission_time) / self.max_duration

        idle_ratio = self.simulator.cluster.get_idle_count() / self.num_gpus
        busy_ratio = self.simulator.cluster.get_busy_count() / self.num_gpus
        obs[-2] = idle_ratio
        obs[-1] = busy_ratio

        return obs


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
        job_queue = self.simulator.get_job_queue()
        cluster = self.simulator.cluster
        
        action_mask = np.ones(self.max_queue_size+1, dtype=bool)
        
        for i in range(1, self.max_queue_size+1):
          if i > len(job_queue) or job_queue[i-1].get_state()["required_gpus"] > cluster.get_idle_count():
            action_mask[i] = False
        
        return action_mask
      

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
        r = -0.01 * len(self.simulator.get_job_queue()) # r_dense
        
        if job_completed:
          r += 1.0 # r_completion
          r += 0.1 * (job_completed.get_state()["priority"]/self.max_priority) # r_priority
        
        if episode_ended:
          metrics = self.simulator.get_metrics()
          r += 10.0 * (metrics["jobs_completed"]/metrics["total_jobs"]) # r_episode
        
        return r

    def render(self) -> None:
        """
        Optional visualization hook.

        Simple ASCII-style dump of the current simulator / cluster state.
        """
        sim = self.simulator
        cluster = sim.cluster
        job_queue = sim.get_job_queue()

        current_time = getattr(sim, "current_time", 0.0)
        idle = cluster.get_idle_count()
        busy = cluster.get_busy_count()

        # ASCII bar for GPU usage
        bar_busy = "#" * busy
        bar_idle = "." * idle
        gpu_bar = f"[{bar_busy}{bar_idle}]"

        print("=" * 60)
        print(f"Time: {current_time:.2f} | Step: {self.step_count}")
        print(f"GPUs: {gpu_bar}  (idle={idle}, busy={busy}, total={self.num_gpus})")
        print(f"Queue length: {len(job_queue)} (showing up to {self.max_queue_size})")

        window_size = min(len(job_queue), self.max_queue_size)
        for idx in range(window_size):
            job = job_queue[idx]
            state = job.get_state()
            desc = (
                f"[{idx}] id={state.get('job_id', '?')}, "
                f"prio={state.get('priority', '?')}, "
                f"gpus={state.get('required_gpus', '?')}, "
                f"est_dur={state.get('estimated_duration', '?')}"
            )
            print("  " + desc)

        if window_size < len(job_queue):
            print(f"  ... ({len(job_queue) - window_size} more jobs)")

        print("=" * 60)
