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
from typing import Tuple, Dict, Any, Optional, Literal

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
        # --- workload scale ---
        jobs_per_episode: int = 400,
        arrival_mode: Literal["all_at_zero", "bursty"] = "bursty",
        # Controls the time window over which jobs arrive (only for bursty mode).
        # If None, defaults to `max_duration`.
        arrival_span: Optional[float] = None,
        # Bursty arrivals: mixture of k Gaussian bursts + uniform background.
        num_bursts: int = 6,
        burst_prob: float = 0.8,
        burst_std_frac: float = 0.03,
        # --- episode horizon ---
        # If provided, uses a fixed max episode time horizon.
        # If None, computes a per-episode horizon from the generated workload.
        max_episode_time: Optional[float] = None,
        horizon_factor: float = 1.3,
        max_priority: int = 10,
        debug: bool = False,
        **kwargs: Any,
    ):
        """
        Wire up the environment according to the MDP spec.
        """
        super().__init__()

        # The simulator max_time is set on each reset (workload-aware horizon).
        self.simulator = Simulator(Cluster(num_gpus), max_time=1.0)
        
        self.num_gpus = num_gpus
        self.max_queue_size = max_queue_size
        self.max_episode_steps = max_episode_steps
        self.max_duration = max_duration
        self._fixed_max_episode_time = max_episode_time
        self.horizon_factor = horizon_factor
        self.max_episode_time = 1.0  # populated at reset
        self.max_priority = max_priority

        self.jobs_per_episode = jobs_per_episode
        self.arrival_mode = arrival_mode
        self.arrival_span = arrival_span if arrival_span is not None else max_duration
        self.num_bursts = num_bursts
        self.burst_prob = burst_prob
        self.burst_std_frac = burst_std_frac
        self.debug = debug
        
        self.state_dim = max_queue_size * 4 + 2
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.state_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.max_queue_size + 1)
        
        self.step_count = 0
        self.total_jobs_generated = 0

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

        # --------- workload generation (jobs + arrivals) ---------------------
        n = int(self.jobs_per_episode)
        self.total_jobs_generated = n

        # Durations are per-job; max_duration is a normalization constant.
        durations = np.random.uniform(
            low=self.max_duration * 0.05, high=self.max_duration * 0.2, size=n
        )
        priorities = np.random.randint(1, self.max_priority + 1, size=n)
        required_gpus = np.random.randint(1, min(4, self.num_gpus) + 1, size=n)

        if self.arrival_mode == "all_at_zero":
            submission_times = np.zeros(n, dtype=np.float32)
        else:
            # Bursty: mixture of k Gaussian bursts + uniform background.
            span = float(self.arrival_span)
            centers = np.random.uniform(0.0, span, size=int(self.num_bursts))
            std = max(1e-6, span * float(self.burst_std_frac))

            choose_burst = np.random.rand(n) < float(self.burst_prob)
            burst_ids = np.random.randint(0, len(centers), size=n)
            burst_times = centers[burst_ids] + np.random.normal(0.0, std, size=n)
            uniform_times = np.random.uniform(0.0, span, size=n)
            submission_times = np.where(choose_burst, burst_times, uniform_times)
            submission_times = np.clip(submission_times, 0.0, span).astype(np.float32)

        # Schedule all arrivals into the simulator event queue.
        for j in range(n):
            job = Job(
                job_id=f"job_{j}",
                submission_time=float(submission_times[j]),
                required_gpus=int(required_gpus[j]),
                estimated_duration=float(durations[j]),
                priority=int(priorities[j]),
            )

            self.simulator.schedule_event(
                Event(
                    event_type=EventType.JOB_ARRIVAL,
                    timestamp=float(submission_times[j]),
                    data={"job": job},
                )
            )

        # --------- workload-aware max episode time horizon -------------------
        if self._fixed_max_episode_time is not None:
            self.max_episode_time = float(self._fixed_max_episode_time)
        else:
            # Lower bound on makespan from total GPU-time demand.
            total_gpu_time = float(np.sum(durations * required_gpus))
            lb = total_gpu_time / float(self.num_gpus)
            last_arrival = float(np.max(submission_times)) if n > 0 else 0.0
            self.max_episode_time = last_arrival + float(self.horizon_factor) * lb

        self.simulator.max_time = self.max_episode_time

        # Process all arrivals scheduled at the current time so the initial
        # observation actually contains the initial queue (submission_time=0).
        # Without this, the agent sees an empty queue at reset and arrivals
        # trickle in one-per-step, inflating queue-penalty artifacts.
        while (
            self.simulator.event_queue
            and self.simulator.event_queue[0].timestamp <= self.simulator.current_time
        ):
            self.simulator.step()

        return self._get_observation(), {
            "action_mask": self._get_action_mask(),
            "max_episode_time": self.max_episode_time,
            "total_jobs": self.total_jobs_generated,
        }
      

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
        # Interpret action safely using the action mask (treat invalid actions as no-op)
        action_mask = self._get_action_mask()
        if action < 0 or action >= len(action_mask) or (action > 0 and not action_mask[action]):
            action = 0

        if action > 0:
            job_queue = self.simulator.get_job_queue()
            if 0 <= (action - 1) < len(job_queue):
                self.simulator.schedule_job(job_queue[action - 1])

        # Advance the simulator by one event if any exist. If the event queue is
        # empty, we do NOT treat that as an episode end by itself (it can happen
        # if the agent keeps no-op'ing while jobs are waiting).
        completed_jobs = []
        if self.simulator.event_queue:
            _, completed_jobs = self.simulator.step()
        job_completed = completed_jobs[0] if completed_jobs else None

        self.step_count += 1

        # Natural termination: no waiting jobs, no busy GPUs, and no future events
        # (important when using non-zero submission_time arrivals).
        job_queue_empty = len(self.simulator.get_job_queue()) == 0
        cluster_state = self.simulator.cluster.get_state()
        no_busy_gpus = cluster_state["busy_count"] == 0
        no_future_events = len(self.simulator.event_queue) == 0
        terminated = job_queue_empty and no_busy_gpus and no_future_events

        # Truncation: episode cut short by limits (time/steps), not by natural end
        time_limit_hit = self.simulator.current_time >= self.max_episode_time
        step_limit_hit = self.step_count >= self.max_episode_steps
        truncated = (time_limit_hit or step_limit_hit) and not terminated

        reward, r_episode = self._calculate_reward(
            job_completed=job_completed, episode_ended=terminated or truncated
        )

        obs = self._get_observation()
        info: Dict[str, Any] = {
            "action_mask": self._get_action_mask(),
            "max_episode_time": self.max_episode_time,
        }
        if terminated or truncated:
            info["r_episode"] = r_episode

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
            # Spec intent: waiting time normalized by episode horizon.
            wt = (self.simulator.current_time - job.submission_time) / max(
                1e-6, float(self.max_episode_time)
            )
            obs[base + 3] = np.clip(wt, 0.0, 1.0)

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
    ) -> Tuple[float, float]:
        """
        Compute the scalar reward for the current step.

        The components and their intended roles are described in section 4
        of the MDP spec (dense term, completion term, priority term,
        and optional episode bonus).

        Design this so that `step()` only needs to tell you "what happened"
        (e.g. which job finished, whether the episode just ended) and the
        arithmetic and weighting live here.
        """
        # Dense queue penalty (scaled to be stable across episode sizes).
        #
        # Old behaviour (20-job episodes):
        #   r_dense = -0.01 * queue_length
        #   max queue penalty magnitude per step ≈ 0.2 (when queue_length≈20)
        #
        # Generalize by keeping the *max per-step penalty* at ~0.2 regardless of
        # how many jobs are in the episode:
        #   r_dense = -0.2 * (queue_length / total_jobs_generated)
        denom = max(1, int(self.total_jobs_generated))
        r = -0.2 * (len(self.simulator.get_job_queue()) / float(denom))
        r_episode = 0.0
        
        if job_completed:
            r += 1.0  # r_completion
            r += 0.1 * (
                job_completed.get_state()["priority"] / self.max_priority
            )  # r_priority
        
        if episode_ended:
            metrics = self.simulator.get_metrics()
            completion_rate = float(metrics["jobs_completed"]) / float(denom)
            r_episode = 10.0 * completion_rate
            r += r_episode
        
        return r, r_episode

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
