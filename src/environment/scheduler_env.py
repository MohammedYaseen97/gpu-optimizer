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
        # --- job attribute distributions (kept simple + realistic) ---
        duration_mode: Literal["uniform", "heavy_tail"] = "heavy_tail",
        # Fraction of jobs drawn from the "huge job" tail in heavy_tail mode.
        heavy_tail_frac: float = 0.15,
        # Maximum GPUs any single job may request.
        max_gpus_per_job: Optional[int] = None,
        # Correlation between duration and required_gpus.
        gpus_duration_corr: float = 0.7,
        # Priority correlation with duration:
        # - inverse: short jobs tend to have higher priority (interactive workloads)
        # - positive: long jobs tend to have higher priority
        # - none: priority independent of duration
        priority_duration_corr: Literal["none", "inverse", "positive"] = "inverse",
        priority_corr_strength: float = 0.6,
        # --- episode horizon ---
        # If provided, uses a fixed max episode time horizon.
        # If None, computes a per-episode horizon from the generated workload.
        max_episode_time: Optional[float] = None,
        horizon_factor: float = 1.3,
        max_priority: int = 10,
        # --- reward coefficients ---
        # Reward is throughput-oriented (work-by-deadline) and intentionally simple.
        #
        # Terms (all normalized by total episode work, so magnitudes are stable):
        # - + progress:     delta_completed_work / total_work
        # - - time_pressure:(dt / max_episode_time) * (available_work / total_work)
        # - - noop_penalty: when agent no-ops despite feasible work
        # - terminal:
        #     - terminated early: + (time_left / max_episode_time)
        #     - truncated (deadline hit): - (unfinished_work / total_work)
        #
        # Scales below let you tune relative importance (defaults are conservative).
        progress_scale: float = 1.0,
        time_pressure_scale: float = 1.0,
        noop_penalty_scale: float = 0.05,
        early_finish_bonus_scale: float = 1.0,
        deadline_miss_penalty_scale: float = 1.0,
        debug: bool = False,
        # (unused) Accept extra kwargs so scripts can pass through safely.
        **_: Any,
    ):
        """
        Wire up the environment according to the MDP spec.
        """
        super().__init__()

        # The simulator max_time is set on each reset (workload-aware horizon).
        self.simulator = Simulator(Cluster(num_gpus), max_time=1.0)

        # Core sizes / limits
        self.num_gpus = int(num_gpus)
        self.max_queue_size = int(max_queue_size)
        self.max_episode_steps = int(max_episode_steps)
        self.max_duration = float(max_duration)
        self._fixed_max_episode_time = max_episode_time
        self.horizon_factor = float(horizon_factor)
        self.max_episode_time = 1.0  # populated at reset
        self.max_priority = int(max_priority)

        # Workload generation configuration
        self.jobs_per_episode = int(jobs_per_episode)
        self.arrival_mode = arrival_mode
        self.arrival_span = arrival_span if arrival_span is not None else max_duration
        self.num_bursts = int(num_bursts)
        self.burst_prob = float(burst_prob)
        self.burst_std_frac = float(burst_std_frac)
        self.debug = bool(debug)

        self.duration_mode = duration_mode
        self.heavy_tail_frac = float(heavy_tail_frac)
        self.max_gpus_per_job = (
            int(max_gpus_per_job)
            if max_gpus_per_job is not None
            else int(min(self.num_gpus, 8))
        )
        self.gpus_duration_corr = float(np.clip(gpus_duration_corr, 0.0, 1.0))
        self.priority_duration_corr = priority_duration_corr
        self.priority_corr_strength = float(np.clip(priority_corr_strength, 0.0, 1.0))
        # Reward scales (throughput-oriented)
        self.progress_scale = float(progress_scale)
        self.time_pressure_scale = float(time_pressure_scale)
        self.noop_penalty_scale = float(noop_penalty_scale)
        self.early_finish_bonus_scale = float(early_finish_bonus_scale)
        self.deadline_miss_penalty_scale = float(deadline_miss_penalty_scale)
        
        # Observation layout:
        # - job window: max_queue_size * 4
        # - per-GPU remaining busy time: num_gpus
        # - cluster ratios: 2 (idle_ratio, busy_ratio)
        self.state_dim = max_queue_size * 4 + self.num_gpus + 2
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.state_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.max_queue_size + 1)
        
        self.step_count = 0
        self.total_jobs_generated = 0
        self.total_work_generated = 1.0
        self._prev_time = 0.0


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
        if seed is not None:
            np.random.seed(seed)

        self.simulator.reset()

        self.step_count = 0

        # Workload generation
        n = int(self.jobs_per_episode)
        self.total_jobs_generated = n
        durations, priorities, required_gpus = self._sample_job_attributes(n)
        submission_times = self._sample_submission_times(n)
        # Total episode work (GPU-seconds), used to normalize rewards.
        self.total_work_generated = float(np.sum(durations * required_gpus))
        self._set_episode_time_horizon(durations, required_gpus, submission_times)
        self._schedule_job_arrivals(durations, priorities, required_gpus, submission_times)
        self._drain_events_at_current_time()  # include t=0 arrivals in initial obs
        self._prev_time = float(self.simulator.current_time)

        return self._get_observation(), {
            "action_mask": self._get_action_mask(),
            "max_episode_time": self.max_episode_time,
            "total_jobs": self.total_jobs_generated,
        }

    # ---------------------------------------------------------------------
    # Workload generation helpers (kept intentionally simple and readable).
    # ---------------------------------------------------------------------

    def _sample_job_attributes(
        self, n: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample per-job (duration, priority, required_gpus).

        IMPORTANT: This method is deliberately written to be easy to read.
        If you change the distributions here, you are changing the experiment.
        """
        # ---- 1) Durations (many tiny jobs + some huge jobs) -----------------
        #
        # We keep the distribution intentionally simple:
        # - Most jobs are "tiny": uniform in [1%, 8%] of max_duration
        # - A minority are "huge": log-uniform in [8%, 100%] of max_duration
        #
        # This creates realistic variance without overdoing it.
        if self.duration_mode == "uniform":
            durations = np.random.uniform(
                low=self.max_duration * 0.05,
                high=self.max_duration * 0.2,
                size=n,
            )
        else:
            p_tail = float(np.clip(self.heavy_tail_frac, 0.0, 0.9))
            is_tail = np.random.rand(n) < p_tail

            tiny = np.random.uniform(
                low=self.max_duration * 0.01,
                high=self.max_duration * 0.08,
                size=n,
            )

            # log-uniform in [0.08, 1.0] * max_duration
            log_lo = np.log(self.max_duration * 0.08)
            log_hi = np.log(self.max_duration * 1.0)
            huge = np.exp(np.random.uniform(log_lo, log_hi, size=n))

            durations = np.where(is_tail, huge, tiny)

        # Safety: clip to [eps, max_duration]
        durations = np.clip(durations, 1e-6, float(self.max_duration)).astype(np.float32)
        duration_norm = durations / float(self.max_duration)  # in [0,1]

        # ---- 2) required_gpus correlated with duration ----------------------
        #
        # Big jobs "often" need more GPUs, but not deterministically.
        max_g = int(np.clip(self.max_gpus_per_job, 1, self.num_gpus))
        # Smooth monotone mapping: duration_norm in [0,1] -> gpus in [1,max_g]
        g_correlated = 1 + np.floor((duration_norm**0.6) * (max_g - 1)).astype(int)
        g_random = np.random.randint(1, max_g + 1, size=n)
        # Mix correlated + random to avoid overfitting to a trivial rule.
        mixed = (1.0 - self.gpus_duration_corr) * g_random + self.gpus_duration_corr * g_correlated
        required_gpus = np.clip(np.rint(mixed), 1, max_g).astype(np.int32)

        # ---- 3) Priority correlated with duration (optional) ---------------
        #
        # By default we model "interactive workloads":
        # - shorter jobs tend to have higher priority (inverse correlation)
        p_random = np.random.randint(1, self.max_priority + 1, size=n)
        if self.priority_duration_corr == "none":
            priorities = p_random
        else:
            if self.priority_duration_corr == "inverse":
                score = (1.0 - duration_norm) ** 0.8
            else:
                score = (duration_norm) ** 0.8
            p_correlated = 1 + np.floor(score * (self.max_priority - 1)).astype(int)
            mixed_p = (1.0 - self.priority_corr_strength) * p_random + self.priority_corr_strength * p_correlated
            priorities = np.clip(np.rint(mixed_p), 1, self.max_priority).astype(np.int32)

        return durations, priorities, required_gpus

    def _sample_submission_times(self, n: int) -> np.ndarray:
        """
        Sample job submission times.

        - all_at_zero: every job arrives at t=0
        - bursty: jobs arrive over a span via a mixture of:
          - Gaussian bursts around random centers
          - uniform background arrivals
        """
        if self.arrival_mode == "all_at_zero":
            return np.zeros(n, dtype=np.float32)

        span = float(self.arrival_span)
        centers = np.random.uniform(0.0, span, size=int(self.num_bursts))
        std = max(1e-6, span * float(self.burst_std_frac))

        choose_burst = np.random.rand(n) < float(self.burst_prob)
        burst_ids = np.random.randint(0, len(centers), size=n)

        burst_times = centers[burst_ids] + np.random.normal(0.0, std, size=n)
        uniform_times = np.random.uniform(0.0, span, size=n)

        submission_times = np.where(choose_burst, burst_times, uniform_times)
        return np.clip(submission_times, 0.0, span).astype(np.float32)

    def _set_episode_time_horizon(
        self, durations: np.ndarray, required_gpus: np.ndarray, submission_times: np.ndarray
    ) -> None:
        """
        Set `max_episode_time` for this episode and update the simulator horizon.

        If a fixed horizon was provided at construction, we use it.
        Otherwise, we compute a workload-aware horizon from a simple lower bound:

          LB = sum(duration * required_gpus) / num_gpus

        and give it some slack via `horizon_factor` plus the last arrival time.
        """
        if self._fixed_max_episode_time is not None:
            self.max_episode_time = float(self._fixed_max_episode_time)
        else:
            total_gpu_time = float(np.sum(durations * required_gpus))
            lower_bound = total_gpu_time / float(self.num_gpus)
            last_arrival = float(np.max(submission_times)) if len(submission_times) else 0.0
            self.max_episode_time = last_arrival + float(self.horizon_factor) * lower_bound

        self.simulator.max_time = self.max_episode_time

    def _schedule_job_arrivals(
        self,
        durations: np.ndarray,
        priorities: np.ndarray,
        required_gpus: np.ndarray,
        submission_times: np.ndarray,
    ) -> None:
        """
        Push all JOB_ARRIVAL events into the simulator's event queue.
        """
        n = int(self.total_jobs_generated)
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

    def _drain_events_at_current_time(self) -> None:
        """
        Process all events scheduled at (or before) the current simulator time.

        This makes `reset()` return a meaningful initial observation when jobs
        arrive at t=0 (rather than an empty queue that fills one event per step).
        """
        while self.simulator.event_queue and (
            self.simulator.event_queue[0].timestamp <= self.simulator.current_time
        ):
            self.simulator.step()
      

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
        # Treat infeasible actions as no-op, but report them in info so training
        # can log invalid-action ratios.
        action_i = int(action)
        action_mask = self._get_action_mask()  # shape: (max_queue_size + 1,)
        had_feasible_job = bool(np.any(action_mask[1:]))
        invalid_action = False
        if action_i != 0:
            # Any out-of-range or masked job choice is considered invalid.
            if action_i < 0 or action_i >= int(action_mask.shape[0]):
                invalid_action = True
            elif not bool(action_mask[action_i]):
                invalid_action = True

        effective_action = 0 if invalid_action else action_i

        if effective_action > 0:
            job_queue = self.simulator.get_job_queue()
            idx = effective_action - 1
            if 0 <= idx < len(job_queue):
                self.simulator.schedule_job(job_queue[idx])

        # Advance the simulator by one event if any exist. If the event queue is
        # empty, we do NOT treat that as an episode end by itself (it can happen
        # if the agent keeps no-op'ing while jobs are waiting).
        prev_t = float(self.simulator.current_time)
        completed_jobs = []
        if self.simulator.event_queue:
            _, completed_jobs = self.simulator.step()
        job_completed = completed_jobs[0] if completed_jobs else None
        dt = float(self.simulator.current_time) - prev_t

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

        took_noop = (effective_action == 0)
        reward, r_terminal = self._calculate_reward(
            job_completed=job_completed,
            episode_ended=terminated or truncated,
            took_noop=took_noop,
            invalid_action=invalid_action,
            dt=dt,
            had_feasible_job=had_feasible_job,
            terminated=terminated,
            truncated=truncated,
        )

        obs = self._get_observation()
        info: Dict[str, Any] = {
            "action_mask": self._get_action_mask(),
            "max_episode_time": self.max_episode_time,
            "invalid_action": invalid_action,
        }
        if terminated or truncated:
            info["r_terminal"] = r_terminal

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

        # Novelty: per-GPU remaining busy time (look-ahead signal).
        # Normalized by max_duration so it is directly comparable to the
        # normalized job durations in the queue window.
        rem = self._get_remaining_busy_times()  # (num_gpus,)
        rem_start = self.max_queue_size * 4
        obs[rem_start : rem_start + self.num_gpus] = rem

        idle_ratio = self.simulator.cluster.get_idle_count() / self.num_gpus
        busy_ratio = self.simulator.cluster.get_busy_count() / self.num_gpus
        obs[-2] = idle_ratio
        obs[-1] = busy_ratio

        return obs

    def _get_remaining_busy_times(self) -> np.ndarray:
        """
        Return a vector of length `num_gpus` where each entry is the remaining
        busy time until that GPU becomes idle.

        - Idle GPU → 0.0
        - Busy GPU → (completion_time - current_time) / max_duration, clipped to [0,1]

        This uses the simulator's scheduled JOB_COMPLETION event timestamps,
        which are in simulator time.
        """
        out = np.zeros(self.num_gpus, dtype=np.float32)
        now = float(self.simulator.current_time)
        for i, gpu in enumerate(self.simulator.cluster.gpus):
            if gpu.is_idle():
                continue
            job = getattr(gpu, "current_job", None)
            end_t = getattr(job, "expected_end_time", None)
            if end_t is None:
                continue
            remaining = max(0.0, float(end_t) - now)
            out[i] = float(np.clip(remaining / float(self.max_duration), 0.0, 1.0))
        return out


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
        took_noop: bool = False,
        invalid_action: bool = False,
        dt: float = 0.0,
        had_feasible_job: bool = False,
        terminated: bool = False,
        truncated: bool = False,
    ) -> Tuple[float, float]:
        """
        Throughput-oriented reward (work-by-deadline) with an explicit incentive
        to finish early.

        Returns (reward, terminal_component). terminal_component is 0.0 except
        at episode end.
        """
        total_work = max(1e-9, float(self.total_work_generated))

        # ---- helpers (work = GPU-seconds) ---------------------------------
        def job_work(job: Job) -> float:
            return float(job.estimated_duration) * float(job.required_gpus)

        # Available work = arrived jobs waiting + remaining work of running jobs.
        queue_work = 0.0
        for j in self.simulator.get_job_queue():
            queue_work += float(j.estimated_duration) * float(j.required_gpus)

        running_remaining_time = float(self._get_remaining_busy_times().sum()) * float(self.max_duration)
        available_work = queue_work + running_remaining_time

        # ---- step reward ---------------------------------------------------
        r_progress = 0.0
        if job_completed is not None:
            r_progress = self.progress_scale * (job_work(job_completed) / total_work)

        # Penalize carrying unfinished work over time (normalized by episode horizon).
        # Uses dt so the magnitude corresponds to "area under remaining-work curve".
        dt_frac = float(np.clip(dt / max(1e-9, float(self.max_episode_time)), 0.0, 10.0))
        r_time_pressure = -self.time_pressure_scale * dt_frac * (available_work / total_work)

        # Penalize no-op when there exists at least one feasible job action.
        r_noop = 0.0
        if took_noop and had_feasible_job and available_work > 0.0:
            r_noop = -self.noop_penalty_scale * (available_work / total_work)

        r_terminal = 0.0
        if episode_ended:
            # Finishing early is good: reward leftover time fraction.
            if terminated:
                time_left = max(0.0, float(self.max_episode_time) - float(self.simulator.current_time))
                r_terminal = self.early_finish_bonus_scale * (
                    time_left / max(1e-9, float(self.max_episode_time))
                )
            # Missing the deadline is bad: penalize unfinished total work fraction.
            elif truncated:
                future_arrival_work = 0.0
                for ev in self.simulator.event_queue:
                    if getattr(ev, "event_type", None) == EventType.JOB_ARRIVAL:
                        jb = getattr(ev, "data", {}).get("job", None)
                        if isinstance(jb, Job):
                            future_arrival_work += job_work(jb)
                unfinished_work = available_work + future_arrival_work
                r_terminal = -self.deadline_miss_penalty_scale * (unfinished_work / total_work)

        r_total = float(r_progress + r_time_pressure + r_noop + r_terminal)
        return r_total, float(r_terminal)

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
