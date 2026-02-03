# MDP Specification: GPU Job Scheduler

**Version:** 2.0  
**Date:** 2026-02-03  
**Author:** Yaseen  
**Status:** Updated to match implementation

---

## 1. Overview

This document specifies the Markov Decision Process (MDP) formulation for the GPU job scheduling environment. The MDP defines the state space, action space, reward function, and transition dynamics that enable reinforcement learning agents to learn optimal scheduling policies.

### 1.1 Objectives

The scheduler optimizes for:
- **Throughput**: Maximize jobs completed per unit time
- **Wait Time**: Minimize time jobs spend in queue
- **Priority**: Respect job priority levels (higher priority jobs scheduled first)
- **Resource Utilization**: Maximize GPU cluster utilization

---

## 2. State Space

### 2.1 State Components

The state space is a concatenated vector combining:

1. **Job Queue Window** (Partially Observable)
   - Fixed-size sliding window of queued jobs
   - Window size: `max_queue_size` (configurable, default: 50)
   - Jobs are ordered by arrival into the queue (event order)
   - When queue is shorter than window, pad with zero vectors

2. **Look-ahead Cluster Signal** (Fully Observable)
   - Per-GPU remaining busy time (how long until each GPU becomes idle)
   - This is the main “predictive” signal that helps the policy reason about
     near-future capacity.

3. **Cluster Ratios** (Fully Observable)
   - `idle_gpus_ratio`
   - `busy_gpus_ratio`

### 2.2 State Representation

**Format**: Flat numpy array (float32)

**Dimensions**:
```
state_dim = (max_queue_size * job_feature_dim) + cluster_feature_dim
```

**Job Feature Vector** (per job in queue window):
- `estimated_duration` (normalized: `duration / max_duration`)
- `priority` (normalized: `priority / max_priority`)
- `required_gpus` (normalized: `required_gpus / num_gpus`)
- `submission_time` (normalized: `(current_time - submission_time) / max_episode_time`)

**Look-ahead Feature Vector**:
- `remaining_busy_time[gpu_i]` for each GPU \(i\)
  - Idle GPU → 0.0
  - Busy GPU → \((t_{end} - t_{now}) / max_duration\), clipped to \([0, 1]\)
  - \(t_{end}\) is the job’s scheduled completion time in simulator time.

**Cluster Ratio Features**:
- `idle_gpus_ratio` = `num_idle_gpus / total_gpus`
- `busy_gpus_ratio` = `num_busy_gpus / total_gpus`

**Total State Dimension**:
```
state_dim = (max_queue_size * 4) + num_gpus + 2
```

### 2.3 Normalization

All features are normalized to [0, 1] range for training stability:
- Durations: Divided by maximum expected duration
- Priorities: Divided by maximum priority level
- GPU counts: Divided by total GPUs
- Waiting time: Relative to submission time, normalized by `max_episode_time`
- Remaining busy time: Normalized by `max_duration`

### 2.4 Design Rationale

- **Fixed-size window**: Enables batch processing and consistent state dimensions
- **Partially observable queue**: Realistic constraint (scheduler doesn't see all future jobs)
- **Normalized features**: Ensures stable neural network training
- **Cluster metrics**: Provides agent with resource availability information

---

## 3. Action Space

### 3.1 Action Type

**Discrete action space** with action masking for invalid actions.

### 3.2 Action Semantics

Action space: `{0, 1, 2, ..., N}` where `N = max_queue_size`

- **Action 0**: No-op (do not schedule any job)
- **Action i (1 ≤ i ≤ N)**: Schedule job at index `i-1` in the queue window

### 3.3 Action Masking

Invalid actions are masked (probability set to 0) when:
- Not enough idle GPUs available for the job (`required_gpus > num_idle_gpus`)
- Job is already running
- Job index is out of bounds (queue shorter than window)
- No jobs in queue (all actions except no-op are invalid)

### 3.4 Design Rationale

- **Discrete actions**: Simpler than continuous, sufficient for scheduling decisions
- **Job index selection**: Direct mapping to queue position, easy to interpret
- **Action masking**: Prevents invalid actions without negative reward penalties
- **No-op action**: Allows agent to wait when no good scheduling opportunity exists

---

## 4. Reward Function

### 4.1 Reward Components

The reward function combines multiple objectives using dense (per-step) and event-based rewards:

#### 4.1.1 Dense Reward (Per Step)
```python
# scaled by episode size so 400-job episodes don't become automatically worse
r_dense = -queue_penalty_scale * (queue_length / total_jobs)
```
- Penalizes jobs waiting in queue
- Provides learning signal at every step
- Encourages agent to keep queue short

#### 4.1.2 Job Completion Reward
```python
# "work-based" completion reward (scaled):
work_norm = (estimated_duration / max_duration) * (required_gpus / num_gpus)
r_completion = completion_work_scale * work_norm
```
- Awarded immediately when a job completes
- Provides throughput / “useful work” signal
- Encourages scheduling that completes meaningful work, not just many tiny jobs

#### 4.1.3 Priority Bonus
```python
r_priority = priority_bonus_scale * (job.priority / max_priority)
```
- Awarded when a job completes
- Higher priority jobs yield higher reward
- Encourages agent to prioritize important jobs

#### 4.1.4 Episode Completion Bonus
```python
r_episode = episode_bonus_scale * completion_rate
```
- Awarded at episode end
- `completion_rate = jobs_completed / total_jobs`
- Provides overall throughput signal
- Applied on **terminated OR truncated** episodes (partial credit on truncation)

### 4.2 Total Reward Formula

```python
reward = r_dense + r_completion + r_priority + r_episode
```

Where:
- `r_dense`: Applied every step
- `r_completion`: Applied when job completes
- `r_priority`: Applied when job completes (if priority > 0)
- `r_episode`: Applied once at episode end

### 4.3 Reward Scale

- Typical reward range: `[-0.5, 15.0]` per step (with completion events)
- Dense component: `[-0.5, 0.0]` (negative, encouraging action)
- Completion events: `[1.0, 1.1]` (positive, encouraging throughput)
- Episode bonus: `[0.0, 10.0]` (positive, encouraging completion)

### 4.4 Design Rationale

- **Dense rewards**: Enable credit assignment and value function learning
- **Multi-objective**: Balances throughput, wait time, and priority
- **Reward shaping**: Intermediate signals guide learning toward desired outcomes
- **Normalized scale**: Prevents reward explosion and training instability

---

## 5. Transition Dynamics

### 5.1 Environment Evolution

The environment evolves through discrete events:

1. **Job Arrival**: New jobs arrive stochastically (configurable arrival process)
2. **Job Completion**: Running jobs complete after their duration
3. **Agent Action**: Agent selects job to schedule (if valid)

### 5.2 Step Semantics

**One RL step** corresponds to:
- Agent observes current state
- Agent selects action
- Environment processes action (if valid, schedule job)
- Environment advances simulator until next decision point:
  - Job arrival event
  - Job completion event
  - Fixed time interval (if no events)

### 5.3 State Transitions

**When job arrives**:
- Job added to queue (sorted by submission time)
- Queue window updated (newest jobs visible if within window)
- State vector updated

**When job completes**:
- Job removed from running set
- GPUs released back to cluster
- Completion reward triggered
- State vector updated

**When agent schedules job**:
- Job removed from queue
- Job assigned to required number of GPUs
- Job completion event scheduled (at `current_time + estimated_duration`)
- State vector updated

### 5.4 Time Advancement

- **Simulator time**: Continuous (floating point)
- **RL steps**: Discrete decision points
- **Time between steps**: Variable (depends on event timing)

### 5.5 Design Rationale

- **Event-driven**: Efficient simulation, only process when needed
- **Decision points**: Agent acts when decisions are needed (not every simulator event)
- **Stochastic arrivals**: Realistic workload patterns

---

## 6. Episodes & Termination

### 6.1 Episode Termination

An episode terminates when **all** conditions are met:
1. Job queue is empty (no jobs waiting)
2. All running jobs have completed (no busy GPUs)
3. No future events remain (important when jobs arrive over time)

### 6.2 Episode Truncation

Episodes are truncated if either limit hits (and the episode did not naturally terminate):
- Step limit: `max_episode_steps`
- Time limit: `max_episode_time`

`max_episode_time` can be:
- Fixed (if provided), or
- Workload-aware per episode:
  - Let \(LB = \sum_i (d_i \cdot g_i) / G\) be a lower bound on makespan
  - Then:
    \[
      max\_episode\_time = last\_arrival + horizon\_factor \cdot LB
    \]

### 6.3 Episode Length

- **Variable length**: Episodes end naturally when all work is done
- **Typical length**: 100-2000 steps (depends on workload)
- **Minimum length**: ~10 steps (very light workload)
- **Maximum length**: 2000 steps (truncated)

### 6.4 Design Rationale

- **Variable length**: Natural termination when work complete
- **Truncation**: Prevents infinite episodes, enables batch training
- **Realistic**: Matches real-world scheduling scenarios

---

## 7. Implementation Details

### 7.1 Environment Interface

- **Framework**: Gymnasium (OpenAI Gym compatible)
- **Observation Space**: `gymnasium.spaces.Box(low=0, high=1, shape=(state_dim,))`
- **Action Space**: `gymnasium.spaces.Discrete(max_queue_size + 1)`
- **Action Masking**: Implemented via `action_mask` in `info` dict

### 7.2 State Updates

State is recalculated at each step:
1. Update queue window (add new arrivals, remove scheduled jobs)
2. Update cluster metrics (idle/busy GPU counts)
3. Normalize all features
4. Concatenate into flat vector

### 7.3 Reward Calculation

Rewards calculated and returned at each step:
- Dense reward: Always applied
- Completion reward: Applied when `job.status == COMPLETED`
- Priority bonus: Applied with completion reward
- Episode bonus: Applied in final step if episode terminates

---

## 8. Assumptions & Constraints

### 8.1 Assumptions

1. **Job durations**: Estimated durations are provided and used for scheduling
2. **GPU homogeneity**: All GPUs are identical (no GPU-specific constraints)
3. **No preemption**: Jobs run to completion once started
4. **Deterministic durations**: Actual duration = estimated duration (can be relaxed later)
5. **Single job per step**: Agent schedules at most one job per decision point

### 8.2 Constraints

1. **GPU capacity**: Job requires `required_gpus` ≤ `total_gpus`
2. **Queue size**: Maximum `max_queue_size` jobs visible to agent
3. **Episode length**: Maximum 2000 steps per episode

---

## 9. Design Decisions & Trade-offs

### 9.1 State Representation

**Decision**: Fixed-size window with normalization  
**Trade-off**: 
- ✅ Consistent dimensions, batch-friendly
- ❌ Loses information about jobs beyond window

**Alternative considered**: Variable-length queue with padding (chose fixed window for simplicity)

### 9.2 Action Space

**Decision**: Discrete job indices  
**Trade-off**:
- ✅ Simple, interpretable, easy to mask
- ❌ Action space size grows with queue size

**Alternative considered**: Stack-based actions (deferred to potential novelty)

### 9.3 Reward Design

**Decision**: Dense + event-based rewards  
**Trade-off**:
- ✅ Enables learning, multi-objective optimization
- ❌ Requires careful tuning of reward weights

**Alternative considered**: Sparse episode-end only (rejected due to credit assignment issues)

### 9.4 Episode Termination

**Decision**: Variable length with truncation  
**Trade-off**:
- ✅ Natural termination, realistic
- ❌ Variable batch sizes in training

**Alternative considered**: Fixed-length episodes (rejected as unrealistic)

---

## 10. Future Extensions

Potential enhancements for future work:

1. **Stack-based actions**: Group jobs by duration/priority, select from stacks
2. **Graph state representation**: Represent cluster as graph, use GNNs
3. **Adaptive reward weights**: Dynamically adjust reward component weights
4. **Multi-GPU job scheduling**: Explicit GPU selection (currently automatic)
5. **Preemption**: Allow job preemption for higher priority jobs
6. **Stochastic durations**: Actual duration ≠ estimated duration

---

## Appendix A: Notation

- `s_t`: State at time step `t`
- `a_t`: Action at time step `t`
- `r_t`: Reward at time step `t`
- `N`: Maximum queue size (window size)
- `G`: Total number of GPUs in cluster
- `d_max`: Maximum job duration
- `p_max`: Maximum job priority

---

## Appendix B: Configuration Parameters

Default configuration values:

```python
max_queue_size = 50
total_gpus = 8
max_duration = 10000  # seconds
max_priority = 10
max_episode_steps = 2000
jobs_per_episode = 400
arrival_mode = "bursty"
horizon_factor = 1.3
```

---

**Document Status**: Final  
**Last Updated**: 2026
**Next Review**: After initial implementation and testing
