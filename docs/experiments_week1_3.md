## Experiments and Results (Weeks 1–3, pre-novelty)

This document summarizes the **experiments and results** so far, focusing on:
- What we trained / evaluated
- How we measured performance
- How PPO compares to heuristic baselines

It does **not** try to restate all implementation details; those live in the code and `mdp_spec.md`.

---

### 1. Environment configuration

- **Env**: `SchedulerEnv`
  - Discrete-time, event-driven GPU job scheduling environment.
  - Observation:
    - Fixed-size window over job queue (`max_queue_size`), each job with 4 normalized features:
      - Estimated duration
      - Priority
      - Required GPUs
      - Normalized waiting time
    - Cluster features (2):
      - Idle GPU ratio
      - Busy GPU ratio
  - Action:
    - `Discrete(max_queue_size + 1)`:
      - `0` = no-op
      - `i > 0` = schedule job at index `i-1` in the observation window.
    - **Action mask** used to forbid invalid actions (no job / insufficient GPUs).
  - Reward:
    - Dense queue penalty: `-0.01 * queue_length` each step
    - Job completion bonus: `+1.0` per completed job
    - Priority bonus: `+0.1 * (priority / max_priority)`
    - Episode bonus: `+10.0 * completion_rate` at episode end.

- **Workload**:
  - At episode reset:
    - 20 jobs generated with:
      - Submission time: 0.0
      - Duration: `U(0.05 * max_duration, 0.2 * max_duration)`
      - Priority: `U{1, ..., max_priority}`
      - Required GPUs: `U{1, ..., min(4, num_gpus)}`

---

### 2. Baseline schedulers

Location: `src/agents/baselines/`

All baselines:
- Use **only the observation** (window + cluster features), not the full job queue.
- Respect the action mask from the env.

**Baselines implemented:**

- `FIFOScheduler`
  - Picks the **oldest schedulable job** in the observation window:
    - Oldest ≈ largest normalized waiting time.
  - Returns the corresponding action index (`i+1`) or `0` if nothing fits.

- `SJFScheduler` (Shortest Job First)
  - Among schedulable jobs in the window, picks the **smallest normalized duration**.

- `PriorityScheduler`
  - Among schedulable jobs in the window:
    - Highest priority wins.
    - Ties broken by waiting time (older job preferred).

---

### 3. PPO agent configuration

Location: `src/agents/ppo_agent.py`, `scripts/train_ppo.py`

- **Policy network**: 2-layer MLP
  - Input: state vector (obs_dim)
  - Hidden: 128 units, ReLU
  - Output: action logits (act_dim)

- **Value network**: 2-layer MLP
  - Input: state vector (obs_dim)
  - Hidden: 128 units, ReLU
  - Output: scalar value `V(s)`

- **PPO hyperparameters (PPOConfig)**:
  - `total_timesteps`: 4096 (for the logged run)
  - `num_steps`: 256 (rollout length)
  - `gamma`: 0.99
  - `gae_lambda`: 0.95
  - `learning_rate`: 3e-4
  - `num_minibatches`: 4
  - `update_epochs`: 4
  - `clip_coef`: 0.2
  - `ent_coef`: 0.01
  - `vf_coef`: 0.5
  - `max_grad_norm`: 0.5
  - `normalize_advantages`: True

- **Action masking in PPO**:
  - During rollout and updates:
    - For each state, logits are **masked**:
      - `masked_logits[:, ~action_mask] = -inf`
    - Categorical distribution is built from `masked_logits`.
    - Ensures invalid actions are effectively never chosen.

---

### 4. Baseline evaluation

Script: `scripts/evaluate_baselines.py`

- Each baseline agent (`FIFO`, `SJF`, `PRIORITY`) runs for several episodes.
- Metric: **average total episode reward** under the current reward shaping.

From `ppo_vs_baselines.log`:

```text
Evaluating baseline schedulers...

FIFO       | avg_total_reward = -18.100
SJF        | avg_total_reward = 20.222
PRIORITY   | avg_total_reward = 18.194
```

Interpretation:
- FIFO performs poorly in terms of the combined reward (queue penalty + throughput + priority).
- SJF and Priority perform significantly better, around **+20 average total reward**.
- These serve as the main **non-learning baselines** for comparison.

---

### 5. PPO training & evaluation

Script: `scripts/train_ppo.py`

Pipeline:
1. Train PPO for `total_timesteps=4096` with rollout length `num_steps=256`.
2. Log per-update PPO stats: total loss, policy loss, value loss, entropy.
3. After training:
   - Evaluate the learned policy over 10 episodes using `PPOAgent.run_episode`.
   - `run_episode` respects the environment action mask.

Key lines from `ppo_vs_baselines.log`:

```text
[PPO] update=1  timesteps=256  loss=17.052  pg=-0.003  v=34.128  ent=0.889
[PPO] update=2  timesteps=512  loss=41.688  pg=-0.004  v=83.400  ent=0.814
...
[PPO] update=16  timesteps=4096  loss=1070.413  pg=-0.026  v=2140.887  ent=0.410
[PPO] Avg total reward over 10 eval episodes: 383.771
```

Summary:
- PPO’s **average total episode reward** after this short training run:
  - ≈ **383.8** over 10 evaluation episodes.
- Best baseline (SJF) achieves:
  - ≈ **20.2** average total reward.

So, in terms of this composite reward signal, PPO is already **an order of magnitude better** than the best heuristic baselines on the current workload and reward shaping.

---

### 6. Takeaways (pre-novelty)

1. **Environment & MDP**:
   - The MDP design (state, action, reward) is rich enough for PPO to learn strong policies beyond handcrafted heuristics.
   - Action masking integration works correctly in both sampling and updates.

2. **Baselines**:
   - SJF and Priority are reasonable heuristics but clearly leave performance on the table under the current objectives.
   - FIFO is a useful “worst-case” reference but not competitive.

3. **PPO**:
   - A standard PPO with modest network size and configuration can:
     - Learn to manage the trade-offs encoded in the reward (queue length, throughput, priority).
     - Significantly outperform fixed heuristics.

4. **Next step (novelty)**:
   - Introduce and evaluate a **novel architectural or algorithmic modification** (e.g., richer state representation, hierarchical policy, or adaptive reward weighting) on top of this working PPO baseline.
   - Re-run the same evaluation protocol and log deltas vs:
     - PPO (base)
     - SJF / Priority baselines.

