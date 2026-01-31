## Final Results: Baselines vs PPO (logged)

This file **replaces prior outdated results** in this repo.

It records the final (current) headline comparison: **heuristic baselines vs a fully trained PPO** under the scaled 400-job bursty workload.

---

### Experiment configuration (final)

- **Env**: `SchedulerEnv`
- **Scale**:
  - `jobs_per_episode = 400`
  - `max_queue_size = 50` (this is both observation window and action space size: `Discrete(51)`)
  - `arrival_mode = "bursty"`
  - `horizon_factor = 1.25`
  - `seed = 1`
- **Reward shaping (current)**:
  - **Dense queue penalty** (scaled by episode size): \(r_{dense} = -0.2 \cdot \frac{\text{queue\_length}}{\text{total\_jobs}}\)
  - **Completion**: `+1.0` per job completion
  - **Priority bonus**: `+0.1 * (priority / max_priority)` on completion
  - **Episode bonus**: `+10.0 * completion_rate` applied on **terminated OR truncated**
- **Episode time horizon**:
  - Workload-aware `max_episode_time` per episode, set at reset from the generated workload:
    - `max_episode_time = last_arrival + horizon_factor * (sum(duration * required_gpus) / num_gpus)`

---

### Baselines (final numbers)

Command used:

```bash
source virtual/bin/activate
PYTHONPATH=. python scripts/evaluate_baselines.py \
  --num-episodes 10 \
  --seed 1 \
  --jobs-per-episode 400 \
  --max-queue-size 50 \
  --arrival-mode bursty \
  --horizon-factor 1.25
```

Results (avg total reward over 10 episodes):

- **FIFO**: `339.596`
- **SJF**: `352.090`  (best baseline)
- **PRIORITY**: `340.312`

---

### PPO (fully trained) — final numbers

Command used:

```bash
source virtual/bin/activate
PYTHONPATH=. python scripts/train_ppo.py \
  --seed 1 \
  --jobs-per-episode 400 \
  --max-queue-size 50 \
  --arrival-mode bursty \
  --horizon-factor 1.25 \
  --total-timesteps 1000000 \
  --num-steps 2048 \
  --num-minibatches 8 \
  --update-epochs 4 \
  --learning-rate 0.0003 \
  --clip-coef 0.2 \
  --ent-coef 0.01 \
  --vf-coef 0.5 \
  --max-grad-norm 0.5 \
  --eval-interval-updates 25 \
  --eval-episodes 5 \
  --num-eval-episodes 20 \
  --run-name ppo_400jobs_bursty_1m_seed1
```

Notes:
- PPO trains in chunks of `num_steps`, so actual timesteps slightly exceed the requested cap:
  - **actual timesteps**: `1,001,472` (489 updates × 2048 steps/update)

Final PPO evaluation (avg total reward over 20 episodes):

- **PPO**: `338.979`

Artifacts:
- Training curve CSV: `runs/ppo_400jobs_bursty_1m_seed1.csv`
- Training curve plot: `runs/ppo_400jobs_bursty_1m_seed1.png`

---

### Summary

- PPO **matches** FIFO / PRIORITY on this setup, but **does not beat SJF**:
  - PPO: `338.979`
  - SJF: `352.090`

