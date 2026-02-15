# GPU Scheduler (PPO vs Heuristics)

This repo implements a **GPU job scheduling environment** (discrete-event simulation wrapped as a Gymnasium MDP), **heuristic baselines** (FIFO/SJF/random-feasible), and a **PPO agent** trained to learn scheduling policies.

This README focuses on **what the system is** and **how to run it**.  
Results and ablations live in a separate markdown file (see `docs/`).

---

## What’s implemented

- **Environment**: `src/environment/scheduler_env.py`
  - Observation: fixed window over the job queue + cluster features
  - Action: choose a job index in the window (or no-op), with action masking
  - Workloads: bursty arrivals; configurable job attribute distributions (including heavy-tailed durations + correlations)
  - Truncation: per-episode time horizon (workload-aware) and/or step limit
- **Baselines**: `src/agents/baselines/`
  - FIFO, SJF, and `RAND_FEAS_NO0` (random feasible non-noop)
- **PPO**: `src/agents/ppo_agent.py`, training script: `scripts/train_ppo.py`
  - On-policy rollouts + clipped PPO update + value function
  - CSV logging for learning curves + plotting script
- **Lookahead signal (multiple modes)**:
  - `lookahead_mode ∈ {off, per_gpu, sorted, cdf}` keeps the observation shape unchanged while changing the representation of future GPU availability.
  - See `docs/final_results.md` for the ablation table and discussion.

---

## Research novelties (ours)

- **Lookahead observation + modality design**: we include a future-capacity signal derived from per-GPU remaining busy time, and we study **how to represent it** (`per_gpu` vs `sorted` vs `cdf` vs `off`) while keeping observation shape fixed. Implemented in `src/environment/scheduler_env.py`.
- **Cross-attention policy network**: PPO policy that scores jobs using cross-attention over GPU tokens (baseline is a standard flat **MLP** policy). Implemented in `src/agents/policy_net.py`.

---

## Repo layout

```text
gpu-scheduler/
├── src/
│   ├── environment/        # SchedulerEnv + Cluster/GPU/Job
│   ├── simulation/         # Discrete event simulator
│   ├── agents/             # PPO + baselines + networks
│   └── utils/
├── scripts/
│   ├── evaluate_baselines.py
│   ├── train_ppo.py
│   └── plot_training_curve.py
├── docs/
│   ├── mdp_spec.md
│   └── final_results.md
└── runs/                   # Training CSVs/plots
```

---

## Setup

You already have a venv in this repo:

```bash
source virtual/bin/activate
```

If you ever need dependencies:

```bash
pip install -r requirements.txt
```

All scripts assume running from repo root with `PYTHONPATH=.`:

```bash
PYTHONPATH=. python scripts/...
```

---

## Run baselines

```bash
source virtual/bin/activate
PYTHONPATH=. python scripts/evaluate_baselines.py \
  --seed 1 \
  --jobs-per-episode 400 \
  --max-queue-size 50 \
  --eval-num-blocks 3 \
  --eval-episodes 20 \
  --lookahead-mode off
```

You can also override environment scale parameters (see `scripts/evaluate_baselines.py --help`), e.g.:

```bash
PYTHONPATH=. python scripts/evaluate_baselines.py \
  --jobs-per-episode 400 \
  --max-queue-size 50 \
  --arrival-mode bursty \
  --horizon-factor 1.25 \
  --lookahead-mode per_gpu
```

---

## Train PPO

```bash
source virtual/bin/activate
PYTHONPATH=. python scripts/train_ppo.py \
  --preset baseline_stable \
  --policy-arch attn \
  --lookahead-mode cdf \
  --total-timesteps 200000 \
  --jobs-per-episode 400 \
  --max-queue-size 50 \
  --arrival-mode bursty
```

This writes:
- a training curve CSV under `runs/`
- best checkpoints under `runs/checkpoints/` (saved during training based on periodic VAL eval)
It also prints final **VAL** and **REPORT** scores at the end.

---

## Plot training curves

```bash
source virtual/bin/activate
PYTHONPATH=. python scripts/plot_training_curve.py runs/my_run.csv --out runs/my_run.png
```

---

## Documentation

- **MDP spec**: `docs/mdp_spec.md` (state/action/reward/termination, aligned to implementation)
- **Results / ablations**: `docs/final_results.md`

