# GPU Scheduler (PPO vs Heuristics)

This repo implements a **GPU job scheduling environment** (discrete-event simulation wrapped as a Gymnasium MDP), **heuristic baselines** (FIFO/SJF/Priority), and a **PPO agent** trained to learn scheduling policies.

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
  - FIFO, SJF, Priority heuristics
- **PPO**: `src/agents/ppo_agent.py`, training script: `scripts/train_ppo.py`
  - On-policy rollouts + clipped PPO update + value function
  - CSV logging for learning curves + plotting script
- **(Novelty-ready) Look-ahead signal**:
  - Observation can include a vector of **per-GPU remaining busy time** (normalized), which helps the policy reason about near-future capacity.

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
│   └── mdp_spec.md
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
PYTHONPATH=. python scripts/evaluate_baselines.py --num-episodes 10 --seed 1
```

You can also override environment scale parameters (see `scripts/evaluate_baselines.py --help`), e.g.:

```bash
PYTHONPATH=. python scripts/evaluate_baselines.py \
  --jobs-per-episode 400 \
  --max-queue-size 50 \
  --arrival-mode bursty \
  --horizon-factor 1.25
```

---

## Train PPO

```bash
source virtual/bin/activate
PYTHONPATH=. python scripts/train_ppo.py \
  --total-timesteps 200000 \
  --jobs-per-episode 400 \
  --max-queue-size 50 \
  --arrival-mode bursty \
  --horizon-factor 1.25 \
  --run-name my_run
```

This will write a CSV to `runs/my_run.csv` with training stats and periodic eval returns.

---

## Plot training curves

```bash
source virtual/bin/activate
PYTHONPATH=. python scripts/plot_training_curve.py runs/my_run.csv --out runs/my_run.png
```

---

## Documentation

- **MDP spec**: `docs/mdp_spec.md` (state/action/reward/termination, aligned to implementation)
- **Results / ablations**: keep in a separate markdown file under `docs/` (not included in this README)

