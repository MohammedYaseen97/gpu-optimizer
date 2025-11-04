# GPU Job Scheduler with Reinforcement Learning

A research project implementing an intelligent GPU job scheduler using Proximal Policy Optimization (PPO).

## Project Structure

```
gpu-scheduler/
├── src/
│   ├── environment/     # MDP environment and core entities
│   ├── simulation/      # Discrete event simulation framework
│   ├── agents/          # RL agents (PPO, baselines) - Week 2
│   ├── utils/           # Utilities, metrics, visualization
│   └── config/          # Configuration files
├── tests/               # Unit tests
├── notebooks/           # Exploratory analysis (optional)
└── scripts/             # Training/evaluation scripts

```

## Week-by-Week Progress

### Week 1: Foundation & Environment
- [x] Project structure setup
- [x] Core data models (Job, GPU, Cluster)
- [x] Event system foundation
- [ ] Discrete event simulator
- [ ] Baseline schedulers (FIFO, priority-based)
- [ ] MDP environment interface (Gym-style)

### Week 2: Core RL Implementation
- [ ] PPO implementation
- [ ] Training loop
- [ ] Integration with environment

### Week 3: Novelty & Evaluation
- [ ] Novel scheduler system
- [ ] Comparative evaluation
- [ ] Visualization dashboard

