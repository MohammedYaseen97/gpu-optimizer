# GPU Job Scheduler with Reinforcement Learning

A research project implementing an intelligent GPU job scheduler using Proximal Policy Optimization (PPO). The system learns optimal scheduling policies through reinforcement learning, balancing multiple objectives: maximizing throughput, minimizing wait times, ensuring fairness, and maximizing GPU utilization.

## Learning Resources

- **Spinning Up in Deep RL** (OpenAI) - Key concepts and policy optimization
- **Hugging Face Deep RL Course** - Practical implementation with PyTorch

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
└── scripts/             # Training/evaluation scripts
```

## Week-by-Week Progress

### Week 1: Foundations + Environment Setup
**Learning:** Spinning Up Part 1-2 (Key Concepts, RL Algorithms), HF Course Unit 1

- [x] Project structure setup
- [x] Core data models (Job, GPU, Cluster)
- [x] Event system foundation
- [ ] Discrete event simulator (reset(), step(), job queue, GPU nodes)
- [ ] MDP environment interface (Gym-style)
- [ ] Simulator running with random/heuristic actions

**Deliverables:** Simulator running with random/heuristic actions, learning logbook of RL definitions

### Week 2: Baselines & Core RL Theory
**Learning:** HF Course Units 2-4 (Q-Learning → Deep Q-Learning → Policy Gradient), Spinning Up Part 3

- [ ] Implement heuristic baselines (FIFO, SJF, power-aware)
- [ ] Instrument metrics (GPU utilization, job queue latency, energy cost)
- [ ] RL agent skeleton (state input → policy network)
- [ ] Baseline results comparison

**Deliverables:** Baseline results, RL agent skeleton, reward function design write-up

### Week 3: PPO Implementation + Novelty Design
**Learning:** HF Course Unit 8 (PPO), Spinning Up PPO deep dive

- [ ] Train PPO agent on environment
- [ ] Tune hyperparameters (clip ε, value coefficient, entropy bonus)
- [ ] Design and implement novelty extension
- [ ] Evaluate RL agent vs baselines under varied workloads
- [ ] Build dashboard/visualization of scheduling results

**Deliverables:** Trained RL model outperforming heuristics, novelty implementation, GitHub repo + README, demo

