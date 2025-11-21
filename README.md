# GPU Job Scheduler with Reinforcement Learning

A research project implementing an intelligent GPU job scheduler using Proximal Policy Optimization (PPO). The system learns optimal scheduling policies through reinforcement learning, balancing multiple objectives: maximizing throughput, minimizing wait times, ensuring fairness, and maximizing GPU utilization.

## Learning Resources

This project follows a **parallel two-lane approach**:
- **Lane A:** Project implementation (environment → baselines → RL)
- **Lane B:** RL theory learning (in lockstep with implementation)

**Primary Resources:**
- **Spinning Up in Deep RL** (OpenAI) - Core text with clear math + code
- **Hugging Face Deep RL Course** (Units 0-4, 6, 8) - Hands-on implementation labs
- **David Silver UCL Lectures** (1-5) - Deep conceptual intuition
- **Sutton & Barto (2e)** - Reference only (optional lookup)

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

### Week 1: Foundations + Environment
**Learning (Lane B):**
- [x] Spinning Up: Intro + Key Concepts
- [x] HF Course: Unit 0 (Introduction) → Unit 1 (Foundations of Deep RL)
- [x] David Silver: Lectures 1-2 (MDPs & Value Functions)

**Project (Lane A):**
- [x] Project structure setup
- [x] Core data models (Job, GPU, Cluster)
- [x] Event system foundation
- [x] Design environment spec (state → action → reward)
- [x] Implement basic simulator (discrete event simulator)
- [x] MDP environment interface (Gymnasium)
- [x] Validate with random policy / trivial heuristic

🎯 **End-of-week goal:** Can describe simulator as an MDP and identify state, action, reward, and transition.

### Week 2: Baselines + Policy Gradient Intuition
**Learning (Lane B):**
- [ ] HF Course: Unit 2 (Q-Learning) → Unit 3 (Deep Q-Learning) → Unit 4 (Policy Gradient)
- [ ] Spinning Up: REINFORCE & A2C concepts
- [ ] David Silver: Lectures 3-4 (DP, Monte Carlo, Temporal Difference)

**Project (Lane A):**
- [ ] Implement heuristic baselines (FIFO, SJF, power-aware)
- [ ] Instrument metrics: GPU utilization, job queue latency, energy
- [ ] Build RL agent skeleton (policy net, env interface, reward logic)

🎯 **End-of-week goal:** Understand why PPO needs advantages (A = Q − V) and have solid baseline numbers to beat.

### Week 3: PPO Deep Dive + Novelty
**Learning (Lane B):**
- [ ] HF Course: Unit 6 (Actor-Critic) → Unit 8 (PPO)
- [ ] Spinning Up: PPO section + implementation notes
- [ ] David Silver: Lecture 5 (Policy Gradient Wrap-up)

**Project (Lane A):**
- [ ] Train PPO on environment; tune hyperparameters (clip ε, entropy coef, value coef)
- [ ] Choose and implement one novelty:
  - Hierarchical policy (job-class → GPU allocation), or
  - Graph-encoded state (GNN over GPU nodes), or
  - Adaptive reward weights (based on load/energy)
- [ ] Evaluate vs baselines under steady/bursty/power-capped scenarios
- [ ] Build Streamlit dashboard + README + demo GIF

🎯 **End-of-week goal:** PPO beats baselines on ≥ 2 KPIs and can clearly explain how novelty changes agent behavior.

## Deliverables

1. Working environment + trained PPO agent outperforming baselines
2. Own derivations of Bellman + PPO loss (in notes)
3. One novelty implemented and evaluated wxith plots
4. Public GitHub repo + dashboard + demo clip

