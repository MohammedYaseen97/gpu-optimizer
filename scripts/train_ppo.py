"""
Train PPO on the GPU scheduling environment.

This script wires together:
- `SchedulerEnv`
- `PPOAgent`
- Training configuration

You will fill in the PPOAgent internals in `src/agents/ppo_agent.py`.
"""

import argparse
import numpy as np
import torch

from src.environment.scheduler_env import SchedulerEnv
from src.agents.ppo_agent import PPOAgent, PPOConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--num-steps", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--normalize-advantages", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Seeding
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Build env
    env = SchedulerEnv()

    # Build PPOConfig from args
    config = PPOConfig(
        total_timesteps=args.total_timesteps,
        num_steps=args.num_steps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        learning_rate=args.learning_rate,
        num_minibatches=args.num_minibatches,
        update_epochs=args.update_epochs,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        normalize_advantages=args.normalize_advantages,
    )

    agent = PPOAgent(env, config)

    # Train PPO
    agent.train()

    # Simple post-training evaluation
    num_eval_episodes = 10
    rewards = []
    for _ in range(num_eval_episodes):
        total_reward = agent.run_episode()
        rewards.append(total_reward)

    avg_reward = float(np.mean(rewards))
    print(f"[PPO] Avg total reward over {num_eval_episodes} eval episodes: {avg_reward:.3f}")


if __name__ == "__main__":
    main()

