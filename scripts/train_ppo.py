"""
Train PPO on the GPU scheduling environment.

This script wires together:
- `SchedulerEnv`
- `PPOAgent`
- Training configuration

You will fill in the PPOAgent internals in `src/agents/ppo_agent.py`.
"""

import argparse
import os
from datetime import datetime
import numpy as np
import torch

from src.environment.scheduler_env import SchedulerEnv
from src.agents.ppo_agent import PPOAgent, PPOConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # Training
    parser.add_argument("--total-timesteps", type=int, default=200_000)
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

    # Env config
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--max-queue-size", type=int, default=50)
    parser.add_argument("--max-episode-steps", type=int, default=2000)
    parser.add_argument("--max-duration", type=float, default=10000.0)
    parser.add_argument("--jobs-per-episode", type=int, default=400)
    parser.add_argument(
        "--arrival-mode",
        type=str,
        default="bursty",
        choices=["all_at_zero", "bursty"],
    )
    parser.add_argument("--arrival-span", type=float, default=None)
    parser.add_argument("--horizon-factor", type=float, default=1.3)

    # Logging
    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--num-eval-episodes", type=int, default=10)
    parser.add_argument("--eval-interval-updates", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Seeding
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Build env
    env = SchedulerEnv(
        num_gpus=args.num_gpus,
        max_queue_size=args.max_queue_size,
        max_episode_steps=args.max_episode_steps,
        max_duration=args.max_duration,
        jobs_per_episode=args.jobs_per_episode,
        arrival_mode=args.arrival_mode,
        arrival_span=args.arrival_span,
        horizon_factor=args.horizon_factor,
    )

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

    os.makedirs(args.log_dir, exist_ok=True)
    run_name = args.run_name
    if run_name is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"ppo_{args.jobs_per_episode}jobs_w{args.max_queue_size}_seed{args.seed}_{ts}"
    csv_path = os.path.join(args.log_dir, f"{run_name}.csv")

    # Train PPO
    agent.train(
        log_csv_path=csv_path,
        eval_interval_updates=int(args.eval_interval_updates),
        eval_episodes=int(args.eval_episodes),
    )

    # Simple post-training evaluation
    num_eval_episodes = int(args.num_eval_episodes)
    rewards = []
    for _ in range(num_eval_episodes):
        total_reward = agent.run_episode()
        rewards.append(total_reward)

    avg_reward = float(np.mean(rewards))
    print(f"[PPO] Avg total reward over {num_eval_episodes} eval episodes: {avg_reward:.3f}")
    print(f"[PPO] Training curve CSV: {csv_path}")


if __name__ == "__main__":
    main()

