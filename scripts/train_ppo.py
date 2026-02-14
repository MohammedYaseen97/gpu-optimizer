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
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import torch

from src.environment.scheduler_env import SchedulerEnv
from src.agents.ppo_agent import PPOAgent, PPOConfig


@dataclass(frozen=True)
class PPOPreset:
    """Hardcoded PPO hyperparameters (no CLI knobs)."""

    # PPO / rollout
    num_steps: int
    gamma: float
    gae_lambda: float
    num_minibatches: int
    update_epochs: int
    clip_coef: float
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    normalize_advantages: bool

    # LR schedule
    learning_rate: float
    lr_schedule: str
    lr_exp_end_frac: float

    # Stabilizers / diagnostics
    entropy_jump_clip: float
    target_kl: float | None

    # Seeding bases (keep fixed for reproducibility)
    train_seed_base: int
    eval_seed_base: int


# ---- Presets ---------------------------------------------------------------
# You can change DEFAULT_PRESET in code if you want to compare behaviors,
# without adding new CLI knobs.
PRESETS: dict[str, PPOPreset] = {
    "baseline_stable": PPOPreset(
        num_steps=2048,
        gamma=0.99,
        gae_lambda=0.95,
        num_minibatches=8,
        update_epochs=2,
        clip_coef=0.1,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        normalize_advantages=True,
        learning_rate=1e-4,
        lr_schedule="exp",
        lr_exp_end_frac=0.2,
        entropy_jump_clip=0.5,
        target_kl=0.02,
        train_seed_base=1_000_000,
        eval_seed_base=2_000_000,
    ),
}

DEFAULT_PRESET = "baseline_stable"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Training (experiment-level)
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--preset",
        type=str,
        default=DEFAULT_PRESET,
        choices=sorted(PRESETS.keys()),
        help="PPO hyperparameter preset (keeps CLI manageable).",
    )
    # Ablations / research toggles (defaults match the final config)
    parser.add_argument(
        "--policy-arch",
        type=str,
        default="attn",
        choices=["attn", "mlp"],
        help="Policy architecture ablation: attn (cross-attn) vs mlp (baseline).",
    )
    parser.add_argument(
        "--lookahead-mode",
        type=str,
        default="per_gpu",
        choices=["off", "per_gpu", "sorted", "cdf"],
        help=(
            "Lookahead representation (keeps observation shape unchanged). "
            "'off' zeros the lookahead segment."
        ),
    )

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

    # Evaluation config (experiment-level)
    # Default to the same eval sizing we used in the earlier multi-block runs.
    parser.add_argument("--num-eval-episodes", type=int, default=60)
    parser.add_argument("--eval-interval-updates", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument(
        "--eval-num-blocks",
        type=int,
        default=3,
        help="Number of fixed-seed eval blocks (each of length --eval-episodes).",
    )
    parser.add_argument(
        "--eval-deterministic",
        type=int,
        default=1,
        help="1 = greedy argmax eval (stable curves), 0 = stochastic sampling",
    )
    parser.add_argument(
        "--report-seed-offset",
        type=int,
        default=10_000_000,
        help="Offset added to VAL eval seed base for disjoint REPORT evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preset = PRESETS[str(args.preset)]

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
        # Reward scales (fixed; remove CLI knobs)
        progress_scale=1.0,
        time_pressure_scale=1.0,
        noop_penalty_scale=0.05,
        early_finish_bonus_scale=1.0,
        deadline_miss_penalty_scale=1.0,
        lookahead_mode=str(args.lookahead_mode),
    )

    # Build PPOConfig from experiment args + hardcoded preset
    config = PPOConfig(
        policy_arch=str(args.policy_arch),
        total_timesteps=args.total_timesteps,
        num_steps=preset.num_steps,
        gamma=preset.gamma,
        gae_lambda=preset.gae_lambda,
        learning_rate=preset.learning_rate,
        num_minibatches=preset.num_minibatches,
        update_epochs=preset.update_epochs,
        clip_coef=preset.clip_coef,
        ent_coef=preset.ent_coef,
        vf_coef=preset.vf_coef,
        max_grad_norm=preset.max_grad_norm,
        normalize_advantages=preset.normalize_advantages,
        lr_schedule=preset.lr_schedule,
        lr_exp_end_frac=preset.lr_exp_end_frac,
        entropy_jump_clip=preset.entropy_jump_clip,
        target_kl=preset.target_kl,
        eval_num_blocks=int(args.eval_num_blocks),
        seed=int(args.seed),
        train_seed_base=int(preset.train_seed_base),
        eval_seed_base=int(preset.eval_seed_base),
    )

    agent = PPOAgent(env, config)

    log_dir = "runs"
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = str(args.lookahead_mode).strip()
    tag_mode = "pergpu" if mode == "per_gpu" else mode
    lookahead_tag = f"lookahead_{tag_mode}"
    run_name = (
        f"ppo_{args.preset}_arch{args.policy_arch}_{lookahead_tag}_"
        f"{args.jobs_per_episode}jobs_w{args.max_queue_size}_seed{args.seed}_{ts}"
    )
    csv_path = os.path.join(log_dir, f"{run_name}.csv")

    # Train PPO
    agent.train(
        log_csv_path=csv_path,
        eval_interval_updates=int(args.eval_interval_updates),
        eval_episodes=int(args.eval_episodes),
        eval_deterministic=bool(int(args.eval_deterministic)),
    )

    # Best-checkpoint reporting: if we saved a best-eval checkpoint during training,
    # evaluate THAT policy rather than the final iterate.
    best_ckpt = getattr(agent, "best_checkpoint_path", None)
    if best_ckpt:
        agent.load_checkpoint(best_ckpt)
        print(
            f"[PPO] Loaded best checkpoint from update={getattr(agent, 'best_eval_update', -1)} "
            f"eval_mean_return={getattr(agent, 'best_eval_mean_return', float('nan')):.3f}"
        )

    # Post-training evaluation (deterministic episodes).
    #
    # We compute TWO numbers:
    # - VAL: same fixed seeds used for periodic eval (comparable to the training curve; used for selection)
    # - REPORT: disjoint fixed seeds (final reported number; reduces "tuned-on-eval" bias)
    num_eval_episodes = int(args.num_eval_episodes)
    val_rewards = []
    val_invalid_fracs = []
    val_noop_fracs = []
    val_seed_base = int(preset.eval_seed_base) + int(args.seed)
    for ep in range(num_eval_episodes):
        stats = agent.run_episode_with_stats(
            seed=val_seed_base + ep, deterministic=bool(int(args.eval_deterministic))
        )
        val_rewards.append(stats["total_reward"])
        val_invalid_fracs.append(stats["invalid_frac"])
        val_noop_fracs.append(stats["noop_frac"])

    val_avg_reward = float(np.mean(val_rewards))
    val_avg_invalid = float(np.mean(val_invalid_fracs))
    val_avg_noop = float(np.mean(val_noop_fracs))
    print(f"[PPO][VAL] Avg total reward over {num_eval_episodes} eval episodes: {val_avg_reward:.3f}")
    print(f"[PPO][VAL] invalid_frac={val_avg_invalid:.3f}  noop_frac={val_avg_noop:.3f}  seed_base={val_seed_base}")

    report_rewards = []
    report_invalid_fracs = []
    report_noop_fracs = []
    report_seed_base = int(preset.eval_seed_base) + int(args.seed) + int(args.report_seed_offset)
    for ep in range(num_eval_episodes):
        stats = agent.run_episode_with_stats(
            seed=report_seed_base + ep, deterministic=bool(int(args.eval_deterministic))
        )
        report_rewards.append(stats["total_reward"])
        report_invalid_fracs.append(stats["invalid_frac"])
        report_noop_fracs.append(stats["noop_frac"])

    report_avg_reward = float(np.mean(report_rewards))
    report_avg_invalid = float(np.mean(report_invalid_fracs))
    report_avg_noop = float(np.mean(report_noop_fracs))
    print(f"[PPO][REPORT] Avg total reward over {num_eval_episodes} eval episodes: {report_avg_reward:.3f}")
    print(
        f"[PPO][REPORT] invalid_frac={report_avg_invalid:.3f}  noop_frac={report_avg_noop:.3f}  seed_base={report_seed_base}"
    )
    print(f"[PPO] Training curve CSV: {csv_path}")


if __name__ == "__main__":
    main()

