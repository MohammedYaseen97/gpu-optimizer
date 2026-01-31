"""
Plot PPO training curves from the CSV produced by `scripts/train_ppo.py`.

Usage:
  PYTHONPATH=. python scripts/plot_training_curve.py runs/<run>.csv --out runs/<run>.png
"""

import argparse
import csv
from typing import List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("csv_path", type=str)
    p.add_argument("--out", type=str, default=None, help="Output PNG path (optional).")
    p.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Simple moving average window over updates.",
    )
    return p.parse_args()


def moving_average(xs: List[float], window: int) -> List[float]:
    if window <= 1:
        return xs
    out: List[float] = []
    for i in range(len(xs)):
        j0 = max(0, i - window + 1)
        chunk = xs[j0 : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def main() -> None:
    args = parse_args()

    updates: List[int] = []
    timesteps: List[int] = []
    mean_returns: List[float] = []
    eval_mean_returns: List[float] = []
    losses: List[float] = []

    with open(args.csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            updates.append(int(row["update"]))
            timesteps.append(int(float(row["timesteps"])))
            mean_returns.append(float(row["mean_ep_return"]))
            if "eval_mean_return" in row and row["eval_mean_return"] not in ("", "nan", "NaN"):
                eval_mean_returns.append(float(row["eval_mean_return"]))
            else:
                eval_mean_returns.append(float("nan"))
            losses.append(float(row["loss"]))

    # Matplotlib import here so you can still inspect CSV without it.
    import matplotlib.pyplot as plt  # type: ignore

    # Prefer eval curve when available (long episodes may not finish inside rollouts).
    plot_returns = eval_mean_returns if any(x == x for x in eval_mean_returns) else mean_returns
    smoothed_returns = moving_average(plot_returns, args.smooth_window)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(timesteps, plot_returns, label="return", alpha=0.35)
    ax1.plot(timesteps, smoothed_returns, label=f"return (MA{args.smooth_window})")
    ax1.set_xlabel("Timesteps")
    ax1.set_ylabel("Mean episodic return (during rollouts)")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(timesteps, losses, color="tab:red", alpha=0.25, label="loss")
    ax2.set_ylabel("PPO loss (scalar)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=160)
        print(f"Saved plot to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

