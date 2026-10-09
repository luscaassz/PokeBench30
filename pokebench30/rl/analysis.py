from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_monitor(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        # SB3 Monitor writes one JSON comment line before the CSV header.
        first = handle.readline()
        if not first.startswith("#"):
            handle.seek(0)
        reader = csv.DictReader(handle)
        rows.extend(reader)
    return rows


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0:
        return values
    if window <= 1:
        return values.copy()
    output = np.empty_like(values, dtype=float)
    for i in range(len(values)):
        start = max(0, i - window + 1)
        output[i] = float(np.mean(values[start:i + 1]))
    return output


def analyze(run_dir: Path, window: int = 20) -> None:
    monitor_path = run_dir / "monitor.csv.monitor.csv"
    if not monitor_path.exists():
        # Depending on SB3 version / filename handling.
        monitor_path = run_dir / "monitor.csv"

    if not monitor_path.exists():
        raise FileNotFoundError(
            f"Could not find SB3 monitor file in {run_dir}."
        )

    rows = read_monitor(monitor_path)
    if not rows:
        raise ValueError("Monitor file contains no completed episodes.")

    rewards = np.asarray(
        [float(row["r"]) for row in rows],
        dtype=float,
    )
    lengths = np.asarray(
        [float(row["l"]) for row in rows],
        dtype=float,
    )
    episodes = np.arange(1, len(rows) + 1)

    figures = run_dir / "figures"
    figures.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        episodes,
        rolling_mean(rewards, window),
    )
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Completed episode")
    ax.set_ylabel(f"Reward (rolling mean, window={window})")
    ax.set_title("MaskablePPO training reward")
    fig.tight_layout()
    fig.savefig(
        figures / "training_reward.png",
        dpi=160,
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        episodes,
        rolling_mean(lengths, window),
    )
    ax.set_xlabel("Completed episode")
    ax.set_ylabel(f"Episode length (rolling mean, window={window})")
    ax.set_title("Training battle length")
    fig.tight_layout()
    fig.savefig(
        figures / "training_episode_length.png",
        dpi=160,
    )
    plt.close(fig)

    print(f"Episodes: {len(rows)}")
    print(f"Mean reward: {float(np.mean(rewards)):.4f}")
    print(f"Last-window reward: {float(np.mean(rewards[-window:])):.4f}")
    print(f"Mean episode length: {float(np.mean(lengths)):.2f}")
    print(f"Figures written to: {figures}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot PokéBench30 v0.5 training curves."
    )
    parser.add_argument("run_dir")
    parser.add_argument("--window", type=int, default=20)
    args = parser.parse_args()
    analyze(Path(args.run_dir).resolve(), window=args.window)


if __name__ == "__main__":
    main()
