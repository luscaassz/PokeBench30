from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import statistics
import subprocess
import shutil
import sys
from typing import Any

import matplotlib.pyplot as plt
import yaml

from pokebench30 import __version__
from pokebench30.rl.model_io import load_model_metadata, validate_model_metadata
from pokebench30.teams.fixed_pool import DEFAULT_POOL_ROOT, load_pool_manifest, verify_pool


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "rl_baseline.yaml"
DEFAULT_EXPERIMENTS_ROOT = PROJECT_ROOT / "multiseed_results"


def load_rl_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class EvaluationResult:
    representation: str
    seed: int
    opponent: str
    model_dir: str
    evaluation_dir: str
    battles: int
    mirror_cycles: int
    wins_rl: int
    wins_opponent: int
    draws: int
    raw_win_rate_rl: float
    mirror_score_rl: float
    team_assignment_gap: float


def _slug_ubr(version: str) -> str:
    return version.replace("-", "_").replace(".", "_")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _monitor_rows(model_dir: Path) -> list[dict[str, str]]:
    candidates = [
        model_dir / "monitor.csv.monitor.csv",
        model_dir / "monitor.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            first = handle.readline()
            if not first.startswith("#"):
                handle.seek(0)
            return list(csv.DictReader(handle))
    return []


def _training_summary(model_dir: Path, last_window: int = 20) -> dict[str, Any]:
    metadata = load_model_metadata(model_dir / "maskable_ppo.zip")
    rows = _monitor_rows(model_dir)
    rewards = [float(row["r"]) for row in rows if row.get("r") not in (None, "")]
    lengths = [float(row["l"]) for row in rows if row.get("l") not in (None, "")]

    return {
        "representation": metadata.observation_schema_version,
        "seed": metadata.seed,
        "generation": metadata.training_generation,
        "training_opponent": metadata.training_opponent,
        "requested_timesteps": metadata.total_timesteps,
        "episodes": len(rows),
        "mean_reward": statistics.fmean(rewards) if rewards else "",
        "last_window_reward": (
            statistics.fmean(rewards[-last_window:]) if rewards else ""
        ),
        "mean_episode_length": statistics.fmean(lengths) if lengths else "",
        "model_dir": str(model_dir),
    }


def _load_eval_result(
    *,
    representation: str,
    seed: int,
    opponent: str,
    model_dir: Path,
    evaluation_dir: Path,
) -> EvaluationResult:
    path = evaluation_dir / "controlled_summary.csv"
    rows = _read_csv_rows(path)
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one controlled summary row at {path}, got {len(rows)}."
        )
    row = rows[0]
    return EvaluationResult(
        representation=representation,
        seed=int(seed),
        opponent=opponent,
        model_dir=str(model_dir),
        evaluation_dir=str(evaluation_dir),
        battles=int(row["battles"]),
        mirror_cycles=int(row["mirror_cycles"]),
        wins_rl=int(row["wins_agent_1"]),
        wins_opponent=int(row["wins_agent_2"]),
        draws=int(row["draws"]),
        raw_win_rate_rl=float(row["raw_win_rate_agent_1"]),
        mirror_score_rl=float(row["normalized_mirror_score_agent_1"]),
        team_assignment_gap=float(row["agent_1_team_assignment_gap"]),
    )


def _aggregate(results: list[EvaluationResult], out_dir: Path) -> None:
    result_rows = [asdict(row) for row in results]
    _write_csv(
        out_dir / "per_model_results.csv",
        result_rows,
        list(asdict(results[0]).keys()) if results else [
            "representation", "seed", "opponent", "model_dir", "evaluation_dir",
            "battles", "mirror_cycles", "wins_rl", "wins_opponent", "draws",
            "raw_win_rate_rl", "mirror_score_rl", "team_assignment_gap",
        ],
    )

    grouped: dict[tuple[str, str], list[EvaluationResult]] = {}
    for result in results:
        grouped.setdefault((result.representation, result.opponent), []).append(result)

    aggregate_rows: list[dict[str, Any]] = []
    for (representation, opponent), rows in sorted(grouped.items()):
        scores = [row.mirror_score_rl for row in rows]
        win_rates = [row.raw_win_rate_rl for row in rows]
        aggregate_rows.append({
            "representation": representation,
            "opponent": opponent,
            "n_seeds": len(rows),
            "mean_mirror_score": statistics.fmean(scores),
            "sample_sd_mirror_score": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "min_mirror_score": min(scores),
            "max_mirror_score": max(scores),
            "mean_raw_win_rate": statistics.fmean(win_rates),
            "total_wins_rl": sum(row.wins_rl for row in rows),
            "total_wins_opponent": sum(row.wins_opponent for row in rows),
            "total_draws": sum(row.draws for row in rows),
            "total_battles": sum(row.battles for row in rows),
        })

    _write_csv(
        out_dir / "aggregate_summary.csv",
        aggregate_rows,
        [
            "representation", "opponent", "n_seeds", "mean_mirror_score",
            "sample_sd_mirror_score", "min_mirror_score", "max_mirror_score",
            "mean_raw_win_rate", "total_wins_rl", "total_wins_opponent",
            "total_draws", "total_battles",
        ],
    )

    by_key = {
        (row.seed, row.opponent, row.representation): row
        for row in results
    }
    seeds = sorted({row.seed for row in results})
    opponents = sorted({row.opponent for row in results})

    paired_rows: list[dict[str, Any]] = []
    for opponent in opponents:
        for seed in seeds:
            old = by_key.get((seed, opponent, "ubr-1.1"))
            new = by_key.get((seed, opponent, "ubr-1.2"))
            if old is None or new is None:
                continue
            paired_rows.append({
                "seed": seed,
                "opponent": opponent,
                "ubr_1_1_mirror_score": old.mirror_score_rl,
                "ubr_1_2_mirror_score": new.mirror_score_rl,
                "difference_1_2_minus_1_1": new.mirror_score_rl - old.mirror_score_rl,
            })

    _write_csv(
        out_dir / "paired_differences.csv",
        paired_rows,
        [
            "seed", "opponent", "ubr_1_1_mirror_score",
            "ubr_1_2_mirror_score", "difference_1_2_minus_1_1",
        ],
    )

    comparison_rows: list[dict[str, Any]] = []
    for opponent in opponents:
        opponent_rows = [row for row in paired_rows if row["opponent"] == opponent]
        if not opponent_rows:
            continue
        diffs = [float(row["difference_1_2_minus_1_1"]) for row in opponent_rows]
        comparison_rows.append({
            "opponent": opponent,
            "paired_seeds": len(opponent_rows),
            "mean_ubr_1_1": statistics.fmean(
                float(row["ubr_1_1_mirror_score"]) for row in opponent_rows
            ),
            "mean_ubr_1_2": statistics.fmean(
                float(row["ubr_1_2_mirror_score"]) for row in opponent_rows
            ),
            "mean_difference_1_2_minus_1_1": statistics.fmean(diffs),
            "sample_sd_difference": statistics.stdev(diffs) if len(diffs) > 1 else 0.0,
        })

    _write_csv(
        out_dir / "representation_comparison.csv",
        comparison_rows,
        [
            "opponent", "paired_seeds", "mean_ubr_1_1", "mean_ubr_1_2",
            "mean_difference_1_2_minus_1_1", "sample_sd_difference",
        ],
    )

    figures = out_dir / "figures"
    figures.mkdir(exist_ok=True)
    for opponent in opponents:
        fig, ax = plt.subplots(figsize=(8, 5))
        for seed in seeds:
            old = by_key.get((seed, opponent, "ubr-1.1"))
            new = by_key.get((seed, opponent, "ubr-1.2"))
            if old is None or new is None:
                continue
            ax.plot(
                ["UBR 1.1", "UBR 1.2"],
                [old.mirror_score_rl, new.mirror_score_rl],
                marker="o",
                alpha=0.65,
                label=f"seed {seed}",
            )
        ax.axhline(0.5, linestyle="--", linewidth=1)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Normalized mirrored score (RL)")
        ax.set_title(f"UBR ablation vs {opponent}")
        if len(seeds) <= 8:
            ax.legend()
        fig.tight_layout()
        fig.savefig(figures / f"ubr_ablation_vs_{opponent}.png", dpi=160)
        plt.close(fig)


def _same_manifest(existing: dict[str, Any], requested: dict[str, Any]) -> bool:
    keys = [
        "generation", "representations", "seeds", "timesteps",
        "training_opponent", "evaluation_opponents", "evaluation_repeats",
        "n_steps", "batch_size", "learning_rate",
    ]
    return all(existing.get(key) == requested.get(key) for key in keys)


def run_multiseed(args: argparse.Namespace) -> Path:
    pool_root = Path(args.pool_root).resolve()
    problems = verify_pool(pool_root)
    if problems:
        raise RuntimeError(
            "Fixed-team pool failed integrity check. Prepare it before the "
            "multi-seed experiment:\n- " + "\n- ".join(problems)
        )
    pool_manifest = load_pool_manifest(pool_root)

    experiment_root = (
        Path(args.experiment_dir).resolve()
        if args.experiment_dir
        else Path(args.output_root).resolve()
        / f"multiseed_gen{args.generation}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    experiment_root.mkdir(parents=True, exist_ok=True)
    models_root = experiment_root / "models"
    eval_root = experiment_root / "evaluations"
    aggregate_root = experiment_root / "aggregate"
    models_root.mkdir(exist_ok=True)
    eval_root.mkdir(exist_ok=True)
    aggregate_root.mkdir(exist_ok=True)

    config = load_rl_config(Path(args.config))
    requested_manifest = {
        "project": "PokeBench30",
        "version": __version__,
        "experiment_type": "multi_seed_ubr_ablation",
        "generation": int(args.generation),
        "representations": list(args.ubrs),
        "seeds": [int(seed) for seed in args.seeds],
        "timesteps": int(args.timesteps),
        "training_opponent": args.train_opponent,
        "evaluation_opponents": list(args.eval_opponents),
        "evaluation_repeats": int(args.repeats),
        "n_steps": int(args.n_steps),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "fixed_team_pool": {
            "config_sha256": pool_manifest.get("config_sha256"),
            "showdown_commit": pool_manifest.get("showdown_commit"),
            "showdown_dirty": pool_manifest.get("showdown_dirty"),
        },
        "important_limitation": (
            "The PPO seed is controlled, but Pokemon Showdown battle-event RNG "
            "is not paired/common-random-number controlled. Seed-wise UBR comparisons "
            "are repeated model runs, not identical stochastic trajectories."
        ),
    }

    manifest_path = experiment_root / "experiment_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not _same_manifest(existing, requested_manifest):
            raise ValueError(
                "The selected --experiment-dir contains a different experiment "
                "configuration. Use a new directory instead of mixing runs."
            )
    else:
        manifest_path.write_text(
            json.dumps(requested_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"[INFO] Multi-seed experiment directory: {experiment_root}")
    print(f"[INFO] Models planned: {len(args.ubrs) * len(args.seeds)}")
    print(
        f"[INFO] Controlled evaluations planned: "
        f"{len(args.ubrs) * len(args.seeds) * len(args.eval_opponents)}"
    )

    training_rows: list[dict[str, Any]] = []
    eval_results: list[EvaluationResult] = []

    total_models = len(args.ubrs) * len(args.seeds)
    model_counter = 0

    for ubr in args.ubrs:
        for seed in args.seeds:
            model_counter += 1
            run_name = f"{_slug_ubr(ubr)}_seed_{int(seed)}"
            model_dir = models_root / run_name
            model_file = model_dir / "maskable_ppo.zip"

            print("\n" + "=" * 72)
            print(
                f"MODEL {model_counter}/{total_models}: {ubr} | seed={seed} | "
                f"Gen {args.generation}"
            )
            print("=" * 72)

            if model_file.exists() and (model_dir / "metadata.json").exists():
                metadata = load_model_metadata(model_file)
                validate_model_metadata(
                    metadata,
                    generation=args.generation,
                    expected_representation_version=ubr,
                )
                if int(metadata.seed) != int(seed):
                    raise ValueError(
                        f"Resume model seed mismatch at {model_dir}: "
                        f"metadata={metadata.seed}, expected={seed}."
                    )
                if int(metadata.total_timesteps) != int(args.timesteps):
                    raise ValueError(
                        f"Resume model timestep mismatch at {model_dir}: "
                        f"metadata={metadata.total_timesteps}, expected={args.timesteps}."
                    )
                if metadata.training_opponent != args.train_opponent:
                    raise ValueError(
                        f"Resume model opponent mismatch at {model_dir}: "
                        f"metadata={metadata.training_opponent}, "
                        f"expected={args.train_opponent}."
                    )
                print(f"[RESUME] Existing model accepted: {model_file}")
            else:
                if model_dir.exists():
                    print(f"[RESUME] Removing incomplete model directory: {model_dir}")
                    shutil.rmtree(model_dir)
                command = [
                    sys.executable,
                    str(PROJECT_ROOT / "train_rl.py"),
                    "--generation", str(args.generation),
                    "--opponent", args.train_opponent,
                    "--ubr", ubr,
                    "--timesteps", str(args.timesteps),
                    "--seed", str(int(seed)),
                    "--device", args.device,
                    "--learning-rate", str(args.learning_rate),
                    "--n-steps", str(args.n_steps),
                    "--batch-size", str(args.batch_size),
                    "--checkpoint-freq", str(args.checkpoint_freq),
                    "--log-level", str(args.log_level),
                    "--config", str(args.config),
                    "--output-root", str(models_root),
                    "--run-name", run_name,
                ]
                subprocess.run(command, cwd=PROJECT_ROOT, check=True)

            training_rows.append(_training_summary(model_dir))

            for opponent in args.eval_opponents:
                eval_name = f"{run_name}_vs_{opponent}"
                eval_dir = eval_root / eval_name
                summary_path = eval_dir / "controlled_summary.csv"

                if summary_path.exists():
                    print(f"[RESUME] Existing evaluation accepted: {eval_dir}")
                else:
                    if eval_dir.exists():
                        print(f"[RESUME] Removing incomplete evaluation directory: {eval_dir}")
                        shutil.rmtree(eval_dir)
                    command = [
                        sys.executable,
                        str(PROJECT_ROOT / "run_rl_controlled.py"),
                        "--model", str(model_file),
                        "--generation", str(args.generation),
                        "--opponent", opponent,
                        "--repeats", str(args.repeats),
                        "--concurrency", str(args.concurrency),
                        "--pool-root", str(pool_root),
                        "--output-root", str(eval_root),
                        "--run-name", eval_name,
                    ]
                    if not args.keep_action_logs:
                        command.append("--no-action-log")
                    if not args.make_eval_plots:
                        command.append("--no-plots")
                    subprocess.run(command, cwd=PROJECT_ROOT, check=True)

                eval_results.append(
                    _load_eval_result(
                        representation=ubr,
                        seed=int(seed),
                        opponent=opponent,
                        model_dir=model_dir,
                        evaluation_dir=eval_dir,
                    )
                )

            # Continuously write partial results so an interrupted experiment
            # remains inspectable and can be resumed safely.
            _write_csv(
                aggregate_root / "training_summary.csv",
                training_rows,
                [
                    "representation", "seed", "generation", "training_opponent",
                    "requested_timesteps", "episodes", "mean_reward",
                    "last_window_reward", "mean_episode_length", "model_dir",
                ],
            )
            _aggregate(eval_results, aggregate_root)

    print("\n=== PokéBench30 v0.5.2 multi-seed experiment complete ===")
    print(f"Experiment: {experiment_root}")
    print(f"Models: {len(training_rows)}")
    print(f"Evaluations: {len(eval_results)}")
    print(f"Aggregate summary: {aggregate_root / 'aggregate_summary.csv'}")
    print(f"Paired UBR comparison: {aggregate_root / 'representation_comparison.csv'}")
    return experiment_root


def build_parser() -> argparse.ArgumentParser:
    config = load_rl_config(DEFAULT_CONFIG)
    algo = config["algorithm"]

    parser = argparse.ArgumentParser(
        description=(
            "Automate the PokéBench30 v0.5.2 multi-seed UBR 1.1 vs 1.2 "
            "MaskablePPO ablation."
        )
    )
    parser.add_argument("--generation", type=int, choices=[1, 4, 9], default=9)
    parser.add_argument(
        "--ubrs",
        nargs="+",
        choices=["ubr-1.1", "ubr-1.2"],
        default=["ubr-1.1", "ubr-1.2"],
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 456, 789, 2026],
    )
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--train-opponent", default="type_aware")
    parser.add_argument(
        "--eval-opponents",
        nargs="+",
        default=["random", "type_aware"],
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--n-steps", type=int, default=int(algo.get("n_steps", 2048)))
    parser.add_argument("--batch-size", type=int, default=int(algo.get("batch_size", 128)))
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=float(algo.get("learning_rate", 3e-4)),
    )
    parser.add_argument("--checkpoint-freq", type=int, default=10000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-level", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--pool-root", default=str(DEFAULT_POOL_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_EXPERIMENTS_ROOT))
    parser.add_argument(
        "--experiment-dir",
        default=None,
        help="Existing/new experiment directory. Reusing it resumes completed runs.",
    )
    parser.add_argument(
        "--keep-action-logs",
        action="store_true",
        help="Keep per-action evaluation CSVs (larger, slower).",
    )
    parser.add_argument(
        "--make-eval-plots",
        action="store_true",
        help="Generate the normal per-evaluation plots in addition to aggregate plots.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.timesteps <= 0:
        raise ValueError("--timesteps must be > 0")
    if args.repeats <= 0:
        raise ValueError("--repeats must be > 0")
    if not args.seeds:
        raise ValueError("At least one seed is required.")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be unique.")
    run_multiseed(args)


if __name__ == "__main__":
    main()
