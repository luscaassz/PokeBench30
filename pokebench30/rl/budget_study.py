from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from typing import Any

import matplotlib.pyplot as plt
import yaml

from pokebench30 import __version__
from pokebench30.rl.model_io import load_model_metadata, validate_model_metadata
from pokebench30.teams.fixed_pool import (
    DEFAULT_POOL_ROOT,
    load_pool_manifest,
    verify_pool,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "rl_baseline.yaml"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "budget_results"


@dataclass(frozen=True)
class BudgetEvaluationResult:
    budget: int
    representation: str
    seed: int
    opponent: str
    source: str
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


def load_rl_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


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


def _training_summary(
    model_dir: Path,
    *,
    budget: int,
    source: str,
    last_window: int = 20,
) -> dict[str, Any]:
    metadata = load_model_metadata(model_dir / "maskable_ppo.zip")
    rows = _monitor_rows(model_dir)
    rewards = [float(row["r"]) for row in rows if row.get("r") not in (None, "")]
    lengths = [float(row["l"]) for row in rows if row.get("l") not in (None, "")]

    return {
        "budget": int(budget),
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
        "source": source,
        "model_dir": str(model_dir),
    }


def _load_eval_result(
    *,
    budget: int,
    representation: str,
    seed: int,
    opponent: str,
    source: str,
    model_dir: Path,
    evaluation_dir: Path,
) -> BudgetEvaluationResult:
    path = evaluation_dir / "controlled_summary.csv"
    rows = _read_csv_rows(path)
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one controlled summary row at {path}, got {len(rows)}."
        )
    row = rows[0]
    return BudgetEvaluationResult(
        budget=int(budget),
        representation=representation,
        seed=int(seed),
        opponent=opponent,
        source=source,
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


def _validate_baseline_manifest(
    source_root: Path,
    *,
    baseline_budget: int,
    generation: int,
    ubrrs: list[str],
    seeds: list[int],
    training_opponent: str,
    eval_opponents: list[str],
    repeats: int,
    n_steps: int,
    batch_size: int,
    learning_rate: float,
    current_pool_manifest: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = source_root / "experiment_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Baseline experiment manifest not found: {manifest_path}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []

    def expect(label: str, observed: Any, expected: Any) -> None:
        if observed != expected:
            errors.append(f"{label}: observed={observed!r}, expected={expected!r}")

    expect("generation", int(manifest.get("generation", -1)), int(generation))
    expect("timesteps", int(manifest.get("timesteps", -1)), int(baseline_budget))
    expect("training_opponent", manifest.get("training_opponent"), training_opponent)
    expect("evaluation_repeats", int(manifest.get("evaluation_repeats", -1)), int(repeats))
    expect("n_steps", int(manifest.get("n_steps", -1)), int(n_steps))
    expect("batch_size", int(manifest.get("batch_size", -1)), int(batch_size))

    observed_lr = float(manifest.get("learning_rate", float("nan")))
    if abs(observed_lr - float(learning_rate)) > 1e-12:
        errors.append(
            f"learning_rate: observed={observed_lr!r}, expected={float(learning_rate)!r}"
        )

    if sorted(manifest.get("representations", [])) != sorted(ubrrs):
        errors.append(
            "representations: "
            f"observed={manifest.get('representations')!r}, expected={ubrrs!r}"
        )
    if sorted(int(x) for x in manifest.get("seeds", [])) != sorted(int(x) for x in seeds):
        errors.append(
            f"seeds: observed={manifest.get('seeds')!r}, expected={seeds!r}"
        )
    if sorted(manifest.get("evaluation_opponents", [])) != sorted(eval_opponents):
        errors.append(
            "evaluation_opponents: "
            f"observed={manifest.get('evaluation_opponents')!r}, "
            f"expected={eval_opponents!r}"
        )

    source_pool = manifest.get("fixed_team_pool", {})
    current_hash = current_pool_manifest.get("config_sha256")
    source_hash = source_pool.get("config_sha256")
    if source_hash != current_hash:
        errors.append(
            "fixed-team config hash mismatch: "
            f"baseline={source_hash}, current={current_hash}"
        )

    source_commit = source_pool.get("showdown_commit")
    current_commit = current_pool_manifest.get("showdown_commit")
    if source_commit != current_commit:
        errors.append(
            "Showdown commit mismatch: "
            f"baseline={source_commit}, current={current_commit}"
        )

    if errors:
        raise ValueError(
            "The supplied v0.5.2 baseline experiment is not directly "
            "comparable to this v0.5.3 budget study:\n- "
            + "\n- ".join(errors)
        )

    return manifest


def import_baseline_experiment(
    source_root: Path,
    *,
    baseline_budget: int,
    generation: int,
    ubrrs: list[str],
    seeds: list[int],
    training_opponent: str,
    eval_opponents: list[str],
    repeats: int,
    n_steps: int,
    batch_size: int,
    learning_rate: float,
    current_pool_manifest: dict[str, Any],
    snapshot_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], list[BudgetEvaluationResult]]:
    source_root = Path(source_root).resolve()

    manifest = _validate_baseline_manifest(
        source_root,
        baseline_budget=baseline_budget,
        generation=generation,
        ubrrs=ubrrs,
        seeds=seeds,
        training_opponent=training_opponent,
        eval_opponents=eval_opponents,
        repeats=repeats,
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        current_pool_manifest=current_pool_manifest,
    )

    aggregate = source_root / "aggregate"
    training_path = aggregate / "training_summary.csv"
    results_path = aggregate / "per_model_results.csv"

    if not training_path.exists() or not results_path.exists():
        raise FileNotFoundError(
            "Baseline experiment is missing aggregate/training_summary.csv "
            "or aggregate/per_model_results.csv."
        )

    raw_training = _read_csv_rows(training_path)
    training_rows: list[dict[str, Any]] = []
    allowed_ubr = set(ubrrs)
    allowed_seed = {int(x) for x in seeds}

    for row in raw_training:
        representation = row["representation"]
        seed = int(row["seed"])
        if representation not in allowed_ubr or seed not in allowed_seed:
            continue
        training_rows.append({
            "budget": int(baseline_budget),
            "representation": representation,
            "seed": seed,
            "generation": int(row["generation"]),
            "training_opponent": row["training_opponent"],
            "requested_timesteps": int(row["requested_timesteps"]),
            "episodes": int(row["episodes"]),
            "mean_reward": float(row["mean_reward"]) if row["mean_reward"] else "",
            "last_window_reward": (
                float(row["last_window_reward"]) if row["last_window_reward"] else ""
            ),
            "mean_episode_length": (
                float(row["mean_episode_length"]) if row["mean_episode_length"] else ""
            ),
            "source": "imported_v0.5.2",
            "model_dir": row["model_dir"],
        })

    expected_training = len(ubrrs) * len(seeds)
    if len(training_rows) != expected_training:
        raise RuntimeError(
            f"Expected {expected_training} imported training rows, "
            f"found {len(training_rows)}."
        )

    raw_results = _read_csv_rows(results_path)
    eval_results: list[BudgetEvaluationResult] = []
    allowed_opponents = set(eval_opponents)

    for row in raw_results:
        representation = row["representation"]
        seed = int(row["seed"])
        opponent = row["opponent"]
        if (
            representation not in allowed_ubr
            or seed not in allowed_seed
            or opponent not in allowed_opponents
        ):
            continue
        eval_results.append(
            BudgetEvaluationResult(
                budget=int(baseline_budget),
                representation=representation,
                seed=seed,
                opponent=opponent,
                source="imported_v0.5.2",
                model_dir=row["model_dir"],
                evaluation_dir=row["evaluation_dir"],
                battles=int(row["battles"]),
                mirror_cycles=int(row["mirror_cycles"]),
                wins_rl=int(row["wins_rl"]),
                wins_opponent=int(row["wins_opponent"]),
                draws=int(row["draws"]),
                raw_win_rate_rl=float(row["raw_win_rate_rl"]),
                mirror_score_rl=float(row["mirror_score_rl"]),
                team_assignment_gap=float(row["team_assignment_gap"]),
            )
        )

    expected_evals = len(ubrrs) * len(seeds) * len(eval_opponents)
    if len(eval_results) != expected_evals:
        raise RuntimeError(
            f"Expected {expected_evals} imported evaluation rows, "
            f"found {len(eval_results)}."
        )

    if snapshot_dir is not None:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            source_root / "experiment_manifest.json",
            snapshot_dir / "source_experiment_manifest.json",
        )
        shutil.copy2(training_path, snapshot_dir / "training_summary.csv")
        shutil.copy2(results_path, snapshot_dir / "per_model_results.csv")
        source_info = {
            "source_experiment_dir": str(source_root),
            "imported_budget": int(baseline_budget),
            "source_manifest": manifest,
            "note": (
                "Only aggregate baseline records are snapshotted here. "
                "Original models/evaluation directories remain in the source "
                "v0.5.2 experiment."
            ),
        }
        (snapshot_dir / "import_info.json").write_text(
            json.dumps(source_info, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return training_rows, eval_results


def _aggregate(
    results: list[BudgetEvaluationResult],
    out_dir: Path,
) -> None:
    result_rows = [asdict(row) for row in results]
    result_fields = list(asdict(results[0]).keys()) if results else [
        "budget", "representation", "seed", "opponent", "source",
        "model_dir", "evaluation_dir", "battles", "mirror_cycles",
        "wins_rl", "wins_opponent", "draws", "raw_win_rate_rl",
        "mirror_score_rl", "team_assignment_gap",
    ]
    _write_csv(out_dir / "per_model_budget_results.csv", result_rows, result_fields)

    grouped: dict[tuple[int, str, str], list[BudgetEvaluationResult]] = {}
    for result in results:
        grouped.setdefault(
            (result.budget, result.representation, result.opponent),
            [],
        ).append(result)

    aggregate_rows: list[dict[str, Any]] = []
    for (budget, representation, opponent), rows in sorted(grouped.items()):
        scores = [row.mirror_score_rl for row in rows]
        win_rates = [row.raw_win_rate_rl for row in rows]
        aggregate_rows.append({
            "budget": budget,
            "representation": representation,
            "opponent": opponent,
            "n_seeds": len(rows),
            "mean_mirror_score": statistics.fmean(scores),
            "sample_sd_mirror_score": (
                statistics.stdev(scores) if len(scores) > 1 else 0.0
            ),
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
            "budget", "representation", "opponent", "n_seeds",
            "mean_mirror_score", "sample_sd_mirror_score",
            "min_mirror_score", "max_mirror_score",
            "mean_raw_win_rate", "total_wins_rl", "total_wins_opponent",
            "total_draws", "total_battles",
        ],
    )

    by_key = {
        (row.budget, row.seed, row.opponent, row.representation): row
        for row in results
    }
    budgets = sorted({row.budget for row in results})
    seeds = sorted({row.seed for row in results})
    opponents = sorted({row.opponent for row in results})

    # Within each budget: UBR 1.2 - UBR 1.1.
    representation_pairs: list[dict[str, Any]] = []
    for budget in budgets:
        for opponent in opponents:
            for seed in seeds:
                old = by_key.get((budget, seed, opponent, "ubr-1.1"))
                new = by_key.get((budget, seed, opponent, "ubr-1.2"))
                if old is None or new is None:
                    continue
                representation_pairs.append({
                    "budget": budget,
                    "seed": seed,
                    "opponent": opponent,
                    "ubr_1_1_mirror_score": old.mirror_score_rl,
                    "ubr_1_2_mirror_score": new.mirror_score_rl,
                    "difference_1_2_minus_1_1": (
                        new.mirror_score_rl - old.mirror_score_rl
                    ),
                })

    _write_csv(
        out_dir / "representation_paired_differences.csv",
        representation_pairs,
        [
            "budget", "seed", "opponent",
            "ubr_1_1_mirror_score", "ubr_1_2_mirror_score",
            "difference_1_2_minus_1_1",
        ],
    )

    representation_summary: list[dict[str, Any]] = []
    for budget in budgets:
        for opponent in opponents:
            rows = [
                row for row in representation_pairs
                if row["budget"] == budget and row["opponent"] == opponent
            ]
            if not rows:
                continue
            diffs = [float(row["difference_1_2_minus_1_1"]) for row in rows]
            representation_summary.append({
                "budget": budget,
                "opponent": opponent,
                "paired_seeds": len(rows),
                "mean_ubr_1_1": statistics.fmean(
                    float(row["ubr_1_1_mirror_score"]) for row in rows
                ),
                "mean_ubr_1_2": statistics.fmean(
                    float(row["ubr_1_2_mirror_score"]) for row in rows
                ),
                "mean_difference_1_2_minus_1_1": statistics.fmean(diffs),
                "sample_sd_difference": (
                    statistics.stdev(diffs) if len(diffs) > 1 else 0.0
                ),
            })

    _write_csv(
        out_dir / "representation_comparison_by_budget.csv",
        representation_summary,
        [
            "budget", "opponent", "paired_seeds", "mean_ubr_1_1",
            "mean_ubr_1_2", "mean_difference_1_2_minus_1_1",
            "sample_sd_difference",
        ],
    )

    # Across budgets: performance gain for the same representation/seed/opponent.
    budget_pairs: list[dict[str, Any]] = []
    for index, from_budget in enumerate(budgets):
        for to_budget in budgets[index + 1:]:
            for representation in ("ubr-1.1", "ubr-1.2"):
                for opponent in opponents:
                    for seed in seeds:
                        before = by_key.get(
                            (from_budget, seed, opponent, representation)
                        )
                        after = by_key.get(
                            (to_budget, seed, opponent, representation)
                        )
                        if before is None or after is None:
                            continue
                        budget_pairs.append({
                            "from_budget": from_budget,
                            "to_budget": to_budget,
                            "representation": representation,
                            "seed": seed,
                            "opponent": opponent,
                            "score_from": before.mirror_score_rl,
                            "score_to": after.mirror_score_rl,
                            "gain": after.mirror_score_rl - before.mirror_score_rl,
                        })

    _write_csv(
        out_dir / "budget_gain_by_seed.csv",
        budget_pairs,
        [
            "from_budget", "to_budget", "representation", "seed",
            "opponent", "score_from", "score_to", "gain",
        ],
    )

    budget_summary: list[dict[str, Any]] = []
    combos = sorted({
        (
            row["from_budget"],
            row["to_budget"],
            row["representation"],
            row["opponent"],
        )
        for row in budget_pairs
    })
    for from_budget, to_budget, representation, opponent in combos:
        rows = [
            row for row in budget_pairs
            if (
                row["from_budget"] == from_budget
                and row["to_budget"] == to_budget
                and row["representation"] == representation
                and row["opponent"] == opponent
            )
        ]
        gains = [float(row["gain"]) for row in rows]
        budget_summary.append({
            "from_budget": from_budget,
            "to_budget": to_budget,
            "representation": representation,
            "opponent": opponent,
            "paired_seeds": len(rows),
            "mean_score_from": statistics.fmean(
                float(row["score_from"]) for row in rows
            ),
            "mean_score_to": statistics.fmean(
                float(row["score_to"]) for row in rows
            ),
            "mean_gain": statistics.fmean(gains),
            "sample_sd_gain": statistics.stdev(gains) if len(gains) > 1 else 0.0,
        })

    _write_csv(
        out_dir / "budget_comparison.csv",
        budget_summary,
        [
            "from_budget", "to_budget", "representation", "opponent",
            "paired_seeds", "mean_score_from", "mean_score_to",
            "mean_gain", "sample_sd_gain",
        ],
    )

    # Difference-in-differences style descriptive summary:
    # does UBR 1.2 benefit more from a larger training budget than UBR 1.1?
    interaction_rows: list[dict[str, Any]] = []
    budget_pairs_available = sorted({
        (row["from_budget"], row["to_budget"])
        for row in budget_summary
    })
    for from_budget, to_budget in budget_pairs_available:
        for opponent in opponents:
            old = next(
                (
                    row for row in budget_summary
                    if row["from_budget"] == from_budget
                    and row["to_budget"] == to_budget
                    and row["representation"] == "ubr-1.1"
                    and row["opponent"] == opponent
                ),
                None,
            )
            new = next(
                (
                    row for row in budget_summary
                    if row["from_budget"] == from_budget
                    and row["to_budget"] == to_budget
                    and row["representation"] == "ubr-1.2"
                    and row["opponent"] == opponent
                ),
                None,
            )
            if old is None or new is None:
                continue
            interaction_rows.append({
                "from_budget": from_budget,
                "to_budget": to_budget,
                "opponent": opponent,
                "mean_gain_ubr_1_1": old["mean_gain"],
                "mean_gain_ubr_1_2": new["mean_gain"],
                "difference_of_gains_ubr_1_2_minus_1_1": (
                    float(new["mean_gain"]) - float(old["mean_gain"])
                ),
            })

    _write_csv(
        out_dir / "budget_representation_interaction.csv",
        interaction_rows,
        [
            "from_budget", "to_budget", "opponent",
            "mean_gain_ubr_1_1", "mean_gain_ubr_1_2",
            "difference_of_gains_ubr_1_2_minus_1_1",
        ],
    )

    figures = out_dir / "figures"
    figures.mkdir(exist_ok=True)

    # Mean budget curves with seed-level SD.
    for opponent in opponents:
        fig, ax = plt.subplots(figsize=(8, 5))
        for representation in ("ubr-1.1", "ubr-1.2"):
            rows = [
                row for row in aggregate_rows
                if (
                    row["representation"] == representation
                    and row["opponent"] == opponent
                )
            ]
            rows = sorted(rows, key=lambda row: int(row["budget"]))
            if not rows:
                continue
            x = [int(row["budget"]) for row in rows]
            y = [float(row["mean_mirror_score"]) for row in rows]
            yerr = [float(row["sample_sd_mirror_score"]) for row in rows]
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker="o",
                capsize=4,
                label=representation,
            )
        ax.axhline(0.5, linestyle="--", linewidth=1)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Training timesteps")
        ax.set_ylabel("Mean normalized mirrored score (± seed SD)")
        ax.set_title(f"Training-budget study vs {opponent}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            figures / f"budget_curve_vs_{opponent}.png",
            dpi=160,
        )
        plt.close(fig)

        # UBR 1.2 - UBR 1.1 mean difference by budget.
        rows = [
            row for row in representation_summary
            if row["opponent"] == opponent
        ]
        rows = sorted(rows, key=lambda row: int(row["budget"]))
        if rows:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(
                [int(row["budget"]) for row in rows],
                [
                    float(row["mean_difference_1_2_minus_1_1"])
                    for row in rows
                ],
                marker="o",
            )
            ax.axhline(0.0, linestyle="--", linewidth=1)
            ax.set_xlabel("Training timesteps")
            ax.set_ylabel("Mean UBR 1.2 − UBR 1.1 mirrored score")
            ax.set_title(f"Representation advantage vs {opponent}")
            fig.tight_layout()
            fig.savefig(
                figures / f"representation_advantage_vs_{opponent}.png",
                dpi=160,
            )
            plt.close(fig)


def _same_study_manifest(
    existing: dict[str, Any],
    requested: dict[str, Any],
) -> bool:
    keys = [
        "generation",
        "representations",
        "seeds",
        "budgets",
        "baseline_budget",
        "training_opponent",
        "evaluation_opponents",
        "evaluation_repeats",
        "n_steps",
        "batch_size",
        "learning_rate",
        "baseline_experiment_dir",
        "fixed_team_pool",
    ]
    return all(existing.get(key) == requested.get(key) for key in keys)


def run_budget_study(args: argparse.Namespace) -> Path:
    pool_root = Path(args.pool_root).resolve()
    problems = verify_pool(pool_root)
    if problems:
        raise RuntimeError(
            "Fixed-team pool failed integrity check. Prepare it before the "
            "budget study:\n- " + "\n- ".join(problems)
        )
    pool_manifest = load_pool_manifest(pool_root)

    study_root = (
        Path(args.study_dir).resolve()
        if args.study_dir
        else Path(args.output_root).resolve()
        / f"budget_gen{args.generation}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    study_root.mkdir(parents=True, exist_ok=True)
    aggregate_root = study_root / "aggregate"
    aggregate_root.mkdir(exist_ok=True)

    requested_manifest = {
        "project": "PokeBench30",
        "version": __version__,
        "experiment_type": "training_budget_study",
        "generation": int(args.generation),
        "representations": list(args.ubrs),
        "seeds": [int(seed) for seed in args.seeds],
        "budgets": [int(budget) for budget in args.budgets],
        "baseline_budget": int(args.baseline_budget),
        "training_opponent": args.train_opponent,
        "evaluation_opponents": list(args.eval_opponents),
        "evaluation_repeats": int(args.repeats),
        "n_steps": int(args.n_steps),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "baseline_experiment_dir": (
            str(Path(args.baseline_experiment_dir).resolve())
            if args.baseline_experiment_dir
            else None
        ),
        "fixed_team_pool": {
            "config_sha256": pool_manifest.get("config_sha256"),
            "showdown_commit": pool_manifest.get("showdown_commit"),
            "showdown_dirty": pool_manifest.get("showdown_dirty"),
        },
        "scientific_design": [
            "Each budget is trained from scratch; 50k is not warm-started from 10k.",
            "Algorithm, reward, network, opponents and evaluation protocol stay fixed.",
            "The key question is whether extra training budget changes UBR 1.1 vs UBR 1.2 performance.",
            "Pokemon Showdown battle-event RNG is not common-random-number paired across models.",
        ],
    }

    manifest_path = study_root / "study_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not _same_study_manifest(existing, requested_manifest):
            raise ValueError(
                "The selected --study-dir contains a different budget-study "
                "configuration. Use a new directory instead of mixing runs."
            )
    else:
        manifest_path.write_text(
            json.dumps(requested_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    training_rows: list[dict[str, Any]] = []
    eval_results: list[BudgetEvaluationResult] = []

    imported_budget: int | None = None
    if args.baseline_experiment_dir:
        if int(args.baseline_budget) not in args.budgets:
            raise ValueError(
                "--baseline-budget must be included in --budgets when "
                "--baseline-experiment-dir is supplied."
            )
        imported_budget = int(args.baseline_budget)
        baseline_training, baseline_results = import_baseline_experiment(
            Path(args.baseline_experiment_dir),
            baseline_budget=imported_budget,
            generation=args.generation,
            ubrrs=list(args.ubrs),
            seeds=[int(seed) for seed in args.seeds],
            training_opponent=args.train_opponent,
            eval_opponents=list(args.eval_opponents),
            repeats=args.repeats,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            current_pool_manifest=pool_manifest,
            snapshot_dir=study_root / "imported_baseline",
        )
        training_rows.extend(baseline_training)
        eval_results.extend(baseline_results)
        print(
            f"[IMPORT] Reused {len(baseline_training)} models and "
            f"{len(baseline_results)} evaluations from budget={imported_budget}."
        )

    budgets_to_train = [
        int(budget)
        for budget in args.budgets
        if int(budget) != imported_budget
    ]

    total_models = len(budgets_to_train) * len(args.ubrs) * len(args.seeds)
    total_evals = total_models * len(args.eval_opponents)

    print(f"[INFO] Budget-study directory: {study_root}")
    print(f"[INFO] Budgets: {list(args.budgets)}")
    print(f"[INFO] Models to train now: {total_models}")
    print(f"[INFO] Controlled evaluations to run now: {total_evals}")

    model_counter = 0

    for budget in budgets_to_train:
        budget_root = study_root / f"budget_{budget}"
        models_root = budget_root / "models"
        eval_root = budget_root / "evaluations"
        models_root.mkdir(parents=True, exist_ok=True)
        eval_root.mkdir(parents=True, exist_ok=True)

        for ubr in args.ubrs:
            for seed in args.seeds:
                model_counter += 1
                run_name = (
                    f"t{int(budget)}_{_slug_ubr(ubr)}_seed_{int(seed)}"
                )
                model_dir = models_root / run_name
                model_file = model_dir / "maskable_ppo.zip"

                print("\n" + "=" * 76)
                print(
                    f"MODEL {model_counter}/{total_models}: "
                    f"budget={budget} | {ubr} | seed={seed} | "
                    f"Gen {args.generation}"
                )
                print("=" * 76)

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
                    if int(metadata.total_timesteps) != int(budget):
                        raise ValueError(
                            f"Resume model timestep mismatch at {model_dir}: "
                            f"metadata={metadata.total_timesteps}, "
                            f"expected={budget}."
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
                        print(
                            f"[RESUME] Removing incomplete model directory: "
                            f"{model_dir}"
                        )
                        shutil.rmtree(model_dir)

                    command = [
                        sys.executable,
                        str(PROJECT_ROOT / "train_rl.py"),
                        "--generation", str(args.generation),
                        "--opponent", args.train_opponent,
                        "--ubr", ubr,
                        "--timesteps", str(budget),
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

                training_rows.append(
                    _training_summary(
                        model_dir,
                        budget=budget,
                        source="trained_v0.5.3",
                    )
                )

                for opponent in args.eval_opponents:
                    eval_name = f"{run_name}_vs_{opponent}"
                    eval_dir = eval_root / eval_name
                    summary_path = eval_dir / "controlled_summary.csv"

                    if summary_path.exists():
                        print(
                            f"[RESUME] Existing evaluation accepted: {eval_dir}"
                        )
                    else:
                        if eval_dir.exists():
                            print(
                                "[RESUME] Removing incomplete evaluation "
                                f"directory: {eval_dir}"
                            )
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
                            budget=budget,
                            representation=ubr,
                            seed=int(seed),
                            opponent=opponent,
                            source="trained_v0.5.3",
                            model_dir=model_dir,
                            evaluation_dir=eval_dir,
                        )
                    )

                # Partial/resumable outputs.
                _write_csv(
                    aggregate_root / "training_summary.csv",
                    sorted(
                        training_rows,
                        key=lambda row: (
                            int(row["budget"]),
                            row["representation"],
                            int(row["seed"]),
                        ),
                    ),
                    [
                        "budget", "representation", "seed", "generation",
                        "training_opponent", "requested_timesteps", "episodes",
                        "mean_reward", "last_window_reward",
                        "mean_episode_length", "source", "model_dir",
                    ],
                )
                _aggregate(eval_results, aggregate_root)

    # Final aggregate also covers the special case where all requested budgets
    # were imported / already complete.
    _write_csv(
        aggregate_root / "training_summary.csv",
        sorted(
            training_rows,
            key=lambda row: (
                int(row["budget"]),
                row["representation"],
                int(row["seed"]),
            ),
        ),
        [
            "budget", "representation", "seed", "generation",
            "training_opponent", "requested_timesteps", "episodes",
            "mean_reward", "last_window_reward",
            "mean_episode_length", "source", "model_dir",
        ],
    )
    _aggregate(eval_results, aggregate_root)

    print("\n=== PokéBench30 v0.5.3 training-budget study complete ===")
    print(f"Study: {study_root}")
    print(f"Training rows: {len(training_rows)}")
    print(f"Evaluation rows: {len(eval_results)}")
    print(f"Aggregate summary: {aggregate_root / 'aggregate_summary.csv'}")
    print(f"Budget comparison: {aggregate_root / 'budget_comparison.csv'}")
    print(
        "Representation × budget interaction: "
        f"{aggregate_root / 'budget_representation_interaction.csv'}"
    )
    return study_root


def build_parser() -> argparse.ArgumentParser:
    config = load_rl_config(DEFAULT_CONFIG)
    algo = config["algorithm"]
    block = config.get("budget_study", {})

    parser = argparse.ArgumentParser(
        description=(
            "PokéBench30 v0.5.3 training-budget study: compare UBR 1.1 and "
            "UBR 1.2 across 10k/50k (or custom) MaskablePPO budgets."
        )
    )
    parser.add_argument("--generation", type=int, choices=[1, 4, 9], default=9)
    parser.add_argument(
        "--ubrs",
        nargs="+",
        choices=["ubr-1.1", "ubr-1.2"],
        default=block.get("representations", ["ubr-1.1", "ubr-1.2"]),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=block.get("seeds", [42, 123, 456, 789, 2026]),
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        default=block.get("budgets", [10000, 50000]),
    )
    parser.add_argument(
        "--baseline-budget",
        type=int,
        default=int(block.get("baseline_budget", 10000)),
    )
    parser.add_argument(
        "--baseline-experiment-dir",
        default=None,
        help=(
            "Optional completed v0.5.2 multi-seed directory to import as the "
            "10k baseline, avoiding retraining/re-evaluating that budget."
        ),
    )
    parser.add_argument(
        "--train-opponent",
        default=block.get("training_opponent", "type_aware"),
    )
    parser.add_argument(
        "--eval-opponents",
        nargs="+",
        default=block.get(
            "evaluation_opponents",
            ["random", "type_aware"],
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=int(block.get("repeats_per_leg", 10)),
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=int(algo.get("n_steps", 2048)),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(algo.get("batch_size", 128)),
    )
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
    parser.add_argument("--output-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument(
        "--study-dir",
        default=None,
        help="Existing/new study directory. Reusing it resumes completed runs.",
    )
    parser.add_argument(
        "--keep-action-logs",
        action="store_true",
        help="Keep per-action evaluation CSVs (larger, slower).",
    )
    parser.add_argument(
        "--make-eval-plots",
        action="store_true",
        help="Generate per-evaluation plots as well as aggregate budget plots.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not args.budgets:
        raise ValueError("At least one budget is required.")
    if any(int(budget) <= 0 for budget in args.budgets):
        raise ValueError("All budgets must be > 0.")
    if len(set(int(x) for x in args.budgets)) != len(args.budgets):
        raise ValueError("Budgets must be unique.")
    args.budgets = sorted(int(x) for x in args.budgets)

    if not args.seeds:
        raise ValueError("At least one seed is required.")
    if len(set(int(x) for x in args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be unique.")

    if args.repeats <= 0:
        raise ValueError("--repeats must be > 0.")

    run_budget_study(args)


if __name__ == "__main__":
    main()
