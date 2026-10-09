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

import yaml

from pokebench30 import __version__
from pokebench30.evaluation.telemetry import (
    canonical_fixed_pool_catalog_rows,
    fixed_pool_catalog_from_manifest,
    generation_dex_catalog,
    validate_fixed_pool_catalog_rows,
)
from pokebench30.rl.model_io import load_model_metadata, validate_model_metadata
from pokebench30.teams.fixed_pool import (
    DEFAULT_POOL_ROOT,
    load_pool_manifest,
    verify_pool,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "transfer_telemetry.yaml"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "transfer_results"


@dataclass(frozen=True)
class TransferScore:
    training_generation: int
    evaluation_generation: int
    seed: int
    opponent: str
    battles: int
    mirror_cycles: int
    wins_rl: int
    wins_opponent: int
    draws: int
    mirror_score_rl: float
    raw_win_rate_rl: float
    team_assignment_gap: float
    evaluation_dir: str
    model_dir: str


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _canonical(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=False))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def validate_source_study(
    source_dir: Path,
    *,
    config: dict[str, Any],
    pool_manifest: dict[str, Any],
) -> dict[str, Any]:
    source_dir = Path(source_dir).resolve()
    manifest_path = source_dir / "study_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Source v0.6 study manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment_type") != "cross_generation_benchmark":
        raise ValueError(
            "--source-study-dir must point to a completed v0.6 cross-generation benchmark."
        )

    study = config["study"]
    errors: list[str] = []

    def expect(label: str, observed: Any, expected: Any) -> None:
        if _canonical(observed) != _canonical(expected):
            errors.append(f"{label}: observed={observed!r}, expected={expected!r}")

    expect("generations", manifest.get("generations"), study["generations"])
    expect("seeds", manifest.get("seeds"), study["seeds"])
    expect("representation", manifest.get("representation"), "ubr-1.2")
    expect(
        "training nominal timesteps",
        manifest.get("training", {}).get("nominal_total_timesteps"),
        study["source_checkpoint"],
    )
    expect(
        "fixed pool config hash",
        manifest.get("fixed_team_pool", {}).get("config_sha256"),
        pool_manifest.get("config_sha256"),
    )
    expect(
        "Showdown commit",
        manifest.get("fixed_team_pool", {}).get("showdown_commit"),
        pool_manifest.get("showdown_commit"),
    )

    if errors:
        raise ValueError("Incompatible v0.6 source study:\n- " + "\n- ".join(errors))

    for generation in study["generations"]:
        for seed in study["seeds"]:
            model_dir = source_model_dir(
                source_dir,
                training_generation=int(generation),
                seed=int(seed),
                checkpoint=int(study["source_checkpoint"]),
            )
            if not model_dir.exists():
                errors.append(f"Missing source model directory: {model_dir}")
                continue
            metadata = load_model_metadata(model_dir)
            validate_model_metadata(
                metadata,
                generation=int(generation),
                require_same_generation=True,
                expected_representation_version="ubr-1.2",
            )
            if int(metadata.seed) != int(seed):
                errors.append(f"Seed mismatch in {model_dir}: {metadata.seed} != {seed}")

    if errors:
        raise ValueError("Source model validation failed:\n- " + "\n- ".join(errors))

    return manifest


def source_model_dir(
    source_dir: Path,
    *,
    training_generation: int,
    seed: int,
    checkpoint: int,
) -> Path:
    return (
        Path(source_dir)
        / "runs"
        / f"gen{int(training_generation)}"
        / f"seed_{int(seed)}"
        / "checkpoints"
        / f"step_{int(checkpoint):06d}"
    )


def _evaluation_dir(
    study_root: Path,
    *,
    training_generation: int,
    evaluation_generation: int,
    seed: int,
    opponent: str,
) -> Path:
    return (
        study_root
        / "runs"
        / f"train_gen{training_generation}"
        / f"eval_gen{evaluation_generation}"
        / f"seed_{seed}"
        / f"vs_{opponent}"
        / "evaluation"
    )


def run_evaluation(
    *,
    source_dir: Path,
    study_root: Path,
    training_generation: int,
    evaluation_generation: int,
    seed: int,
    opponent: str,
    checkpoint: int,
    repeats: int,
    pool_root: Path,
    concurrency: int,
    pairs: list[str] | None = None,
) -> Path:
    model_dir = source_model_dir(
        source_dir,
        training_generation=training_generation,
        seed=seed,
        checkpoint=checkpoint,
    )
    parent = _evaluation_dir(
        study_root,
        training_generation=training_generation,
        evaluation_generation=evaluation_generation,
        seed=seed,
        opponent=opponent,
    ).parent
    parent.mkdir(parents=True, exist_ok=True)
    run_dir = parent / "evaluation"

    expected = [
        run_dir / "controlled_summary.csv",
        run_dir / "actions.csv",
        run_dir / "telemetry_behavior_summary.csv",
        run_dir / "telemetry_battle_dynamics.csv",
        run_dir / "telemetry_move_usage.csv",
        run_dir / "telemetry_move_feature_usage.csv",
        run_dir / "telemetry_team_pokemon.csv",
        run_dir / "telemetry_team_moves.csv",
    ]
    if all(path.exists() for path in expected):
        print(f"[RESUME] Existing telemetry evaluation accepted: {run_dir}")
        return run_dir

    if run_dir.exists():
        shutil.rmtree(run_dir)

    command = [
        sys.executable,
        str(PROJECT_ROOT / "run_rl_controlled.py"),
        "--model", str(model_dir),
        "--generation", str(evaluation_generation),
        "--opponent", opponent,
        "--repeats", str(repeats),
        "--concurrency", str(concurrency),
        "--pool-root", str(Path(pool_root).resolve()),
        "--output-root", str(parent),
        "--run-name", "evaluation",
        "--allow-cross-generation",
        "--no-plots",
    ]
    if pairs:
        command.extend(["--pairs", *pairs])
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return run_dir


def read_transfer_score(
    run_dir: Path,
    *,
    training_generation: int,
    evaluation_generation: int,
    seed: int,
    opponent: str,
    model_dir: Path,
) -> TransferScore:
    rows = _read_csv(run_dir / "controlled_summary.csv")
    if len(rows) != 1:
        raise RuntimeError(f"Expected one controlled summary row at {run_dir}; got {len(rows)}")
    row = rows[0]
    return TransferScore(
        training_generation=int(training_generation),
        evaluation_generation=int(evaluation_generation),
        seed=int(seed),
        opponent=opponent,
        battles=_int(row.get("battles")),
        mirror_cycles=_int(row.get("mirror_cycles")),
        wins_rl=_int(row.get("wins_agent_1")),
        wins_opponent=_int(row.get("wins_agent_2")),
        draws=_int(row.get("draws")),
        mirror_score_rl=_float(row.get("normalized_mirror_score_agent_1")),
        raw_win_rate_rl=_float(row.get("raw_win_rate_agent_1")),
        team_assignment_gap=_float(row.get("agent_1_team_assignment_gap")),
        evaluation_dir=str(run_dir),
        model_dir=str(model_dir),
    )


def _attach_context(
    rows: list[dict[str, str]],
    *,
    training_generation: int,
    evaluation_generation: int,
    seed: int,
    opponent: str,
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        output.append({
            "training_generation": int(training_generation),
            "evaluation_generation": int(evaluation_generation),
            "seed": int(seed),
            "opponent": opponent,
            **row,
        })
    return output


def collect_telemetry(
    run_dir: Path,
    *,
    training_generation: int,
    evaluation_generation: int,
    seed: int,
    opponent: str,
) -> dict[str, list[dict[str, Any]]]:
    files = {
        "behavior": "telemetry_behavior_summary.csv",
        "battle": "telemetry_battle_dynamics.csv",
        "move": "telemetry_move_usage.csv",
        "move_type": "telemetry_move_type_usage.csv",
        "move_category": "telemetry_move_category_usage.csv",
        "move_feature": "telemetry_move_feature_usage.csv",
        "pokemon_type": "telemetry_pokemon_type_exposure.csv",
        "team_pokemon": "telemetry_team_pokemon.csv",
        "team_move": "telemetry_team_moves.csv",
    }
    output: dict[str, list[dict[str, Any]]] = {}
    for key, name in files.items():
        path = run_dir / name
        rows = _read_csv(path) if path.exists() else []
        output[key] = _attach_context(
            rows,
            training_generation=training_generation,
            evaluation_generation=evaluation_generation,
            seed=seed,
            opponent=opponent,
        )
    return output


def _mean_sd(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def build_score_outputs(study_root: Path, scores: list[TransferScore]) -> None:
    aggregate = study_root / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    raw = [asdict(row) for row in scores]
    _write_csv(aggregate / "transfer_scores_by_seed.csv", raw)

    grouped: dict[tuple[int, int, str], list[TransferScore]] = {}
    for row in scores:
        grouped.setdefault(
            (row.training_generation, row.evaluation_generation, row.opponent), []
        ).append(row)

    matrix_rows: list[dict[str, Any]] = []
    for (train_gen, eval_gen, opponent), rows in sorted(grouped.items()):
        values = [row.mirror_score_rl for row in rows]
        mean, sd = _mean_sd(values)
        matrix_rows.append({
            "training_generation": train_gen,
            "evaluation_generation": eval_gen,
            "opponent": opponent,
            "n_seeds": len(rows),
            "mean_mirror_score": mean,
            "sample_sd_mirror_score": sd,
            "min_mirror_score": min(values),
            "max_mirror_score": max(values),
            "total_battles": sum(row.battles for row in rows),
        })
    _write_csv(aggregate / "transfer_matrix_summary.csv", matrix_rows)

    lookup = {
        (row.training_generation, row.evaluation_generation, row.seed, row.opponent): row
        for row in scores
    }
    gap_rows: list[dict[str, Any]] = []
    for row in scores:
        in_domain = lookup.get((
            row.evaluation_generation,
            row.evaluation_generation,
            row.seed,
            row.opponent,
        ))
        if in_domain is None:
            continue
        gap_rows.append({
            "training_generation": row.training_generation,
            "evaluation_generation": row.evaluation_generation,
            "seed": row.seed,
            "opponent": row.opponent,
            "transfer_score": row.mirror_score_rl,
            "in_domain_score": in_domain.mirror_score_rl,
            "transfer_gap": row.mirror_score_rl - in_domain.mirror_score_rl,
            "is_in_domain": row.training_generation == row.evaluation_generation,
        })
    _write_csv(aggregate / "transfer_gap_by_seed.csv", gap_rows)

    gap_groups: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
    for row in gap_rows:
        gap_groups.setdefault(
            (int(row["training_generation"]), int(row["evaluation_generation"]), str(row["opponent"])),
            [],
        ).append(row)
    gap_summary: list[dict[str, Any]] = []
    for (train_gen, eval_gen, opponent), rows in sorted(gap_groups.items()):
        gaps = [float(row["transfer_gap"]) for row in rows]
        mean, sd = _mean_sd(gaps)
        gap_summary.append({
            "training_generation": train_gen,
            "evaluation_generation": eval_gen,
            "opponent": opponent,
            "paired_seeds": len(rows),
            "mean_transfer_gap": mean,
            "sample_sd_transfer_gap": sd,
            "mean_transfer_score": statistics.fmean(float(row["transfer_score"]) for row in rows),
            "mean_in_domain_score": statistics.fmean(float(row["in_domain_score"]) for row in rows),
        })
    _write_csv(aggregate / "transfer_gap_summary.csv", gap_summary)

    generalization: list[dict[str, Any]] = []
    for train_gen in sorted({row.training_generation for row in scores}):
        for opponent in sorted({row.opponent for row in scores}):
            off = [
                row for row in scores
                if row.training_generation == train_gen
                and row.opponent == opponent
                and row.evaluation_generation != train_gen
            ]
            if not off:
                continue
            by_seed: dict[int, list[float]] = {}
            for row in off:
                by_seed.setdefault(row.seed, []).append(row.mirror_score_rl)
            seed_means = [statistics.fmean(values) for values in by_seed.values()]
            mean, sd = _mean_sd(seed_means)
            generalization.append({
                "training_generation": train_gen,
                "opponent": opponent,
                "n_seeds": len(seed_means),
                "mean_off_diagonal_score": mean,
                "sample_sd_off_diagonal_score": sd,
            })
    _write_csv(aggregate / "zero_shot_generalization_summary.csv", generalization)

    _write_score_heatmaps(aggregate / "figures", matrix_rows, gap_summary)


def _write_score_heatmaps(
    figures_dir: Path,
    matrix_rows: list[dict[str, Any]],
    gap_rows: list[dict[str, Any]],
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    figures_dir.mkdir(parents=True, exist_ok=True)
    generations = [1, 4, 9]
    for opponent in sorted({str(row["opponent"]) for row in matrix_rows}):
        matrix = np.full((3, 3), np.nan)
        for row in matrix_rows:
            if row["opponent"] != opponent:
                continue
            i = generations.index(int(row["training_generation"]))
            j = generations.index(int(row["evaluation_generation"]))
            matrix[i, j] = float(row["mean_mirror_score"])
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        image = ax.imshow(matrix, vmin=0.0, vmax=1.0)
        ax.set_xticks(range(3), [f"Gen {g}" for g in generations])
        ax.set_yticks(range(3), [f"Gen {g}" for g in generations])
        ax.set_xlabel("Evaluation generation")
        ax.set_ylabel("Training generation")
        ax.set_title(f"Zero-shot transfer score vs {opponent}")
        for i in range(3):
            for j in range(3):
                if not np.isnan(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center")
        fig.colorbar(image, ax=ax, label="Mean mirrored score")
        fig.tight_layout()
        fig.savefig(figures_dir / f"transfer_heatmap_vs_{opponent}.png", dpi=160)
        plt.close(fig)

        gap = np.full((3, 3), np.nan)
        for row in gap_rows:
            if row["opponent"] != opponent:
                continue
            i = generations.index(int(row["training_generation"]))
            j = generations.index(int(row["evaluation_generation"]))
            gap[i, j] = float(row["mean_transfer_gap"])
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        image = ax.imshow(gap, vmin=-1.0, vmax=1.0)
        ax.set_xticks(range(3), [f"Gen {g}" for g in generations])
        ax.set_yticks(range(3), [f"Gen {g}" for g in generations])
        ax.set_xlabel("Evaluation generation")
        ax.set_ylabel("Training generation")
        ax.set_title(f"Transfer gap vs in-domain model — {opponent}")
        for i in range(3):
            for j in range(3):
                if not np.isnan(gap[i, j]):
                    ax.text(j, i, f"{gap[i,j]:+.3f}", ha="center", va="center")
        fig.colorbar(image, ax=ax, label="Transfer score − in-domain score")
        fig.tight_layout()
        fig.savefig(figures_dir / f"transfer_gap_heatmap_vs_{opponent}.png", dpi=160)
        plt.close(fig)


def _aggregate_seed_numeric(
    rows: list[dict[str, Any]],
    *,
    group_keys: tuple[str, ...],
    numeric_fields: list[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row.get(key, "") for key in group_keys), []).append(row)
    output = []
    for key, members in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        base = {name: value for name, value in zip(group_keys, key)}
        base["n_seed_rows"] = len(members)
        for field in numeric_fields:
            values = []
            for member in members:
                try:
                    values.append(float(member.get(field, "")))
                except (TypeError, ValueError):
                    pass
            if values:
                base[f"mean_{field}"] = statistics.fmean(values)
                base[f"sd_{field}"] = statistics.stdev(values) if len(values) > 1 else 0.0
        output.append(base)
    return output


def build_telemetry_outputs(
    study_root: Path,
    telemetry: dict[str, list[dict[str, Any]]],
) -> None:
    aggregate = study_root / "aggregate" / "telemetry"
    aggregate.mkdir(parents=True, exist_ok=True)

    behavior = telemetry["behavior"]
    _write_csv(aggregate / "behavior_by_seed.csv", behavior)
    behavior_numeric = [
        key for key in behavior[0].keys()
        if key not in {"training_generation", "evaluation_generation", "seed", "opponent"}
        and key not in {"decisions", "move_actions", "switch_actions"}
    ] if behavior else []
    behavior_summary = _aggregate_seed_numeric(
        behavior,
        group_keys=("training_generation", "evaluation_generation", "opponent"),
        numeric_fields=behavior_numeric,
    )
    _write_csv(aggregate / "behavior_summary.csv", behavior_summary)

    battle = telemetry["battle"]
    _write_csv(aggregate / "battle_dynamics_by_seed.csv", battle)
    battle_numeric = [
        "raw_win_rate_rl", "mean_turns", "mean_action_requests_rl",
        "mean_moves_rl", "mean_switches_rl", "mean_voluntary_switches_rl",
        "mean_remaining_rl", "mean_remaining_opponent",
        "mean_opponent_reveal_fraction",
    ]
    battle_summary = _aggregate_seed_numeric(
        battle,
        group_keys=("training_generation", "evaluation_generation", "opponent"),
        numeric_fields=battle_numeric,
    )
    _write_csv(aggregate / "battle_dynamics_summary.csv", battle_summary)

    specs = {
        "move": ("move_id", ["selected_count", "available_decisions", "selection_given_available", "selected_share"]),
        "move_type": ("move_type", ["selected_count", "available_decisions", "selection_given_available", "selected_share"]),
        "move_category": ("move_category", ["selected_count", "available_decisions", "selection_given_available", "selected_share"]),
        "move_feature": ("feature", ["selected_count", "selected_rate_among_moves", "availability_decisions", "availability_rate", "selection_given_available"]),
        "pokemon_type": ("context", ["count", "share"]),
    }
    for key, (entity_field, numeric_fields) in specs.items():
        rows = telemetry[key]
        _write_csv(aggregate / f"{key}_by_seed.csv", rows)
        group = ["training_generation", "evaluation_generation", "opponent"]
        if key == "pokemon_type":
            group += ["context", "pokemon_type"]
        else:
            group.append(entity_field)
        summary = _aggregate_seed_numeric(
            rows,
            group_keys=tuple(group),
            numeric_fields=numeric_fields,
        )
        _write_csv(aggregate / f"{key}_summary.csv", summary)

    _write_fixed_pool_catalogs(aggregate, telemetry["team_pokemon"], telemetry["team_move"])
    _write_behavior_figures(aggregate / "figures", behavior_summary)


def _write_fixed_pool_catalogs(
    aggregate: Path,
    pokemon_rows: list[dict[str, Any]],
    move_rows: list[dict[str, Any]],
) -> None:
    # Structural pool ecology comes exclusively from immutable packed teams.
    # Post-battle telemetry is intentionally ignored here because transformations
    # and repeated/mirrored observations are behavioral data, not pool structure.
    pokemon, moves = canonical_fixed_pool_catalog_rows(
        pokemon_rows,
        move_rows,
    )

    validate_fixed_pool_catalog_rows(
        pokemon,
        moves,
        generations=(1, 4, 9),
        expected_pairs=10,
        expected_team_size=6,
    )

    # Preserve evaluation_generation for downstream v0.6.1 CSV compatibility.
    pokemon = [
        {
            **{
                k: v for k, v in row.items()
                if k not in {"training_generation", "seed", "opponent"}
            },
            "evaluation_generation": int(row["generation"]),
        }
        for row in pokemon
    ]
    moves = [
        {
            **{
                k: v for k, v in row.items()
                if k not in {"training_generation", "seed", "opponent"}
            },
            "evaluation_generation": int(row["generation"]),
        }
        for row in moves
    ]

    _write_csv(aggregate / "fixed_pool_pokemon_catalog.csv", pokemon)
    _write_csv(aggregate / "fixed_pool_move_catalog.csv", moves)

    type_counts: dict[tuple[int, str], int] = {}
    generation_type_total: dict[int, int] = {}
    for row in pokemon:
        generation = int(row["evaluation_generation"])
        for field in ("type_1", "type_2"):
            value = str(row.get(field, ""))
            if value:
                type_counts[(generation, value)] = type_counts.get((generation, value), 0) + 1
                generation_type_total[generation] = generation_type_total.get(generation, 0) + 1
    type_summary = [{
        "generation": gen,
        "pokemon_type": type_name,
        "type_slot_count": count,
        "share_of_type_slots": count / generation_type_total[gen],
    } for (gen, type_name), count in sorted(type_counts.items())]
    _write_csv(aggregate / "fixed_pool_pokemon_type_summary.csv", type_summary)

    for field, filename in (
        ("move_type", "fixed_pool_move_type_summary.csv"),
        ("move_category", "fixed_pool_move_category_summary.csv"),
        ("move_id", "fixed_pool_move_presence_summary.csv"),
    ):
        counts: dict[tuple[int, str], int] = {}
        totals: dict[int, int] = {}
        team_sets: dict[tuple[int, str], set[str]] = {}
        for row in moves:
            generation = int(row["evaluation_generation"])
            value = str(row.get(field, ""))
            if not value:
                continue
            counts[(generation, value)] = counts.get((generation, value), 0) + 1
            totals[generation] = totals.get(generation, 0) + 1
            team_sets.setdefault((generation, value), set()).add(str(row.get("team_id", "")))
        out = [{
            "generation": gen,
            field: value,
            "move_slot_count": count,
            "share_of_move_slots": count / totals[gen],
            "teams_with_value": len(team_sets[(gen, value)]),
        } for (gen, value), count in sorted(counts.items())]
        _write_csv(aggregate / filename, out)

    feature_names = [
        key.removeprefix("feature_")
        for key in (moves[0].keys() if moves else [])
        if key.startswith("feature_")
    ]
    feature_summary = []
    for generation in sorted({int(row["evaluation_generation"]) for row in moves}):
        gen_rows = [row for row in moves if int(row["evaluation_generation"]) == generation]
        for feature in feature_names:
            count = sum(str(row.get(f"feature_{feature}", "")).lower() in {"1", "true"} for row in gen_rows)
            feature_summary.append({
                "generation": generation,
                "feature": feature,
                "move_slot_count": count,
                "share_of_move_slots": count / len(gen_rows) if gen_rows else 0.0,
            })
    _write_csv(aggregate / "fixed_pool_move_feature_summary.csv", feature_summary)


def _write_behavior_figures(figures_dir: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    figures_dir.mkdir(parents=True, exist_ok=True)

    # In-domain behavior only: cleanest description of how independently
    # trained agents behave in their native generation.
    in_domain = [
        row for row in rows
        if int(row["training_generation"]) == int(row["evaluation_generation"])
    ]
    features = ["damaging", "status", "self_boost", "target_debuff", "healing", "hazard_set", "priority", "pivot_like"]
    for opponent in sorted({str(row["opponent"]) for row in in_domain}):
        selected_rows = [row for row in in_domain if row["opponent"] == opponent]
        if not selected_rows:
            continue
        selected_rows.sort(key=lambda row: int(row["evaluation_generation"]))
        fig, ax = plt.subplots(figsize=(10, 5.5))
        x = list(range(len(selected_rows)))
        width = 0.09
        for index, feature in enumerate(features):
            values = [float(row.get(f"mean_selection_given_available_{feature}", 0.0) or 0.0) for row in selected_rows]
            offsets = [value + (index - len(features)/2) * width for value in x]
            ax.bar(offsets, values, width=width, label=feature)
        ax.set_xticks(x, [f"Gen {int(row['evaluation_generation'])}" for row in selected_rows])
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Selection given feature available")
        ax.set_title(f"In-domain behavioral profile vs {opponent}")
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(figures_dir / f"in_domain_behavior_vs_{opponent}.png", dpi=160)
        plt.close(fig)


def _write_generation_dex_catalogs(
    aggregate: Path,
    generations: list[int],
) -> None:
    pokemon, moves = generation_dex_catalog(generations)
    _write_csv(aggregate / "generation_pokemon_dex_catalog.csv", pokemon)
    _write_csv(aggregate / "generation_move_dex_catalog.csv", moves)

    overview: list[dict[str, Any]] = []
    for generation in generations:
        mons_all = [row for row in pokemon if int(row["generation"]) == int(generation)]
        moves_all = [row for row in moves if int(row["generation"]) == int(generation)]
        # generation_dex_catalog already restricts Pokémon rows to canonical
        # base species introduced by the requested generation. Do not apply a
        # second `is_nonstandard` filter here: current-format labels such as
        # Past/Unobtainable are not historical-introduction metadata.
        mons = list(mons_all)
        gen_moves = [row for row in moves_all if not str(row.get("is_nonstandard", ""))]
        damaging = sum(str(row.get("feature_damaging", "")).lower() in {"1", "true"} for row in gen_moves)
        status = sum(str(row.get("feature_status", "")).lower() in {"1", "true"} for row in gen_moves)
        buffs = sum(str(row.get("feature_any_buff", "")).lower() in {"1", "true"} for row in gen_moves)
        debuffs = sum(str(row.get("feature_any_debuff", "")).lower() in {"1", "true"} for row in gen_moves)
        dual = sum(bool(str(row.get("type_2", ""))) for row in mons)
        overview.append({
            "generation": generation,
            "pokemon_catalog_scope": "canonical_base_species_introduced_by_generation",
            "pokemon_base_species_entries": len(mons),
            # Kept for backward-compatible CSV consumers; in hotfix1 these
            # fields refer to the filtered base-species catalog, not format
            # legality in Random Battles.
            "pokemon_entries_standard": len(mons),
            "pokemon_entries_all": len(mons_all),
            "move_entries_standard": len(gen_moves),
            "move_entries_all": len(moves_all),
            "dual_type_pokemon_entries": dual,
            "dual_type_share": dual / len(mons) if mons else 0.0,
            "damaging_moves": damaging,
            "damaging_move_share": damaging / len(gen_moves) if gen_moves else 0.0,
            "status_moves": status,
            "status_move_share": status / len(gen_moves) if gen_moves else 0.0,
            "moves_with_any_buff": buffs,
            "any_buff_share": buffs / len(gen_moves) if gen_moves else 0.0,
            "moves_with_any_debuff": debuffs,
            "any_debuff_share": debuffs / len(gen_moves) if gen_moves else 0.0,
        })
    _write_csv(aggregate / "generation_dex_overview.csv", overview)

    def categorical_summary(rows, value_field, filename):
        counts: dict[tuple[int, str], int] = {}
        totals: dict[int, int] = {}
        for row in rows:
            if str(row.get("is_nonstandard", "")):
                continue
            generation = int(row["generation"])
            value = str(row.get(value_field, ""))
            if not value:
                continue
            counts[(generation, value)] = counts.get((generation, value), 0) + 1
            totals[generation] = totals.get(generation, 0) + 1
        output = [{
            "generation": generation,
            value_field: value,
            "count": count,
            "share": count / totals[generation],
        } for (generation, value), count in sorted(counts.items())]
        _write_csv(aggregate / filename, output)

    categorical_summary(moves, "move_type", "generation_move_dex_type_summary.csv")
    categorical_summary(moves, "move_category", "generation_move_dex_category_summary.csv")

    type_rows = []
    counts: dict[tuple[int, str], int] = {}
    totals: dict[int, int] = {}
    for row in pokemon:
        generation = int(row["generation"])
        for field in ("type_1", "type_2"):
            value = str(row.get(field, ""))
            if value:
                counts[(generation, value)] = counts.get((generation, value), 0) + 1
                totals[generation] = totals.get(generation, 0) + 1
    for (generation, value), count in sorted(counts.items()):
        type_rows.append({
            "generation": generation,
            "pokemon_type": value,
            "type_slot_count": count,
            "share_of_type_slots": count / totals[generation],
        })
    _write_csv(aggregate / "generation_pokemon_dex_type_summary.csv", type_rows)

    feature_names = sorted({
        key.removeprefix("feature_")
        for row in moves for key in row
        if key.startswith("feature_")
    })
    feature_rows = []
    for generation in generations:
        gen_moves = [
            row for row in moves
            if int(row["generation"]) == int(generation)
            and not str(row.get("is_nonstandard", ""))
        ]
        for feature in feature_names:
            count = sum(str(row.get(f"feature_{feature}", "")).lower() in {"1", "true"} for row in gen_moves)
            feature_rows.append({
                "generation": generation,
                "feature": feature,
                "count": count,
                "share": count / len(gen_moves) if gen_moves else 0.0,
            })
    _write_csv(aggregate / "generation_move_dex_feature_summary.csv", feature_rows)


def build_manifest(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    source_manifest: dict[str, Any],
    pool_manifest: dict[str, Any],
    opponents: list[str],
) -> dict[str, Any]:
    study = config["study"]
    return {
        "project": "PokeBench30",
        "version": __version__,
        "experiment_type": "zero_shot_transfer_behavioral_telemetry",
        "source_study_dir": str(Path(args.source_study_dir).resolve()),
        "source_study_version": source_manifest.get("version"),
        "source_checkpoint": int(study["source_checkpoint"]),
        "training_generations": list(study["generations"]),
        "evaluation_generations": list(study["generations"]),
        "seeds": list(study["seeds"]),
        "opponents": opponents,
        "repeats_per_leg": int(study["repeats_per_leg"]),
        "team_pairs_per_generation": int(study["team_pairs_per_generation"]),
        "deterministic_policy": True,
        "fine_tuning": False,
        "weights_modified": False,
        "telemetry": {
            "action_level": True,
            "opportunity_adjusted": True,
            "move_multi_label_semantics": True,
            "fixed_pool_team_catalog": True,
            "compressed_rl_move_opportunities": True,
        },
        "fixed_team_pool": {
            "config_sha256": pool_manifest.get("config_sha256"),
            "showdown_commit": pool_manifest.get("showdown_commit"),
            "showdown_dirty": pool_manifest.get("showdown_dirty"),
        },
        "notes": [
            "All 3x3 train-generation/evaluation-generation cells are evaluated, including the diagonal, so behavioral telemetry is collected under one protocol.",
            "Transfer gap is zero-shot score minus the same-seed in-domain score for the evaluation generation.",
            "Availability-adjusted metrics distinguish what moves/mechanics were legal from what the policy selected.",
            "Fixed-pool catalogs characterize the benchmark's frozen team sample, not the entire generation metagame.",
        ],
    }


def _same_manifest(left: dict[str, Any], right: dict[str, Any]) -> bool:
    keys = [
        "source_study_dir", "source_checkpoint", "training_generations",
        "evaluation_generations", "seeds", "opponents", "repeats_per_leg",
        "team_pairs_per_generation", "deterministic_policy", "fine_tuning",
        "fixed_team_pool",
    ]
    return all(_canonical(left.get(key)) == _canonical(right.get(key)) for key in keys)


def run_study(args: argparse.Namespace) -> Path:
    config = load_config(Path(args.config))
    study = config["study"]
    source_dir = Path(args.source_study_dir).resolve()
    pool_root = Path(args.pool_root).resolve()

    problems = verify_pool(pool_root)
    if problems:
        raise RuntimeError("Fixed-team pool failed integrity check:\n- " + "\n- ".join(problems))
    pool_manifest = load_pool_manifest(pool_root)
    source_manifest = validate_source_study(
        source_dir,
        config=config,
        pool_manifest=pool_manifest,
    )

    opponents = list(study["primary_opponents"])
    if args.all_opponents:
        opponents += [
            value for value in study.get("supplementary_opponents", [])
            if value not in opponents
        ]
    if args.opponents:
        opponents = list(args.opponents)

    if args.smoke:
        study = dict(study)
        study["seeds"] = [int(study["seeds"][0])]
        opponents = ["max_damage"]
        args.repeats_override = 1
        args.pairs_override = ["pair01", "pair02", "pair03"]
    else:
        args.repeats_override = None
        args.pairs_override = None

    study_root = (
        Path(args.study_dir).resolve()
        if args.study_dir
        else Path(args.output_root).resolve()
        / f"transfer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    study_root.mkdir(parents=True, exist_ok=True)

    # Build manifest with the active seed/opponent scope.
    active_config = json.loads(json.dumps(config))
    active_config["study"]["seeds"] = study["seeds"]
    if args.repeats_override is not None:
        active_config["study"]["repeats_per_leg"] = args.repeats_override
    if args.pairs_override is not None:
        active_config["study"]["team_pairs_per_generation"] = len(args.pairs_override)

    requested_manifest = build_manifest(
        args=args,
        config=active_config,
        source_manifest=source_manifest,
        pool_manifest=pool_manifest,
        opponents=opponents,
    )
    manifest_path = study_root / "study_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not _same_manifest(existing, requested_manifest):
            raise ValueError("The selected --study-dir contains a different v0.6.1 transfer configuration.")
    else:
        manifest_path.write_text(
            json.dumps(requested_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"[INFO] Transfer study: {study_root}")
    print(f"[INFO] Source models: {source_dir}")
    print(f"[INFO] Generations: {study['generations']}")
    print(f"[INFO] Seeds: {study['seeds']}")
    print(f"[INFO] Opponents: {opponents}")
    print("[INFO] Full 3x3 train/evaluation generation matrix; no fine-tuning.")

    scores: list[TransferScore] = []
    telemetry: dict[str, list[dict[str, Any]]] = {
        key: [] for key in (
            "behavior", "battle", "move", "move_type", "move_category", "move_feature",
            "pokemon_type", "team_pokemon", "team_move",
        )
    }
    static_pokemon, static_moves = fixed_pool_catalog_from_manifest(
        pool_manifest,
        project_root=PROJECT_ROOT,
    )
    telemetry["team_pokemon"].extend(static_pokemon)
    telemetry["team_move"].extend(static_moves)
    _write_generation_dex_catalogs(
        study_root / "aggregate" / "telemetry",
        [int(value) for value in study["generations"]],
    )

    for training_generation in study["generations"]:
        for evaluation_generation in study["generations"]:
            for seed in study["seeds"]:
                for opponent in opponents:
                    model_dir = source_model_dir(
                        source_dir,
                        training_generation=int(training_generation),
                        seed=int(seed),
                        checkpoint=int(study["source_checkpoint"]),
                    )
                    run_dir = run_evaluation(
                        source_dir=source_dir,
                        study_root=study_root,
                        training_generation=int(training_generation),
                        evaluation_generation=int(evaluation_generation),
                        seed=int(seed),
                        opponent=opponent,
                        checkpoint=int(study["source_checkpoint"]),
                        repeats=(
                            int(args.repeats_override)
                            if args.repeats_override is not None
                            else int(study["repeats_per_leg"])
                        ),
                        pool_root=pool_root,
                        concurrency=int(args.concurrency),
                        pairs=args.pairs_override,
                    )
                    scores.append(read_transfer_score(
                        run_dir,
                        training_generation=int(training_generation),
                        evaluation_generation=int(evaluation_generation),
                        seed=int(seed),
                        opponent=opponent,
                        model_dir=model_dir,
                    ))
                    block = collect_telemetry(
                        run_dir,
                        training_generation=int(training_generation),
                        evaluation_generation=int(evaluation_generation),
                        seed=int(seed),
                        opponent=opponent,
                    )
                    for key, rows in block.items():
                        telemetry[key].extend(rows)

                    build_score_outputs(study_root, scores)
                    build_telemetry_outputs(study_root, telemetry)

    build_score_outputs(study_root, scores)
    build_telemetry_outputs(study_root, telemetry)

    print("\n=== PokéBench30 v0.6.1 transfer + telemetry complete ===")
    print(f"Study: {study_root}")
    print(f"Transfer matrix: {study_root / 'aggregate' / 'transfer_matrix_summary.csv'}")
    print(f"Transfer gaps: {study_root / 'aggregate' / 'transfer_gap_summary.csv'}")
    print(f"Behavior telemetry: {study_root / 'aggregate' / 'telemetry' / 'behavior_summary.csv'}")
    return study_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "PokéBench30 v0.6.1: zero-shot 3x3 cross-generation transfer "
            "with availability-adjusted behavioral telemetry."
        )
    )
    parser.add_argument("--source-study-dir", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--pool-root", default=str(DEFAULT_POOL_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--study-dir", default=None)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--all-opponents", action="store_true")
    parser.add_argument("--opponents", nargs="+", default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be > 0")
    run_study(args)


if __name__ == "__main__":
    main()
