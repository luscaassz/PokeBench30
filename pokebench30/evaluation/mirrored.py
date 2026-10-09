from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def points_for_result(result_agent_1: str) -> float:
    if result_agent_1 == "win":
        return 1.0
    if result_agent_1 == "draw":
        return 0.5
    return 0.0


def build_mirror_cycles(battle_rows: list[dict]) -> list[dict]:
    groups = defaultdict(dict)

    for row in battle_rows:
        key = (
            int(row["generation"]),
            row["pair_id"],
            row["agent_1"],
            row["agent_2"],
            int(row["repeat_index"]),
        )
        groups[key][row["mirror_leg"]] = row

    cycles = []
    for key, legs in sorted(groups.items()):
        if "AB" not in legs or "BA" not in legs:
            raise ValueError(
                f"Incomplete mirrored cycle {key}: expected AB and BA, "
                f"found {sorted(legs)}"
            )

        generation, pair_id, agent_1, agent_2, repeat_index = key
        ab = legs["AB"]
        ba = legs["BA"]

        points_1 = (
            points_for_result(ab["result_agent_1"])
            + points_for_result(ba["result_agent_1"])
        )
        points_2 = 2.0 - points_1

        if points_1 > 1.0:
            cycle_outcome = "agent_1_advantage"
        elif points_1 < 1.0:
            cycle_outcome = "agent_2_advantage"
        else:
            cycle_outcome = "balanced"

        cycles.append({
            "generation": generation,
            "pair_id": pair_id,
            "agent_1": agent_1,
            "agent_2": agent_2,
            "repeat_index": repeat_index,
            "leg_ab_result_agent_1": ab["result_agent_1"],
            "leg_ba_result_agent_1": ba["result_agent_1"],
            "agent_1_points": points_1,
            "agent_2_points": points_2,
            "normalized_agent_1_score": round(points_1 / 2.0, 6),
            "cycle_outcome": cycle_outcome,
        })

    return cycles


def _battle_winner_team(row: dict) -> str:
    if row["result_agent_1"] == "draw":
        return "draw"
    if row["result_agent_1"] == "win":
        return row["agent_1_team_id"]
    return row["agent_2_team_id"]


def _summary_row(rows: list[dict], cycles: list[dict]) -> dict:
    wins_1 = sum(r["result_agent_1"] == "win" for r in rows)
    wins_2 = sum(r["result_agent_1"] == "loss" for r in rows)
    draws = sum(r["result_agent_1"] == "draw" for r in rows)

    as_a = [r for r in rows if r["agent_1_team_id"] == r["team_a_id"]]
    as_b = [r for r in rows if r["agent_1_team_id"] == r["team_b_id"]]

    wr_a = (
        sum(r["result_agent_1"] == "win" for r in as_a) / len(as_a)
        if as_a else 0.0
    )
    wr_b = (
        sum(r["result_agent_1"] == "win" for r in as_b) / len(as_b)
        if as_b else 0.0
    )

    team_a_wins = 0
    team_b_wins = 0
    for row in rows:
        winner_team = _battle_winner_team(row)
        if winner_team == row["team_a_id"]:
            team_a_wins += 1
        elif winner_team == row["team_b_id"]:
            team_b_wins += 1

    points_1 = sum(float(c["agent_1_points"]) for c in cycles)
    mirror_denominator = 2.0 * len(cycles)

    return {
        "battles": len(rows),
        "mirror_cycles": len(cycles),
        "wins_agent_1": wins_1,
        "wins_agent_2": wins_2,
        "draws": draws,
        "raw_win_rate_agent_1": round(wins_1 / len(rows), 6) if rows else 0.0,
        "normalized_mirror_score_agent_1": round(
            points_1 / mirror_denominator, 6
        ) if mirror_denominator else 0.0,
        "mean_points_per_cycle_agent_1": round(
            mean(float(c["agent_1_points"]) for c in cycles), 6
        ) if cycles else 0.0,
        "cycles_agent_1_advantage": sum(
            c["cycle_outcome"] == "agent_1_advantage" for c in cycles
        ),
        "cycles_agent_2_advantage": sum(
            c["cycle_outcome"] == "agent_2_advantage" for c in cycles
        ),
        "cycles_balanced": sum(
            c["cycle_outcome"] == "balanced" for c in cycles
        ),
        "agent_1_win_rate_as_team_a": round(wr_a, 6),
        "agent_1_win_rate_as_team_b": round(wr_b, 6),
        "agent_1_team_assignment_gap": round(abs(wr_a - wr_b), 6),
        "team_a_wins": team_a_wins,
        "team_b_wins": team_b_wins,
        "team_draws": draws,
        "team_a_win_rate_all": round(
            team_a_wins / len(rows), 6
        ) if rows else 0.0,
    }


def build_pair_summary(
    battle_rows: list[dict],
    cycles: list[dict],
) -> list[dict]:
    battle_groups = defaultdict(list)
    cycle_groups = defaultdict(list)

    for row in battle_rows:
        key = (
            int(row["generation"]),
            row["pair_id"],
            row["agent_1"],
            row["agent_2"],
        )
        battle_groups[key].append(row)

    for row in cycles:
        key = (
            int(row["generation"]),
            row["pair_id"],
            row["agent_1"],
            row["agent_2"],
        )
        cycle_groups[key].append(row)

    output = []
    for key, rows in sorted(battle_groups.items()):
        generation, pair_id, agent_1, agent_2 = key
        summary = _summary_row(rows, cycle_groups[key])
        output.append({
            "generation": generation,
            "pair_id": pair_id,
            "agent_1": agent_1,
            "agent_2": agent_2,
            **summary,
        })
    return output


def build_controlled_summary(
    battle_rows: list[dict],
    cycles: list[dict],
) -> list[dict]:
    battle_groups = defaultdict(list)
    cycle_groups = defaultdict(list)

    for row in battle_rows:
        key = (
            int(row["generation"]),
            row["agent_1"],
            row["agent_2"],
        )
        battle_groups[key].append(row)

    for row in cycles:
        key = (
            int(row["generation"]),
            row["agent_1"],
            row["agent_2"],
        )
        cycle_groups[key].append(row)

    output = []
    for key, rows in sorted(battle_groups.items()):
        generation, agent_1, agent_2 = key
        summary = _summary_row(rows, cycle_groups[key])
        output.append({
            "generation": generation,
            "agent_1": agent_1,
            "agent_2": agent_2,
            "team_pairs": len({r["pair_id"] for r in rows}),
            **summary,
        })
    return output


def write_mirrored_outputs(
    run_dir: Path,
    battle_rows: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    cycles = build_mirror_cycles(battle_rows)
    pair_summary = build_pair_summary(battle_rows, cycles)
    controlled_summary = build_controlled_summary(battle_rows, cycles)

    _write_csv(
        run_dir / "mirror_cycles.csv",
        cycles,
        [
            "generation", "pair_id", "agent_1", "agent_2", "repeat_index",
            "leg_ab_result_agent_1", "leg_ba_result_agent_1",
            "agent_1_points", "agent_2_points",
            "normalized_agent_1_score", "cycle_outcome",
        ],
    )

    common_fields = [
        "battles", "mirror_cycles",
        "wins_agent_1", "wins_agent_2", "draws",
        "raw_win_rate_agent_1",
        "normalized_mirror_score_agent_1",
        "mean_points_per_cycle_agent_1",
        "cycles_agent_1_advantage",
        "cycles_agent_2_advantage",
        "cycles_balanced",
        "agent_1_win_rate_as_team_a",
        "agent_1_win_rate_as_team_b",
        "agent_1_team_assignment_gap",
        "team_a_wins", "team_b_wins", "team_draws",
        "team_a_win_rate_all",
    ]

    _write_csv(
        run_dir / "pair_summary.csv",
        pair_summary,
        ["generation", "pair_id", "agent_1", "agent_2", *common_fields],
    )

    _write_csv(
        run_dir / "controlled_summary.csv",
        controlled_summary,
        [
            "generation", "agent_1", "agent_2", "team_pairs",
            *common_fields,
        ],
    )

    return cycles, pair_summary, controlled_summary


def create_controlled_plots(
    run_dir: Path,
    controlled_summary: list[dict],
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    if not controlled_summary:
        return []

    figure_dir = run_dir / "figures"
    figure_dir.mkdir(exist_ok=True)

    generations = sorted(
        {int(row["generation"]) for row in controlled_summary}
    )
    matchups = sorted({
        (row["agent_1"], row["agent_2"])
        for row in controlled_summary
    })

    generated = []

    fig, ax = plt.subplots(figsize=(8, 5))
    for agent_1, agent_2 in matchups:
        values = []
        for generation in generations:
            row = next(
                (
                    r for r in controlled_summary
                    if int(r["generation"]) == generation
                    and r["agent_1"] == agent_1
                    and r["agent_2"] == agent_2
                ),
                None,
            )
            values.append(
                float(row["normalized_mirror_score_agent_1"])
                if row else float("nan")
            )
        ax.plot(
            generations,
            values,
            marker="o",
            label=f"{agent_1} vs {agent_2}",
        )

    ax.axhline(0.5, linestyle="--", linewidth=1)
    ax.set_xticks(generations)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Normalized mirrored score (Agent 1)")
    ax.set_title("Mirrored controlled score by generation")
    ax.legend()
    fig.tight_layout()

    path = figure_dir / "mirrored_score.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    generated.append(path)

    fig, ax = plt.subplots(figsize=(8, 5))
    for agent_1, agent_2 in matchups:
        values = []
        for generation in generations:
            row = next(
                (
                    r for r in controlled_summary
                    if int(r["generation"]) == generation
                    and r["agent_1"] == agent_1
                    and r["agent_2"] == agent_2
                ),
                None,
            )
            values.append(
                float(row["agent_1_team_assignment_gap"])
                if row else float("nan")
            )
        ax.plot(
            generations,
            values,
            marker="o",
            label=f"{agent_1} vs {agent_2}",
        )

    ax.set_xticks(generations)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Generation")
    ax.set_ylabel("|WR as Team A - WR as Team B|")
    ax.set_title("Agent team-assignment sensitivity")
    ax.legend()
    fig.tight_layout()

    path = figure_dir / "team_assignment_gap.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    generated.append(path)

    return generated
