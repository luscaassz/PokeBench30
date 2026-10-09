from __future__ import annotations

import argparse
import asyncio
import csv
import json
import platform
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Iterable

import yaml

from pokebench30 import __version__
from pokebench30.agents import MaxDamageAgent, RandomAgent, TypeAwareAgent
from pokebench30.evaluation import wilson_interval


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "generations.yaml"
RESULTS_DIR = PROJECT_ROOT / "results"

AGENT_CLASSES = {
    "random": RandomAgent,
    "max_damage": MaxDamageAgent,
    "type_aware": TypeAwareAgent,
}

MATCHUPS = [
    ("random", "max_damage"),
    ("random", "type_aware"),
    ("max_damage", "type_aware"),
]


def load_generation_config() -> dict[int, dict]:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    generations = raw.get("generations", {})
    return {int(key): value for key, value in generations.items()}


def make_player(
    agent_key: str,
    battle_format: str,
    *,
    max_concurrent_battles: int,
    log_decisions: bool,
):
    cls = AGENT_CLASSES[agent_key]
    return cls(
        battle_format=battle_format,
        max_concurrent_battles=max_concurrent_battles,
        log_decisions=log_decisions,
    )


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_decisions(player, generation: int, battle_format: str) -> list[dict]:
    rows = []
    for item in player.decision_log:
        rows.append(
            {
                "battle_tag": item["battle_tag"],
                "generation": generation,
                "battle_format": battle_format,
                "turn": item["turn"],
                "agent": item["agent"],
                "action_type": item["action_type"],
                "action": item["action"],
                "score": item["score"],
            }
        )
    return rows


def summarize_battles(battle_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)

    for row in battle_rows:
        key = (
            row["generation"],
            row["battle_format"],
            row["agent_1"],
            row["agent_2"],
        )
        grouped[key].append(row)

    summaries = []

    for (generation, battle_format, agent_1, agent_2), rows in grouped.items():
        n = len(rows)
        wins_1 = sum(row["result_agent_1"] == "win" for row in rows)
        wins_2 = sum(row["result_agent_1"] == "loss" for row in rows)
        draws = sum(row["result_agent_1"] == "draw" for row in rows)

        decisive = wins_1 + wins_2
        ci_low, ci_high = wilson_interval(wins_1, decisive)

        turn_values = [
            int(row["turns"])
            for row in rows
            if row["turns"] not in ("", None)
        ]

        summaries.append(
            {
                "generation": generation,
                "battle_format": battle_format,
                "agent_1": agent_1,
                "agent_2": agent_2,
                "battles": n,
                "wins_agent_1": wins_1,
                "wins_agent_2": wins_2,
                "draws": draws,
                "win_rate_agent_1": round(wins_1 / n, 6) if n else 0.0,
                "win_rate_agent_2": round(wins_2 / n, 6) if n else 0.0,
                "draw_rate": round(draws / n, 6) if n else 0.0,
                "mean_turns": round(mean(turn_values), 3) if turn_values else "",
                "ci95_low_agent_1_decisive": round(ci_low, 6),
                "ci95_high_agent_1_decisive": round(ci_high, 6),
            }
        )

    summaries.sort(
        key=lambda row: (
            int(row["generation"]),
            row["agent_1"],
            row["agent_2"],
        )
    )
    return summaries


async def run_matchup(
    generation: int,
    generation_cfg: dict,
    agent_1_key: str,
    agent_2_key: str,
    n_battles: int,
    max_concurrent_battles: int,
    log_decisions: bool,
    run_id: str,
) -> tuple[list[dict], list[dict]]:
    battle_format = generation_cfg["format"]

    player_1 = make_player(
        agent_1_key,
        battle_format,
        max_concurrent_battles=max_concurrent_battles,
        log_decisions=log_decisions,
    )
    player_2 = make_player(
        agent_2_key,
        battle_format,
        max_concurrent_battles=max_concurrent_battles,
        log_decisions=log_decisions,
    )

    print(
        f"\n[Gen {generation}] "
        f"{player_1.agent_name} vs {player_2.agent_name} "
        f"— {n_battles} battle(s)"
    )

    await player_1.battle_against(player_2, n_battles=n_battles)

    rows = []

    # The same battle tags are visible from both players. We use player 1's
    # battle objects because battle.won/lost are from that player's viewpoint.
    for battle_tag, battle in sorted(player_1.battles.items()):
        if getattr(battle, "won", False):
            result_1 = "win"
            winner = player_1.agent_name
        elif getattr(battle, "lost", False):
            result_1 = "loss"
            winner = player_2.agent_name
        else:
            result_1 = "draw"
            winner = "draw"

        rows.append(
            {
                "run_id": run_id,
                "battle_tag": battle_tag,
                "generation": generation,
                "battle_format": battle_format,
                "agent_1": player_1.agent_name,
                "agent_2": player_2.agent_name,
                "winner": winner,
                "result_agent_1": result_1,
                "turns": getattr(battle, "turn", ""),
                "agent_1_decisions": player_1.decision_counts.get(battle_tag, 0),
                "agent_2_decisions": player_2.decision_counts.get(battle_tag, 0),
            }
        )

    decision_rows: list[dict] = []
    if log_decisions:
        decision_rows.extend(
            collect_decisions(player_1, generation, battle_format)
        )
        decision_rows.extend(
            collect_decisions(player_2, generation, battle_format)
        )

    print(
        f"  {player_1.agent_name}: {player_1.n_won_battles} win(s) | "
        f"{player_2.agent_name}: {player_2.n_won_battles} win(s) | "
        f"draws: {player_1.n_tied_battles}"
    )

    return rows, decision_rows


async def run_experiment(args: argparse.Namespace) -> Path:
    configs = load_generation_config()

    missing = [gen for gen in args.generations if gen not in configs]
    if missing:
        available = ", ".join(str(gen) for gen in sorted(configs))
        raise ValueError(
            f"Generation(s) not configured: {missing}. Available: {available}"
        )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)

    all_battles: list[dict] = []
    all_decisions: list[dict] = []

    for generation in args.generations:
        for agent_1, agent_2 in MATCHUPS:
            battle_rows, decision_rows = await run_matchup(
                generation=generation,
                generation_cfg=configs[generation],
                agent_1_key=agent_1,
                agent_2_key=agent_2,
                n_battles=args.battles,
                max_concurrent_battles=args.concurrency,
                log_decisions=not args.no_decision_log,
                run_id=run_id,
            )
            all_battles.extend(battle_rows)
            all_decisions.extend(decision_rows)

    battle_fields = [
        "run_id",
        "battle_tag",
        "generation",
        "battle_format",
        "agent_1",
        "agent_2",
        "winner",
        "result_agent_1",
        "turns",
        "agent_1_decisions",
        "agent_2_decisions",
    ]
    write_csv(run_dir / "battles.csv", all_battles, battle_fields)

    if not args.no_decision_log:
        decision_fields = [
            "battle_tag",
            "generation",
            "battle_format",
            "turn",
            "agent",
            "action_type",
            "action",
            "score",
        ]
        write_csv(
            run_dir / "decisions.csv",
            all_decisions,
            decision_fields,
        )

    summary_rows = summarize_battles(all_battles)
    summary_fields = [
        "generation",
        "battle_format",
        "agent_1",
        "agent_2",
        "battles",
        "wins_agent_1",
        "wins_agent_2",
        "draws",
        "win_rate_agent_1",
        "win_rate_agent_2",
        "draw_rate",
        "mean_turns",
        "ci95_low_agent_1_decisive",
        "ci95_high_agent_1_decisive",
    ]
    write_csv(run_dir / "summary.csv", summary_rows, summary_fields)

    metadata = {
        "project": "PokeBench30",
        "version": __version__,
        "run_id": run_id,
        "created_at_local": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "generations": args.generations,
        "battles_per_matchup": args.battles,
        "max_concurrent_battles": args.concurrency,
        "decision_log": not args.no_decision_log,
        "matchups": MATCHUPS,
        "note": (
            "v0.1 does not control Pokemon Showdown Random Battle seeds. "
            "Do not treat this run as exactly reproducible at the battle level."
        ),
    }

    with (run_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print("\n=== Summary ===")
    for row in summary_rows:
        print(
            f"Gen {row['generation']} | "
            f"{row['agent_1']} vs {row['agent_2']} | "
            f"{row['wins_agent_1']}-{row['wins_agent_2']} "
            f"(draws={row['draws']}) | "
            f"WR1={float(row['win_rate_agent_1']):.1%} | "
            f"mean turns={row['mean_turns']}"
        )

    print(f"\nResults written to: {run_dir}")
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the PokéBench30 v0.1 baseline pilot."
    )
    parser.add_argument(
        "--generations",
        nargs="+",
        type=int,
        default=[1, 4, 9],
        help="Generations to test. Default: 1 4 9",
    )
    parser.add_argument(
        "--battles",
        type=int,
        default=50,
        help="Battles per matchup per generation. Default: 50",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help=(
            "Maximum simultaneous battles per player. "
            "Use 1 first; increase only after validating your setup."
        ),
    )
    parser.add_argument(
        "--no-decision-log",
        action="store_true",
        help="Do not write per-decision logs.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.battles <= 0:
        raise ValueError("--battles must be greater than zero.")
    if args.concurrency < 0:
        raise ValueError("--concurrency cannot be negative.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    try:
        asyncio.run(run_experiment(args))
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
    except Exception as exc:
        print(f"\n[ERROR] {type(exc).__name__}: {exc}")
        print(
            "\nCheck whether Pokémon Showdown is running locally on port 8000 "
            "and run: python scripts/check_setup.py"
        )
        raise


if __name__ == "__main__":
    main()
