from __future__ import annotations

import argparse
import asyncio
import csv
import importlib.metadata
import itertools
import json
import platform
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

import yaml

from pokebench30 import __version__
from pokebench30.agents import MaxDamageAgent, RandomAgent, SwitchAwareAgent, TypeAwareAgent
from pokebench30.agents.base_agent import team_snapshot
from pokebench30.evaluation import exact_binomial_two_sided, wilson_interval
from pokebench30.evaluation.analysis import analyze_run


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "generations.yaml"
RESULTS_DIR = PROJECT_ROOT / "results"

AGENT_CLASSES = {
    "random": RandomAgent,
    "max_damage": MaxDamageAgent,
    "type_aware": TypeAwareAgent,
    "switch_aware": SwitchAwareAgent,
}

ACTION_FIELDS = [
    "battle_tag", "generation", "battle_format", "request_index", "turn", "agent",
    "forced_switch_request", "available_moves", "available_switches",
    "active_pokemon", "active_hp_fraction", "active_status", "active_type_1", "active_type_2", "active_boosts",
    "opponent_pokemon", "opponent_hp_fraction", "opponent_status", "opponent_type_1", "opponent_type_2", "opponent_boosts",
    "own_team_known", "own_remaining_known", "own_fainted_known", "own_hp_sum_known",
    "opponent_team_known", "opponent_remaining_known", "opponent_fainted_known", "opponent_hp_sum_known",
    "action_type", "action", "voluntary_switch", "forced_switch", "decision_reason", "heuristic_score",
    "current_matchup_score", "target_matchup_score",
    "move_base_power", "move_accuracy", "move_type", "move_category", "move_stab", "move_type_effectiveness",
    "switch_target_hp_fraction", "switch_target_type_1", "switch_target_type_2",
]


def load_generation_config() -> dict[int, dict]:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return {int(key): value for key, value in raw.get("generations", {}).items()}


def make_player(agent_key: str, battle_format: str, *, max_concurrent_battles: int, log_actions: bool):
    return AGENT_CLASSES[agent_key](
        battle_format=battle_format,
        max_concurrent_battles=max_concurrent_battles,
        log_actions=log_actions,
    )


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def collect_actions(player, generation: int, battle_format: str) -> list[dict]:
    rows = []
    for item in player.action_log:
        row = dict(item)
        row["generation"] = generation
        row["battle_format"] = battle_format
        rows.append(row)
    return rows


def action_counts(player, battle_tag: str) -> dict[str, int]:
    rows = [row for row in player.action_log if row["battle_tag"] == battle_tag]
    return {
        "requests": len(rows),
        "moves": sum(row["action_type"] == "move" for row in rows),
        "switches": sum(row["action_type"] == "switch" for row in rows),
        "voluntary_switches": sum(bool(row["voluntary_switch"]) for row in rows),
        "forced_switches": sum(bool(row["forced_switch"]) for row in rows),
    }


def summarize_battles(battle_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in battle_rows:
        grouped[(row["generation"], row["battle_format"], row["agent_1"], row["agent_2"])].append(row)

    summaries = []
    for (generation, battle_format, agent_1, agent_2), rows in grouped.items():
        n = len(rows)
        wins_1 = sum(row["result_agent_1"] == "win" for row in rows)
        wins_2 = sum(row["result_agent_1"] == "loss" for row in rows)
        draws = sum(row["result_agent_1"] == "draw" for row in rows)
        decisive = wins_1 + wins_2
        ci_low, ci_high = wilson_interval(wins_1, decisive)
        p_value = exact_binomial_two_sided(wins_1, decisive) if decisive else 1.0
        turn_values = [int(row["turns"]) for row in rows if row["turns"] not in ("", None)]

        summaries.append({
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
            "two_sided_binomial_p_vs_0_5": round(p_value, 8),
            "mean_remaining_agent_1": round(mean(int(r["agent_1_remaining_known"]) for r in rows), 3),
            "mean_remaining_agent_2": round(mean(int(r["agent_2_remaining_known"]) for r in rows), 3),
        })

    return sorted(summaries, key=lambda r: (int(r["generation"]), r["agent_1"], r["agent_2"]))


async def run_matchup(
    generation: int, generation_cfg: dict, agent_1_key: str, agent_2_key: str,
    n_battles: int, max_concurrent_battles: int, log_actions: bool, run_id: str,
) -> tuple[list[dict], list[dict]]:
    battle_format = generation_cfg["format"]
    p1 = make_player(agent_1_key, battle_format, max_concurrent_battles=max_concurrent_battles, log_actions=log_actions)
    p2 = make_player(agent_2_key, battle_format, max_concurrent_battles=max_concurrent_battles, log_actions=log_actions)

    print(f"\n[Gen {generation}] {p1.agent_name} vs {p2.agent_name} - {n_battles} battle(s)")
    await p1.battle_against(p2, n_battles=n_battles)

    rows = []
    for battle_tag, battle in sorted(p1.battles.items()):
        if getattr(battle, "won", False):
            result_1, winner = "win", p1.agent_name
        elif getattr(battle, "lost", False):
            result_1, winner = "loss", p2.agent_name
        else:
            result_1, winner = "draw", "draw"

        own_known, own_remaining, own_fainted, own_hp = team_snapshot(getattr(battle, "team", {}))
        opp_known, opp_remaining, opp_fainted, opp_hp = team_snapshot(getattr(battle, "opponent_team", {}))
        c1 = action_counts(p1, battle_tag) if log_actions else {k: 0 for k in ["requests", "moves", "switches", "voluntary_switches", "forced_switches"]}
        c2 = action_counts(p2, battle_tag) if log_actions else {k: 0 for k in ["requests", "moves", "switches", "voluntary_switches", "forced_switches"]}

        rows.append({
            "run_id": run_id,
            "battle_tag": battle_tag,
            "generation": generation,
            "battle_format": battle_format,
            "agent_1": p1.agent_name,
            "agent_2": p2.agent_name,
            "winner": winner,
            "result_agent_1": result_1,
            "turns": getattr(battle, "turn", ""),
            "agent_1_action_requests": p1.action_request_counts.get(battle_tag, 0),
            "agent_2_action_requests": p2.action_request_counts.get(battle_tag, 0),
            "agent_1_moves": c1["moves"],
            "agent_2_moves": c2["moves"],
            "agent_1_switches": c1["switches"],
            "agent_2_switches": c2["switches"],
            "agent_1_voluntary_switches": c1["voluntary_switches"],
            "agent_2_voluntary_switches": c2["voluntary_switches"],
            "agent_1_forced_switches": c1["forced_switches"],
            "agent_2_forced_switches": c2["forced_switches"],
            "agent_1_team_known": own_known,
            "agent_2_team_known": opp_known,
            "agent_1_remaining_known": own_remaining,
            "agent_2_remaining_known": opp_remaining,
            "agent_1_fainted_known": own_fainted,
            "agent_2_fainted_known": opp_fainted,
            "agent_1_hp_sum_known": own_hp,
            "agent_2_hp_sum_known": opp_hp,
        })

    actions = []
    if log_actions:
        actions.extend(collect_actions(p1, generation, battle_format))
        actions.extend(collect_actions(p2, generation, battle_format))

    print(f"  {p1.agent_name}: {p1.n_won_battles} win(s) | {p2.agent_name}: {p2.n_won_battles} win(s) | draws: {p1.n_tied_battles}")
    return rows, actions


async def run_experiment(args: argparse.Namespace) -> Path:
    configs = load_generation_config()
    missing_gens = [g for g in args.generations if g not in configs]
    if missing_gens:
        raise ValueError(f"Generation(s) not configured: {missing_gens}. Available: {sorted(configs)}")
    missing_agents = [a for a in args.agents if a not in AGENT_CLASSES]
    if missing_agents:
        raise ValueError(f"Unknown agent(s): {missing_agents}. Available: {sorted(AGENT_CLASSES)}")
    if len(args.agents) < 2:
        raise ValueError("At least two agents are required.")

    matchups = list(itertools.combinations(args.agents, 2))
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)

    all_battles, all_actions = [], []
    for generation in args.generations:
        for a1, a2 in matchups:
            battles, actions = await run_matchup(
                generation, configs[generation], a1, a2, args.battles,
                args.concurrency, not args.no_action_log, run_id,
            )
            all_battles.extend(battles)
            all_actions.extend(actions)

    battle_fields = [
        "run_id", "battle_tag", "generation", "battle_format", "agent_1", "agent_2", "winner", "result_agent_1", "turns",
        "agent_1_action_requests", "agent_2_action_requests", "agent_1_moves", "agent_2_moves",
        "agent_1_switches", "agent_2_switches", "agent_1_voluntary_switches", "agent_2_voluntary_switches",
        "agent_1_forced_switches", "agent_2_forced_switches", "agent_1_team_known", "agent_2_team_known",
        "agent_1_remaining_known", "agent_2_remaining_known", "agent_1_fainted_known", "agent_2_fainted_known",
        "agent_1_hp_sum_known", "agent_2_hp_sum_known",
    ]
    write_csv(run_dir / "battles.csv", all_battles, battle_fields)

    if not args.no_action_log:
        write_csv(run_dir / "actions.csv", all_actions, ACTION_FIELDS)

    summary_rows = summarize_battles(all_battles)
    summary_fields = [
        "generation", "battle_format", "agent_1", "agent_2", "battles", "wins_agent_1", "wins_agent_2", "draws",
        "win_rate_agent_1", "win_rate_agent_2", "draw_rate", "mean_turns",
        "ci95_low_agent_1_decisive", "ci95_high_agent_1_decisive", "two_sided_binomial_p_vs_0_5",
        "mean_remaining_agent_1", "mean_remaining_agent_2",
    ]
    write_csv(run_dir / "summary.csv", summary_rows, summary_fields)

    try:
        poke_env_version = importlib.metadata.version("poke-env")
    except importlib.metadata.PackageNotFoundError:
        poke_env_version = "unknown"

    metadata = {
        "project": "PokeBench30", "version": __version__, "run_id": run_id,
        "created_at_local": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version, "platform": platform.platform(), "poke_env": poke_env_version,
        "generations": args.generations, "agents": args.agents, "matchups": matchups,
        "battles_per_matchup": args.battles, "max_concurrent_battles": args.concurrency,
        "action_log": not args.no_action_log,
        "limitations": [
            "Random Battle team generation is not seed-controlled in this Player.battle_against workflow.",
            "Random teams confound agent strength with team matchup; fixed mirrored teams are planned for v0.3.",
            "SwitchAwareAgent uses a transparent type-matchup heuristic, not a full damage or search model.",
            "Earlier generations require targeted validation because poke-env is primarily developed around recent generations.",
        ],
    }
    with (run_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    if not args.no_action_log:
        analyze_run(run_dir, make_plots=not args.no_plots)

    print("\n=== Summary ===")
    for row in summary_rows:
        print(
            f"Gen {row['generation']} | {row['agent_1']} vs {row['agent_2']} | "
            f"{row['wins_agent_1']}-{row['wins_agent_2']} (draws={row['draws']}) | "
            f"WR1={float(row['win_rate_agent_1']):.1%} | mean turns={row['mean_turns']} | "
            f"p={float(row['two_sided_binomial_p_vs_0_5']):.4g}"
        )
    print(f"\nResults written to: {run_dir}")
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the PokéBench30 v0.2 instrumented pilot.")
    parser.add_argument("--generations", nargs="+", type=int, default=[1, 4, 9])
    parser.add_argument("--agents", nargs="+", default=["random", "max_damage", "type_aware", "switch_aware"])
    parser.add_argument("--battles", type=int, default=20, help="Battles per matchup per generation. Default: 20")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--no-action-log", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.battles <= 0: raise ValueError("--battles must be greater than zero.")
    if args.concurrency < 1: raise ValueError("--concurrency must be at least 1.")
    try:
        asyncio.run(run_experiment(args))
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
    except Exception as exc:
        print(f"\n[ERROR] {type(exc).__name__}: {exc}")
        print("Check that Pokémon Showdown is running on localhost:8000 and run scripts\\check_setup.py.")
        raise


if __name__ == "__main__":
    main()
