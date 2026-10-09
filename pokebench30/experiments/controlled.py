from __future__ import annotations

import argparse
import asyncio
import csv
from datetime import datetime
import importlib.metadata
import itertools
import json
import platform
from pathlib import Path
import shutil
import sys

from pokebench30 import __version__
from pokebench30.evaluation.analysis import analyze_run
from pokebench30.evaluation.information import final_information_snapshot
from pokebench30.evaluation.mirrored import (
    create_controlled_plots,
    write_mirrored_outputs,
)
from pokebench30.experiments.pilot import (
    ACTION_FIELDS,
    AGENT_CLASSES,
    action_counts,
    summarize_battles,
)
from pokebench30.teams.fixed_pool import (
    DEFAULT_POOL_ROOT,
    TeamRecord,
    load_pool_manifest,
    read_packed_team,
    records_by_key,
    verify_pool,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"

CONTROLLED_CONTEXT_FIELDS = [
    "experiment_mode",
    "pair_id",
    "mirror_leg",
    "repeat_index",
    "agent_team_id",
    "opponent_team_id",
    "team_a_id",
    "team_b_id",
]

CONTROLLED_ACTION_FIELDS = [
    *CONTROLLED_CONTEXT_FIELDS,
    *ACTION_FIELDS,
]

CONTROLLED_BATTLE_FIELDS = [
    "run_id",
    "battle_tag",
    "experiment_mode",
    "generation",
    "battle_format",
    "pair_id",
    "mirror_leg",
    "repeat_index",
    "agent_1",
    "agent_2",
    "team_a_id",
    "team_b_id",
    "agent_1_team_id",
    "agent_2_team_id",
    "agent_1_team_sha256",
    "agent_2_team_sha256",
    "winner",
    "result_agent_1",
    "turns",
    "agent_1_action_requests",
    "agent_2_action_requests",
    "agent_1_moves",
    "agent_2_moves",
    "agent_1_switches",
    "agent_2_switches",
    "agent_1_voluntary_switches",
    "agent_2_voluntary_switches",
    "agent_1_forced_switches",
    "agent_2_forced_switches",
    "agent_1_team_size_true",
    "agent_2_team_size_true",
    "agent_1_remaining_true",
    "agent_2_remaining_true",
    "agent_1_fainted_true",
    "agent_2_fainted_true",
    "agent_1_hp_sum_true",
    "agent_2_hp_sum_true",
    "agent_1_opponent_revealed",
    "agent_2_opponent_revealed",
    "agent_1_opponent_revealed_remaining",
    "agent_2_opponent_revealed_remaining",
    "agent_1_opponent_revealed_fainted",
    "agent_2_opponent_revealed_fainted",
    "agent_1_opponent_revealed_hp_sum",
    "agent_2_opponent_revealed_hp_sum",
    "agent_1_opponent_reveal_fraction",
    "agent_2_opponent_reveal_fraction",
]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def make_player(
    agent_key: str,
    battle_format: str,
    team: str,
    *,
    max_concurrent_battles: int,
    log_actions: bool,
):
    return AGENT_CLASSES[agent_key](
        battle_format=battle_format,
        team=team,
        max_concurrent_battles=max_concurrent_battles,
        log_actions=log_actions,
    )


def _zero_counts() -> dict[str, int]:
    return {
        key: 0
        for key in (
            "requests",
            "moves",
            "switches",
            "voluntary_switches",
            "forced_switches",
        )
    }


def _annotate_actions(
    player,
    *,
    generation: int,
    battle_format: str,
    pair_id: str,
    mirror_leg: str,
    repeat_map: dict[str, int],
    agent_team_id: str,
    opponent_team_id: str,
    team_a_id: str,
    team_b_id: str,
) -> list[dict]:
    output = []
    for item in player.action_log:
        battle_tag = item["battle_tag"]
        row = {
            "experiment_mode": "fixed_mirrored",
            "pair_id": pair_id,
            "mirror_leg": mirror_leg,
            "repeat_index": repeat_map[battle_tag],
            "agent_team_id": agent_team_id,
            "opponent_team_id": opponent_team_id,
            "team_a_id": team_a_id,
            "team_b_id": team_b_id,
            **item,
            "generation": generation,
            "battle_format": battle_format,
        }
        output.append(row)
    return output


async def run_leg(
    *,
    generation: int,
    pair_id: str,
    mirror_leg: str,
    agent_1_key: str,
    agent_2_key: str,
    agent_1_team: TeamRecord,
    agent_2_team: TeamRecord,
    team_a: TeamRecord,
    team_b: TeamRecord,
    repeats: int,
    max_concurrent_battles: int,
    log_actions: bool,
    run_id: str,
) -> tuple[list[dict], list[dict]]:
    if agent_1_team.battle_format != agent_2_team.battle_format:
        raise ValueError("Team records disagree on battle format.")

    battle_format = agent_1_team.battle_format

    p1 = make_player(
        agent_1_key,
        battle_format,
        read_packed_team(agent_1_team),
        max_concurrent_battles=max_concurrent_battles,
        log_actions=log_actions,
    )
    p2 = make_player(
        agent_2_key,
        battle_format,
        read_packed_team(agent_2_team),
        max_concurrent_battles=max_concurrent_battles,
        log_actions=log_actions,
    )

    print(
        f"\n[Gen {generation}] {pair_id} {mirror_leg} | "
        f"{p1.agent_name}({agent_1_team.team_id}) vs "
        f"{p2.agent_name}({agent_2_team.team_id}) | "
        f"{repeats} battle(s)"
    )

    await p1.battle_against(p2, n_battles=repeats)

    tags = sorted(p1.battles)
    repeat_map = {
        battle_tag: index
        for index, battle_tag in enumerate(tags, start=1)
    }

    rows = []

    for battle_tag in tags:
        battle = p1.battles[battle_tag]
        p2_battle = p2.battles.get(battle_tag)
        if p2_battle is None:
            raise RuntimeError(
                f"Battle {battle_tag} missing from agent 2 registry."
            )

        if getattr(battle, "won", False):
            result_1, winner = "win", p1.agent_name
        elif getattr(battle, "lost", False):
            result_1, winner = "loss", p2.agent_name
        else:
            result_1, winner = "draw", "draw"

        info = final_information_snapshot(battle, p2_battle)

        c1 = (
            action_counts(p1, battle_tag)
            if log_actions
            else _zero_counts()
        )
        c2 = (
            action_counts(p2, battle_tag)
            if log_actions
            else _zero_counts()
        )

        rows.append({
            "run_id": run_id,
            "battle_tag": battle_tag,
            "experiment_mode": "fixed_mirrored",
            "generation": generation,
            "battle_format": battle_format,
            "pair_id": pair_id,
            "mirror_leg": mirror_leg,
            "repeat_index": repeat_map[battle_tag],
            "agent_1": p1.agent_name,
            "agent_2": p2.agent_name,
            "team_a_id": team_a.team_id,
            "team_b_id": team_b.team_id,
            "agent_1_team_id": agent_1_team.team_id,
            "agent_2_team_id": agent_2_team.team_id,
            "agent_1_team_sha256": agent_1_team.sha256,
            "agent_2_team_sha256": agent_2_team.sha256,
            "winner": winner,
            "result_agent_1": result_1,
            "turns": getattr(battle, "turn", ""),
            "agent_1_action_requests": c1["requests"],
            "agent_2_action_requests": c2["requests"],
            "agent_1_moves": c1["moves"],
            "agent_2_moves": c2["moves"],
            "agent_1_switches": c1["switches"],
            "agent_2_switches": c2["switches"],
            "agent_1_voluntary_switches": c1["voluntary_switches"],
            "agent_2_voluntary_switches": c2["voluntary_switches"],
            "agent_1_forced_switches": c1["forced_switches"],
            "agent_2_forced_switches": c2["forced_switches"],
            **info,
        })

    actions = []
    if log_actions:
        actions.extend(
            _annotate_actions(
                p1,
                generation=generation,
                battle_format=battle_format,
                pair_id=pair_id,
                mirror_leg=mirror_leg,
                repeat_map=repeat_map,
                agent_team_id=agent_1_team.team_id,
                opponent_team_id=agent_2_team.team_id,
                team_a_id=team_a.team_id,
                team_b_id=team_b.team_id,
            )
        )
        actions.extend(
            _annotate_actions(
                p2,
                generation=generation,
                battle_format=battle_format,
                pair_id=pair_id,
                mirror_leg=mirror_leg,
                repeat_map=repeat_map,
                agent_team_id=agent_2_team.team_id,
                opponent_team_id=agent_1_team.team_id,
                team_a_id=team_a.team_id,
                team_b_id=team_b.team_id,
            )
        )

    print(
        f"  {p1.agent_name}: {p1.n_won_battles} win(s) | "
        f"{p2.agent_name}: {p2.n_won_battles} win(s) | "
        f"draws: {p1.n_tied_battles}"
    )

    return rows, actions


def _snapshot_teams(
    run_dir: Path,
    manifest: dict,
    records: list[TeamRecord],
) -> None:
    snapshot_dir = run_dir / "teams_snapshot"
    snapshot_dir.mkdir(exist_ok=True)

    unique = {record.team_id: record for record in records}

    for record in unique.values():
        source = PROJECT_ROOT / record.packed_path
        target = snapshot_dir / f"{record.team_id}.packed.txt"
        shutil.copy2(source, target)

        source_export = PROJECT_ROOT / record.export_path
        if source_export.exists():
            shutil.copy2(
                source_export,
                snapshot_dir / f"{record.team_id}.txt",
            )

    snapshot_manifest = {
        "source_pool": manifest,
        "teams_used": [
            {
                "team_id": record.team_id,
                "generation": record.generation,
                "pair_id": record.pair_id,
                "side": record.side,
                "seed": record.seed,
                "sha256": record.sha256,
            }
            for record in sorted(
                unique.values(),
                key=lambda r: (r.generation, r.pair_id, r.side),
            )
        ],
    }

    (snapshot_dir / "manifest.json").write_text(
        json.dumps(snapshot_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _resolve_matchups(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.matchup:
        a1, a2 = args.matchup
        for agent in (a1, a2):
            if agent not in AGENT_CLASSES:
                raise ValueError(
                    f"Unknown agent {agent!r}. Available: {sorted(AGENT_CLASSES)}"
                )
        return [(a1, a2)]

    missing = [a for a in args.agents if a not in AGENT_CLASSES]
    if missing:
        raise ValueError(
            f"Unknown agent(s): {missing}. Available: {sorted(AGENT_CLASSES)}"
        )
    if len(args.agents) < 2:
        raise ValueError("At least two agents are required.")
    return list(itertools.combinations(args.agents, 2))


async def run_experiment(args: argparse.Namespace) -> Path:
    pool_root = Path(args.pool_root).resolve()
    problems = verify_pool(pool_root)
    if problems:
        raise RuntimeError(
            "Fixed-team pool failed integrity check:\n- "
            + "\n- ".join(problems)
        )

    manifest = load_pool_manifest(pool_root)
    records = records_by_key(manifest)
    matchups = _resolve_matchups(args)

    available_generations = sorted(
        {key[0] for key in records}
    )
    missing_generations = [
        generation
        for generation in args.generations
        if generation not in available_generations
    ]
    if missing_generations:
        raise ValueError(
            f"Generation(s) missing from fixed pool: {missing_generations}"
        )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / f"controlled_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)

    all_battles: list[dict] = []
    all_actions: list[dict] = []
    used_records: list[TeamRecord] = []

    for generation in args.generations:
        available_pairs = sorted({
            pair_id
            for gen, pair_id, side in records
            if gen == generation
        })

        selected_pairs = (
            args.pairs
            if args.pairs
            else available_pairs
        )

        missing_pairs = [
            pair_id
            for pair_id in selected_pairs
            if pair_id not in available_pairs
        ]
        if missing_pairs:
            raise ValueError(
                f"Gen {generation} does not contain pair(s): {missing_pairs}"
            )

        for agent_1_key, agent_2_key in matchups:
            for pair_id in selected_pairs:
                team_a = records[(generation, pair_id, "A")]
                team_b = records[(generation, pair_id, "B")]
                used_records.extend([team_a, team_b])

                # Leg AB: agent 1 uses A, agent 2 uses B.
                battles, actions = await run_leg(
                    generation=generation,
                    pair_id=pair_id,
                    mirror_leg="AB",
                    agent_1_key=agent_1_key,
                    agent_2_key=agent_2_key,
                    agent_1_team=team_a,
                    agent_2_team=team_b,
                    team_a=team_a,
                    team_b=team_b,
                    repeats=args.repeats,
                    max_concurrent_battles=args.concurrency,
                    log_actions=not args.no_action_log,
                    run_id=run_id,
                )
                all_battles.extend(battles)
                all_actions.extend(actions)

                # Leg BA: same agents, fixed teams swapped.
                battles, actions = await run_leg(
                    generation=generation,
                    pair_id=pair_id,
                    mirror_leg="BA",
                    agent_1_key=agent_1_key,
                    agent_2_key=agent_2_key,
                    agent_1_team=team_b,
                    agent_2_team=team_a,
                    team_a=team_a,
                    team_b=team_b,
                    repeats=args.repeats,
                    max_concurrent_battles=args.concurrency,
                    log_actions=not args.no_action_log,
                    run_id=run_id,
                )
                all_battles.extend(battles)
                all_actions.extend(actions)

    write_csv(
        run_dir / "battles.csv",
        all_battles,
        CONTROLLED_BATTLE_FIELDS,
    )

    if not args.no_action_log:
        write_csv(
            run_dir / "actions.csv",
            all_actions,
            CONTROLLED_ACTION_FIELDS,
        )

    # Backwards-compatible raw summary.
    raw_summary = summarize_battles(all_battles)
    if raw_summary:
        write_csv(
            run_dir / "summary.csv",
            raw_summary,
            list(raw_summary[0].keys()),
        )

    cycles, pair_summary, controlled_summary = write_mirrored_outputs(
        run_dir,
        all_battles,
    )

    if not args.no_action_log:
        analyze_run(
            run_dir,
            make_plots=not args.no_plots,
        )

    if not args.no_plots:
        create_controlled_plots(
            run_dir,
            controlled_summary,
        )

    _snapshot_teams(
        run_dir,
        manifest,
        used_records,
    )

    try:
        poke_env_version = importlib.metadata.version("poke-env")
    except importlib.metadata.PackageNotFoundError:
        poke_env_version = "unknown"

    metadata = {
        "project": "PokeBench30",
        "version": __version__,
        "run_id": run_id,
        "experiment_mode": "fixed_mirrored",
        "created_at_local": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "poke_env": poke_env_version,
        "generations": args.generations,
        "matchups": matchups,
        "pairs": args.pairs or "all",
        "repeats_per_leg": args.repeats,
        "battles_per_mirror_cycle": 2,
        "max_concurrent_battles": args.concurrency,
        "action_log": not args.no_action_log,
        "fixed_team_pool": {
            "config_sha256": manifest.get("config_sha256"),
            "showdown_commit": manifest.get("showdown_commit"),
            "showdown_dirty": manifest.get("showdown_dirty"),
            "node_version": manifest.get("node_version"),
        },
        "information_policy": {
            "team_preview": "explicitly removed from controlled formats",
            "agent_observation": (
                "Agents use only their own battle perspective. Hidden opponent "
                "slots are not reconstructed."
            ),
            "ground_truth": (
                "Full team ground truth is computed only after a battle for "
                "evaluation."
            ),
        },
        "design": {
            "fixed_teams": True,
            "deterministic_team_generation_seeds": True,
            "mirrored_team_assignment": True,
            "exact_team_hash_snapshot": True,
        },
        "limitations": [
            "Battle-event RNG is not seed-controlled through poke-env challenge battles.",
            "Fixed teams are generated by the Pokémon Showdown version recorded in the pool manifest.",
            "Mirroring controls team assignment but does not remove all team-by-agent interaction effects.",
            "Generation-specific special mechanics remain disabled intentionally in scripted baselines.",
        ],
    }

    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== Controlled mirrored summary ===")
    for row in controlled_summary:
        print(
            f"Gen {row['generation']} | "
            f"{row['agent_1']} vs {row['agent_2']} | "
            f"mirror score A1={float(row['normalized_mirror_score_agent_1']):.3f} | "
            f"raw {row['wins_agent_1']}-{row['wins_agent_2']} "
            f"(draws={row['draws']}) | "
            f"assignment gap={float(row['agent_1_team_assignment_gap']):.3f}"
        )

    print(f"\nResults written to: {run_dir}")
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run PokéBench30 v0.3 fixed-team mirrored experiments."
    )
    parser.add_argument(
        "--generations",
        nargs="+",
        type=int,
        default=[1, 4, 9],
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["type_aware", "switch_aware"],
        help="Agents used to create all pairwise matchups.",
    )
    parser.add_argument(
        "--matchup",
        nargs=2,
        metavar=("AGENT_1", "AGENT_2"),
        help=(
            "Run exactly one matchup. Unlike --agents, this supports self-play "
            "such as --matchup type_aware type_aware."
        ),
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=None,
        help="Subset of fixed pair IDs. Default: all prepared pairs.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Battles per mirror leg. Each repeat produces two mirrored battles.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--pool-root",
        default=str(DEFAULT_POOL_ROOT),
    )
    parser.add_argument("--no-action-log", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.repeats <= 0:
        raise ValueError("--repeats must be greater than zero.")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1.")

    try:
        asyncio.run(run_experiment(args))
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
    except Exception as exc:
        print(f"\n[ERROR] {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
