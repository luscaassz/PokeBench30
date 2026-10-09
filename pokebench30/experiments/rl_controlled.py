from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from pokebench30 import __version__
from pokebench30.evaluation.analysis import analyze_run
from pokebench30.evaluation.information import final_information_snapshot
from pokebench30.evaluation.telemetry import (
    team_catalog,
    write_telemetry_outputs,
)
from pokebench30.evaluation.mirrored import (
    create_controlled_plots,
    write_mirrored_outputs,
)
from pokebench30.experiments.controlled import (
    CONTROLLED_ACTION_FIELDS,
    CONTROLLED_BATTLE_FIELDS,
    _annotate_actions,
    _snapshot_teams,
    _zero_counts,
    write_csv,
)
from pokebench30.experiments.pilot import (
    AGENT_CLASSES,
    action_counts,
    summarize_battles,
)
from pokebench30.rl.model_io import (
    load_model_metadata,
    resolve_model_file,
    validate_model_metadata,
)
from pokebench30.rl.policy_agent import RLPolicyAgent
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


def make_baseline(
    key: str,
    *,
    battle_format: str,
    team: str,
    max_concurrent_battles: int,
    log_actions: bool,
):
    if key not in AGENT_CLASSES:
        raise ValueError(
            f"Unknown opponent {key!r}. Available: {sorted(AGENT_CLASSES)}"
        )

    return AGENT_CLASSES[key](
        battle_format=battle_format,
        team=team,
        max_concurrent_battles=max_concurrent_battles,
        log_actions=log_actions,
        log_universal_state=False,
    )


async def run_rl_leg(
    *,
    generation: int,
    pair_id: str,
    mirror_leg: str,
    model_path: Path,
    opponent_key: str,
    rl_team: TeamRecord,
    opponent_team: TeamRecord,
    team_a: TeamRecord,
    team_b: TeamRecord,
    repeats: int,
    max_concurrent_battles: int,
    log_actions: bool,
    deterministic: bool,
    run_id: str,
    allow_cross_generation: bool = False,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    battle_format = rl_team.battle_format
    if opponent_team.battle_format != battle_format:
        raise ValueError("Team records disagree on battle format.")

    rl_player = RLPolicyAgent(
        model_path=model_path,
        deterministic=deterministic,
        require_same_generation=not allow_cross_generation,
        battle_format=battle_format,
        team=read_packed_team(rl_team),
        max_concurrent_battles=max_concurrent_battles,
        log_actions=log_actions,
        log_universal_state=False,
    )
    opponent = make_baseline(
        opponent_key,
        battle_format=battle_format,
        team=read_packed_team(opponent_team),
        max_concurrent_battles=max_concurrent_battles,
        log_actions=log_actions,
    )

    print(
        f"\n[Gen {generation}] {pair_id} {mirror_leg} | "
        f"{rl_player.agent_name}({rl_team.team_id}) vs "
        f"{opponent.agent_name}({opponent_team.team_id}) | "
        f"{repeats} battle(s)"
    )

    await rl_player.battle_against(
        opponent,
        n_battles=repeats,
    )

    tags = sorted(rl_player.battles)
    repeat_map = {
        battle_tag: index
        for index, battle_tag in enumerate(tags, start=1)
    }

    rows: list[dict] = []
    team_pokemon_rows: list[dict] = []
    team_move_rows: list[dict] = []

    for battle_tag in tags:
        battle = rl_player.battles[battle_tag]
        opponent_battle = opponent.battles.get(battle_tag)
        if opponent_battle is None:
            raise RuntimeError(
                f"Battle {battle_tag} missing from opponent registry."
            )

        pokemon_rows, move_rows = team_catalog(
            battle,
            generation=generation,
            team_id=rl_team.team_id,
            pair_id=pair_id,
            side=rl_team.side,
        )
        team_pokemon_rows.extend(pokemon_rows)
        team_move_rows.extend(move_rows)
        pokemon_rows, move_rows = team_catalog(
            opponent_battle,
            generation=generation,
            team_id=opponent_team.team_id,
            pair_id=pair_id,
            side=opponent_team.side,
        )
        team_pokemon_rows.extend(pokemon_rows)
        team_move_rows.extend(move_rows)

        if getattr(battle, "won", False):
            result_1, winner = "win", rl_player.agent_name
        elif getattr(battle, "lost", False):
            result_1, winner = "loss", opponent.agent_name
        else:
            result_1, winner = "draw", "draw"

        info = final_information_snapshot(
            battle,
            opponent_battle,
        )

        c1 = (
            action_counts(rl_player, battle_tag)
            if log_actions else _zero_counts()
        )
        c2 = (
            action_counts(opponent, battle_tag)
            if log_actions else _zero_counts()
        )

        rows.append({
            "run_id": run_id,
            "battle_tag": battle_tag,
            "experiment_mode": "rl_fixed_mirrored",
            "generation": generation,
            "battle_format": battle_format,
            "pair_id": pair_id,
            "mirror_leg": mirror_leg,
            "repeat_index": repeat_map[battle_tag],
            "agent_1": rl_player.agent_name,
            "agent_2": opponent.agent_name,
            "team_a_id": team_a.team_id,
            "team_b_id": team_b.team_id,
            "agent_1_team_id": rl_team.team_id,
            "agent_2_team_id": opponent_team.team_id,
            "agent_1_team_sha256": rl_team.sha256,
            "agent_2_team_sha256": opponent_team.sha256,
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

    actions: list[dict] = []
    if log_actions:
        actions.extend(
            _annotate_actions(
                rl_player,
                generation=generation,
                battle_format=battle_format,
                pair_id=pair_id,
                mirror_leg=mirror_leg,
                repeat_map=repeat_map,
                agent_team_id=rl_team.team_id,
                opponent_team_id=opponent_team.team_id,
                team_a_id=team_a.team_id,
                team_b_id=team_b.team_id,
                agent_role="rl",
            )
        )
        actions.extend(
            _annotate_actions(
                opponent,
                generation=generation,
                battle_format=battle_format,
                pair_id=pair_id,
                mirror_leg=mirror_leg,
                repeat_map=repeat_map,
                agent_team_id=opponent_team.team_id,
                opponent_team_id=rl_team.team_id,
                team_a_id=team_a.team_id,
                team_b_id=team_b.team_id,
                agent_role="opponent",
            )
        )

    for row in actions:
        row["experiment_mode"] = "rl_fixed_mirrored"

    print(
        f"  {rl_player.agent_name}: {rl_player.n_won_battles} win(s) | "
        f"{opponent.agent_name}: {opponent.n_won_battles} win(s) | "
        f"draws: {rl_player.n_tied_battles}"
    )

    return rows, actions, team_pokemon_rows, team_move_rows


async def run_experiment(args: argparse.Namespace) -> Path:
    model_path = resolve_model_file(args.model)
    model_meta = load_model_metadata(model_path)

    generation = int(args.generation)
    validate_model_metadata(
        model_meta,
        generation=generation,
        require_same_generation=not args.allow_cross_generation,
    )

    pool_root = Path(args.pool_root).resolve()
    problems = verify_pool(pool_root)
    if problems:
        raise RuntimeError(
            "Fixed-team pool failed integrity check:\n- "
            + "\n- ".join(problems)
        )

    manifest = load_pool_manifest(pool_root)
    records = records_by_key(manifest)

    available_pairs = sorted({
        pair_id
        for gen, pair_id, side in records
        if gen == generation
    })
    selected_pairs = args.pairs or available_pairs

    missing = [
        pair_id for pair_id in selected_pairs
        if pair_id not in available_pairs
    ]
    if missing:
        raise ValueError(
            f"Gen {generation} does not contain pair(s): {missing}"
        )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or f"rl_controlled_{run_id}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    all_battles: list[dict] = []
    all_actions: list[dict] = []
    all_team_pokemon: list[dict] = []
    all_team_moves: list[dict] = []
    used_records: list[TeamRecord] = []

    for pair_id in selected_pairs:
        team_a = records[(generation, pair_id, "A")]
        team_b = records[(generation, pair_id, "B")]
        used_records.extend([team_a, team_b])

        # AB: RL uses Team A.
        battles, actions, team_pokemon, team_moves = await run_rl_leg(
            generation=generation,
            pair_id=pair_id,
            mirror_leg="AB",
            model_path=model_path,
            opponent_key=args.opponent,
            rl_team=team_a,
            opponent_team=team_b,
            team_a=team_a,
            team_b=team_b,
            repeats=args.repeats,
            max_concurrent_battles=args.concurrency,
            log_actions=not args.no_action_log,
            deterministic=not args.stochastic,
            run_id=run_id,
            allow_cross_generation=args.allow_cross_generation,
        )
        all_battles.extend(battles)
        all_actions.extend(actions)
        all_team_pokemon.extend(team_pokemon)
        all_team_moves.extend(team_moves)

        # BA: same RL policy uses Team B.
        battles, actions, team_pokemon, team_moves = await run_rl_leg(
            generation=generation,
            pair_id=pair_id,
            mirror_leg="BA",
            model_path=model_path,
            opponent_key=args.opponent,
            rl_team=team_b,
            opponent_team=team_a,
            team_a=team_a,
            team_b=team_b,
            repeats=args.repeats,
            max_concurrent_battles=args.concurrency,
            log_actions=not args.no_action_log,
            deterministic=not args.stochastic,
            run_id=run_id,
            allow_cross_generation=args.allow_cross_generation,
        )
        all_battles.extend(battles)
        all_actions.extend(actions)
        all_team_pokemon.extend(team_pokemon)
        all_team_moves.extend(team_moves)

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
        write_telemetry_outputs(
            run_dir,
            actions=all_actions,
            battles=all_battles,
            team_pokemon_rows=all_team_pokemon,
            team_move_rows=all_team_moves,
        )

    raw_summary = summarize_battles(all_battles)
    if raw_summary:
        write_csv(
            run_dir / "summary.csv",
            raw_summary,
            list(raw_summary[0].keys()),
        )

    _, _, controlled_summary = write_mirrored_outputs(
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

    metadata = {
        "project": "PokeBench30",
        "version": __version__,
        "run_id": run_id,
        "experiment_mode": "rl_fixed_mirrored",
        "generation": generation,
        "training_generation": int(model_meta.training_generation),
        "evaluation_generation": generation,
        "cross_generation_evaluation": (
            int(model_meta.training_generation) != generation
        ),
        "model_path": str(model_path),
        "model_metadata": model_meta.to_dict(),
        "representation_version": model_meta.observation_schema_version,
        "opponent": args.opponent,
        "pairs": selected_pairs,
        "repeats_per_leg": args.repeats,
        "deterministic_policy": not args.stochastic,
        "fixed_team_pool": {
            "config_sha256": manifest.get("config_sha256"),
            "showdown_commit": manifest.get("showdown_commit"),
            "showdown_dirty": manifest.get("showdown_dirty"),
        },
        "notes": [
            "RL is always Agent 1 in the mirrored summary.",
            "Each team assignment is mirrored AB/BA.",
            "v0.6.1 permits zero-shot cross-generation evaluation when explicitly enabled.",
            "No model weights are changed during controlled evaluation.",
            "Telemetry distinguishes availability from selection to reduce opportunity-set confounding.",
        ],
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== RL controlled mirrored summary ===")
    for row in controlled_summary:
        print(
            f"Gen {row['generation']} | "
            f"{row['agent_1']} vs {row['agent_2']} | "
            f"mirror score RL={float(row['normalized_mirror_score_agent_1']):.3f} | "
            f"raw {row['wins_agent_1']}-{row['wins_agent_2']} "
            f"(draws={row['draws']})"
        )

    print(f"\nResults written to: {run_dir}")
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a PokéBench30 RL model with fixed teams and mirrored "
            "assignments, with optional v0.6.1 zero-shot cross-generation evaluation."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--generation",
        type=int,
        choices=[1, 4, 9],
        required=True,
    )
    parser.add_argument(
        "--opponent",
        choices=sorted(AGENT_CLASSES),
        default="type_aware",
    )
    parser.add_argument("--pairs", nargs="+", default=None)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--pool-root",
        default=str(DEFAULT_POOL_ROOT),
    )
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument(
        "--allow-cross-generation",
        action="store_true",
        help=(
            "Allow a model trained in one generation to be evaluated zero-shot "
            "in another generation. Schema/action-space compatibility is still checked."
        ),
    )
    parser.add_argument("--no-action-log", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--output-root",
        default=str(RESULTS_DIR),
        help="Directory where this evaluation run is written.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional deterministic run directory name (used by multi-seed runner).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.repeats <= 0:
        raise ValueError("--repeats must be greater than zero.")
    asyncio.run(run_experiment(args))


if __name__ == "__main__":
    main()
