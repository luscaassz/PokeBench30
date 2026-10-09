from pathlib import Path
import argparse, zipfile, json, hashlib
import numpy as np
import pandas as pd

def records(df):
    return df.replace({np.nan: None}).to_dict(orient="records")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("aggregate_zip", help="Path to PokéBench30 transfer aggregate ZIP")
    ap.add_argument("--out", default="data/battle_lab_data.json")
    args = ap.parse_args()

    source = Path(args.aggregate_zip)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(source, "r") as z:
        def csv(name): return pd.read_csv(z.open(name))
        transfer = csv("aggregate/transfer_matrix_summary.csv")
        gaps = csv("aggregate/transfer_gap_summary.csv")
        behavior = csv("aggregate/telemetry/behavior_summary.csv")
        dynamics = csv("aggregate/telemetry/battle_dynamics_summary.csv")
        moves = csv("aggregate/telemetry/move_summary.csv")
        move_types = csv("aggregate/telemetry/move_type_summary.csv")
        pokemon_types = csv("aggregate/telemetry/pokemon_type_summary.csv")
        dex = csv("aggregate/telemetry/generation_dex_overview.csv")
        pool_features = csv("aggregate/telemetry/fixed_pool_move_feature_summary.csv")
        pool_types = csv("aggregate/telemetry/fixed_pool_pokemon_type_summary.csv")
        pool_pokemon = csv("aggregate/telemetry/fixed_pool_pokemon_catalog.csv")
        pool_move_cat = csv("aggregate/telemetry/fixed_pool_move_category_summary.csv")
        pool_move_presence = csv("aggregate/telemetry/fixed_pool_move_presence_summary.csv")
        zero_shot = csv("aggregate/zero_shot_generalization_summary.csv")

    behavior_cols = [
        "training_generation","evaluation_generation","opponent",
        "mean_move_rate","mean_switch_rate","mean_voluntary_switch_rate",
        "mean_forced_switch_rate","mean_mean_selected_base_power",
        "mean_mean_selected_accuracy","mean_stab_move_rate",
        "mean_super_effective_rate","mean_neutral_effective_rate",
        "mean_resisted_rate","mean_selected_damaging_rate",
        "mean_selected_status_rate","mean_selected_any_buff_rate",
        "mean_selected_any_debuff_rate",
        "mean_selection_given_available_damaging",
        "mean_selection_given_available_status",
        "mean_selection_given_available_any_buff",
        "mean_selection_given_available_any_debuff",
    ]
    dynamics_cols = [
        "training_generation","evaluation_generation","opponent",
        "mean_raw_win_rate_rl","mean_mean_turns","mean_mean_action_requests_rl",
        "mean_mean_moves_rl","mean_mean_switches_rl","mean_mean_voluntary_switches_rl",
        "mean_mean_remaining_rl","mean_mean_remaining_opponent",
        "mean_mean_opponent_reveal_fraction"
    ]

    payload = {
        "meta": {
            "project": "PokéBench30 Battle Lab",
            "source": source.name,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "generations": [1,4,9],
            "generationLabels": {"1":"Gen I","4":"Gen IV","9":"Gen IX"},
            "opponents": sorted(transfer["opponent"].unique().tolist()),
            "seeds": int(transfer["n_seeds"].max()),
            "totalControlledBattles": int(transfer["total_battles"].sum()),
            "notes": {
                "replay": "Aggregate-only export: real teams and aggregate profiles are available; literal turn-by-turn replay requires raw actions.csv."
            }
        },
        "transferMatrix": records(transfer),
        "transferGap": records(gaps),
        "zeroShot": records(zero_shot),
        "behavior": records(behavior[[c for c in behavior_cols if c in behavior.columns]]),
        "battleDynamics": records(dynamics[[c for c in dynamics_cols if c in dynamics.columns]]),
        "moves": records(moves[[
            "training_generation","evaluation_generation","opponent","move_id",
            "mean_selected_count","mean_available_decisions",
            "mean_selection_given_available","mean_selected_share"
        ]]),
        "moveTypes": records(move_types[[
            "training_generation","evaluation_generation","opponent","move_type",
            "mean_selected_count","mean_available_decisions",
            "mean_selection_given_available","mean_selected_share"
        ]]),
        "pokemonTypes": records(pokemon_types[[
            "training_generation","evaluation_generation","opponent","context",
            "pokemon_type","mean_count","mean_share"
        ]]),
        "dexOverview": records(dex),
        "fixedPoolFeatures": records(pool_features),
        "fixedPoolPokemonTypes": records(pool_types),
        "fixedPoolPokemon": records(pool_pokemon[[
            "generation","pair_id","side","team_slot","species",
            "type_1","type_2","ability","item"
        ]]),
        "fixedPoolMoveCategories": records(pool_move_cat),
        "fixedPoolMovePresence": records(pool_move_presence),
    }

    out.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(out)

if __name__ == "__main__":
    main()
