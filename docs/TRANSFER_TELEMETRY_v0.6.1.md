# PokéBench30 v0.6.1 — Zero-Shot Transfer + Behavioral Telemetry

## Goal

v0.6.1 does **not train a new RL agent**.

It reuses the 15 final v0.6.0 models and asks two complementary questions:

1. **Transfer:** does knowledge learned in one generation transfer zero-shot to another?
2. **Behavior:** what exactly changes in the action ecology and learned strategy across generations?

The full transfer matrix is:

```text
TRAIN ↓ / EVALUATE →   Gen I   Gen IV   Gen IX
Gen I                    ✓       ✓        ✓
Gen IV                   ✓       ✓        ✓
Gen IX                   ✓       ✓        ✓
```

The diagonal is re-evaluated instead of copied from v0.6.0 so all nine cells have the
same telemetry instrumentation.

## Primary evaluation

Default opponents:

```text
MaxDamage
TypeAware
```

These are the primary conditions because v0.6.0 showed that Random is close to a ceiling
and SwitchAware is close to a floor. The supplementary opponents remain available with:

```powershell
python run_transfer_telemetry.py ... --all-opponents
```

Default primary workload:

```text
9 train/eval generation cells
× 5 seeds
× 2 opponents
× 10 team pairs
× 2 mirror legs
× 3 repeats
= 5,400 battles
```

All four opponents:

```text
10,800 battles
```

No weights are updated during any of these battles.

## Transfer metrics

For each seed and opponent, v0.6.1 records the raw zero-shot mirrored score.

It also computes the main transfer metric:

```text
transfer_gap(A → B)
=
score(model trained in A, evaluated in B)
-
score(model trained in B, evaluated in B)
```

Interpretation:

```text
 0.00   transfer matches an in-domain model
-0.10   10 percentage-point transfer penalty
+0.05   transferred model exceeds the same-seed in-domain model by 5 pp
```

The same seed is used for the paired comparison.

## Telemetry philosophy

The core distinction is:

```text
WHAT EXISTS IN THE GENERATION (poke-env GenData)
                ↓
WHAT THE FROZEN BENCHMARK POOL CONTAINS
                ↓
WHAT WAS LEGAL AT THAT DECISION
                ↓
WHAT THE POLICY SELECTED
```

This prevents misleading conclusions based only on raw frequencies.

Example:

```text
Gen I:  Psychic move available in 30% of decisions, selected in 20%
Gen IX: Psychic move available in 10% of decisions, selected in 8%
```

Raw use suggests 20% vs 8%, but the opportunity-adjusted preference is:

```text
Gen I:  20 / 30 = 66.7%
Gen IX:  8 / 10 = 80.0%
```

Therefore v0.6.1 reports:

- availability rate;
- selected rate;
- selection given available.

## Multi-label move semantics

A move is not forced into one exclusive category.

A single move can be simultaneously:

```text
damaging
+ target debuff
+ status infliction
+ drain
+ priority
```

Tracked semantic flags include:

- damaging;
- status move;
- self buff;
- self debuff;
- target buff;
- target debuff;
- any buff;
- any debuff;
- status infliction;
- volatile-status application;
- healing;
- drain;
- recoil;
- hazard setting;
- hazard removal;
- weather setting;
- terrain setting;
- priority;
- protect-like;
- pivot-like;
- forced-switch / phazing behavior.

For each chosen move the logger also stores:

- move ID;
- type;
- physical / special / status category;
- base power;
- accuracy;
- priority value;
- target;
- status / volatile status;
- boosts and self-boosts;
- STAB;
- type effectiveness.

## Legal-action opportunity telemetry

Each RL decision records a compact JSON description of every currently legal move.

A compressed exploded table is also written:

```text
telemetry_move_opportunities.csv.gz
```

One row represents:

```text
one RL decision × one legal move
```

and contains a `selected` flag.

This makes later analyses possible without rerunning battles.

## Pokémon typing telemetry

At action level, v0.6.1 records:

- active Pokémon type 1 / type 2;
- opponent active type 1 / type 2;
- chosen switch target type 1 / type 2.

It summarizes active-type exposure and switch-target type distributions separately.


## Generation-wide dex catalog

v0.6.1 reads poke-env's per-generation `GenData.moves` and `GenData.pokedex` and writes:

```text
generation_pokemon_dex_catalog.csv
generation_move_dex_catalog.csv
generation_dex_overview.csv
generation_pokemon_dex_type_summary.csv
generation_move_dex_type_summary.csv
generation_move_dex_category_summary.csv
generation_move_dex_feature_summary.csv
```

This structural layer is independent of the frozen teams and the learned policy. It can be used to
ask whether later generations contain different proportions of damaging/status, buff/debuff,
move types or Pokémon types before considering what appears in the benchmark pool.

## Fixed-pool structural catalog

Every controlled evaluation also snapshots the complete **own team** from each side.

After deduplication, v0.6.1 produces one static catalog for the frozen evaluation pool:

```text
fixed_pool_pokemon_catalog.csv
fixed_pool_move_catalog.csv
fixed_pool_pokemon_type_summary.csv
fixed_pool_move_type_summary.csv
fixed_pool_move_category_summary.csv
fixed_pool_move_feature_summary.csv
fixed_pool_move_presence_summary.csv
```

This lets the paper distinguish:

```text
structural difference in the benchmark pool
vs
behavioral difference in the learned policy
```

Important scope note: this catalog describes the **20 frozen evaluation teams per generation**.
It is not claimed to be the complete competitive metagame or the entire set of legal moves in
a generation.

## Main outputs

```text
transfer_results/
└── transfer_.../
    ├── study_manifest.json
    ├── runs/
    │   └── train_genX/eval_genY/seed_Z/vs_OPP/evaluation/
    │       ├── battles.csv
    │       ├── actions.csv
    │       ├── telemetry_behavior_summary.csv
    │       ├── telemetry_move_usage.csv
    │       ├── telemetry_move_type_usage.csv
    │       ├── telemetry_move_category_usage.csv
    │       ├── telemetry_move_feature_usage.csv
    │       ├── telemetry_pokemon_type_exposure.csv
    │       ├── telemetry_team_pokemon.csv
    │       ├── telemetry_team_moves.csv
    │       └── telemetry_move_opportunities.csv.gz
    └── aggregate/
        ├── transfer_scores_by_seed.csv
        ├── transfer_matrix_summary.csv
        ├── transfer_gap_by_seed.csv
        ├── transfer_gap_summary.csv
        ├── zero_shot_generalization_summary.csv
        ├── figures/
        └── telemetry/
            ├── behavior_by_seed.csv
            ├── behavior_summary.csv
            ├── move_summary.csv
            ├── move_type_summary.csv
            ├── move_category_summary.csv
            ├── move_feature_summary.csv
            ├── pokemon_type_summary.csv
            ├── fixed_pool_pokemon_catalog.csv
            ├── fixed_pool_move_catalog.csv
            └── ...
```

## Statistical unit

The study retains seed-level results and the ten fixed team pairs.

Repeated battles must not be treated as fully independent biological/statistical units.
The primary paper analysis should continue to use seed- and pair-aware inference rather than
naive binomial significance over all repeated battles.
