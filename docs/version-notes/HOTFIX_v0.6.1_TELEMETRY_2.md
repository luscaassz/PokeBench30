# PokéBench30 v0.6.1 — Telemetry Hotfix 2

## Symptom

After Hotfix 1, the generation-wide species catalog and SELF-target buff semantics were
correct, but `aggregate/telemetry/fixed_pool_pokemon_catalog.csv` still contained:

```text
156 Pokémon rows / generation
pair01–pair03: 12 Pokémon rows per side
pair04–pair10: 6 Pokémon rows per side
```

The expected geometry is always:

```text
10 pairs × 2 sides × 6 Pokémon = 120 Pokémon rows / generation
```

## Root cause

`fixed_pool_catalog_from_manifest()` already produced the correct immutable 120-row
catalog. The error happened later in transfer aggregation.

The aggregate mixed:

```text
packed_fixed_team     <- structural source of truth
battle_observation    <- post-battle/transformed observations
```

Post-battle rows are useful behavioral telemetry, but are not valid evidence for the
composition of the frozen benchmark pool. Repeated/mirrored evaluations could therefore
introduce extra observed slots and transformed state.

## Fix

The structural fixed-pool aggregate now:

1. selects `catalog_source=packed_fixed_team` exclusively whenever available;
2. deduplicates Pokémon by `generation,pair_id,side,team_slot`;
3. deduplicates moves by `generation,pair_id,side,team_slot,move_slot`;
4. validates 120 Pokémon rows per generation for the 10-pair benchmark;
5. rejects duplicate Pokémon or move-slot keys;
6. rejects move slots outside 1–4;
7. explicitly labels runtime team telemetry as `battle_observation`.

## Regression against the observed smoke aggregate

Applying the Hotfix 2 canonicalizer to the contaminated Hotfix 1 smoke catalog reduces:

```text
Pokémon: 156 -> 120 per generation
```

and removes all battle-observation rows from the structural catalog.

Move counts are allowed to be below 480 because some frozen Random Battle sets can expose
fewer than four packed moves; the required invariant is unique move slots 1–4 per Pokémon,
not an artificial 4-move count for every row.

## Scientific scope

This is a telemetry aggregation correction only. It does not change:

- the 15 trained PPO models;
- v0.6.0 training;
- transfer inference;
- transfer scores;
- UBR 1.2;
- action space;
- fixed teams;
- opponent implementations.

The previous Hotfix 1 smoke aggregate should be regenerated before scientific use.
