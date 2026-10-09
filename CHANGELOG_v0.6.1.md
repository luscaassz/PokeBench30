# PokéBench30 v0.6.1

## Zero-Shot Transfer + Behavioral Telemetry

v0.6.1 does not change or retrain the v0.6.0 PPO agent.

### New experiment

A full 3×3 zero-shot matrix evaluates final 900k models across:

```text
train: Gen I / IV / IX
×
evaluate: Gen I / IV / IX
```

No fine-tuning occurs.

### New transfer outputs

- `transfer_scores_by_seed.csv`
- `transfer_matrix_summary.csv`
- `transfer_gap_by_seed.csv`
- `transfer_gap_summary.csv`
- `zero_shot_generalization_summary.csv`
- transfer score / transfer-gap heatmaps

### Robust telemetry

Action logs now record both chosen-action semantics and the currently legal move set.

Move semantics are multi-label and include damage/status, buff/debuff, healing, hazards,
priority, pivoting, recoil/drain, weather/terrain and other tactical properties.

Opportunity-adjusted metrics report:

```text
availability
selection
selection_given_available
```

for individual moves, move types, move categories and semantic move features.

### Type and pool composition

v0.6.1 adds Pokémon type exposure telemetry and fixed-pool team/move catalogs so observed
behavior can be separated from differences in what each generation's frozen evaluation teams
offered.

### Controlled evaluator

`run_rl_controlled.py` now supports:

```text
--allow-cross-generation
```

Schema and action-space compatibility are still validated. The flag only relaxes the
training-generation equality check.

### Existing v0.6.0 agent frozen

The following remain unchanged:

- PPO weights;
- UBR 1.2;
- 10-action common-core interface;
- Wang-style PPO training protocol;
- sparse reward;
- final 900k checkpoints;
- fixed 10-pair evaluation pool.

## Hotfix 1 — telemetry semantics and structural catalogs

- generation-wide Pokémon catalog now contains canonical base species only;
- National Dex historical cutoffs used for Gen I/IV/IX structural counts;
- fixed-pool manifest rows canonicalized with evaluator-equivalent last-record-wins semantics;
- fixed-pool aggregate deduplication now uses pair/side/slot identity;
- SELF-target `boosts` are routed to self-buff/self-debuff telemetry;
- added `scripts/check_telemetry_hotfix.py`;
- added regression tests for future forms/CAP filtering, cosmetic aliases, manifest duplicates, and Swords Dance self-buff classification.

## Hotfix 2 — fixed-pool structural aggregate isolation

Hotfix 1 fixed manifest canonicalization but the transfer-level aggregate still mixed two
semantically different sources:

- immutable packed-team rows (`catalog_source=packed_fixed_team`);
- post-battle team observations (`catalog_source=battle_observation`).

Across mirrored/repeated smoke evaluations, the latter could introduce observed slots 7–12
for pair01–pair03, producing 156 Pokémon rows per generation instead of 120.

Hotfix 2:

- fixed-pool structural catalogs now prefer **only** `packed_fixed_team` rows whenever present;
- battle-derived team rows are explicitly tagged `catalog_source=battle_observation`;
- canonical identity is exactly `generation + pair_id + side + team_slot`;
- move identity additionally includes `move_slot`;
- aggregate generation now fails fast if any duplicate key remains;
- aggregate generation fails if a 10-pair generation does not contain exactly 120 Pokémon rows;
- added regression reproducing the observed pair01–03 slot-7–12 contamination;
- checker now validates the final aggregate path, not only the manifest-level static catalog.

No PPO model, reward, UBR, fixed team, transfer score, or training artifact is changed.

## Hotfix 3 — finite-safe UBR observations

- Prevented transient non-finite battle scalars from propagating to PPO logits.
- Added a final UBR finite invariant with diagnostic feature names.
- Added a policy-boundary non-finite guard for vector-wrapper robustness.
- No scientific-condition change for finite observations; completed models remain valid.
