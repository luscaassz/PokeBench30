# PokéBench30 v0.6.1 — Telemetry Hotfix 1

This hotfix corrects three telemetry problems discovered in the first v0.6.1 smoke run.

## 1. Generation-wide Pokémon catalog

### Problem

The first implementation iterated directly over `GenData.from_gen(...).pokedex`.
Those files are useful for battle parsing, but they may contain aliases, cosmetic rows,
later-generation forms and nonstandard entries. As a result, the Gen I structural catalog
incorrectly contained entries such as Mega/regional forms and CAP species.

### Fix

The broad generation catalog is now deliberately defined as:

```text
canonical base species introduced by that generation
```

The filter requires:

- a positive National Dex number;
- a National Dex number not exceeding the historical generation cutoff;
- no `forme` value;
- a canonical species id matching the row name/base species.

Expected counts are therefore:

```text
Gen I    151
Gen IV   493
Gen IX  1025
```

This broad catalog is **not** a Random Battles legality list. Form-level and benchmark-specific
ecology is measured from the frozen fixed-team pool instead.

## 2. Duplicated fixed-pool rows

### Problem

Legacy pool manifests can contain multiple records for the same:

```text
generation + pair_id + side
```

The battle evaluator already resolves this with last-record-wins semantics, but the telemetry
catalog previously traversed every manifest row. This double-weighted some historical pairs.

### Fix

Static telemetry now uses the exact same canonicalization rule as the evaluator:

```text
(generation, pair_id, side) -> last manifest record
```

The aggregate catalog also deduplicates Pokémon by:

```text
evaluation_generation + pair_id + side + team_slot
```

and moves by:

```text
evaluation_generation + pair_id + side + team_slot + move_slot
```

For the current benchmark, each generation must therefore contain:

```text
10 pairs × 2 sides × 6 Pokémon = 120 Pokémon rows
```

## 3. SELF-target boosts/debuffs

### Problem

Showdown-style move metadata stores main-stat changes in `boosts`, while the semantic target
is stored separately. For moves such as Swords Dance, Agility and Acid Armor:

```text
target = SELF
boosts = {... positive stat change ...}
```

The original v0.6.1 telemetry treated all `boosts` as target effects, incorrectly classifying
Swords Dance as `target_boost`.

### Fix

Main boosts are now routed according to the move target.

For `SELF`-target moves:

```text
Swords Dance
self_boost   = True
target_boost = False
any_buff     = True
```

Targeted debuffs such as Growl remain target debuffs.

## Scientific impact

This hotfix changes **telemetry only**.

It does not modify:

- the 15 trained v0.6.0 models;
- PPO weights;
- UBR 1.2;
- action space;
- training protocol;
- zero-shot transfer definition;
- fixed battle teams;
- mirror geometry;
- opponent policies.

No model retraining is required.

The initial v0.6.1 smoke telemetry should be discarded and regenerated after applying this hotfix.
