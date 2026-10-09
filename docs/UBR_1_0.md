# UBR 1.0 — Universal Battle Representation

## Status

Schema: `ubr-1.0`

Observation dimension:

\[
6 + 14 + (12 \times 29) + (4 \times 26) = 472
\]

Action dimension:

\[
4 + 6 = 10
\]

The exact ordered feature list is stored in:

```text
config/observation_schema.json
```

The SHA-256 in that file identifies the exact schema.

---

## Design principles

### Fixed dimensionality

The tensor shape must not depend on generation.

```text
Gen I  -> (472,)
Gen IV -> (472,)
Gen IX -> (472,)
```

### Partial observability

The encoder receives the same information exposed to the agent.

Opponent slots not present in `battle.opponent_team` are represented as
all-zero vectors.

Evaluator ground truth is not an encoder input.

### Species agnosticism

Species identity is deliberately omitted from UBR 1.0.

The representation describes:

```text
HP
level
types
status
boosts
moves
team occupancy
legal actions
```

rather than assigning a separate dimension to each Pokémon species.

This avoids an observation vocabulary whose size grows from generation to
generation and provides a cleaner first baseline for cross-generation
transfer.

### Explicit action legality

Every state is paired with:

```text
legal_action_mask[10]
```

A policy must choose only indices with mask value `1`.

---

## Observation blocks

### Global — 6

```text
generation_norm
turn_norm
force_switch
own_remaining_fraction
opponent_revealed_fraction
opponent_revealed_remaining_fraction
```

### Active boosts — 14

Seven normalized boost stages for each active Pokémon:

```text
atk
def
spa
spd
spe
accuracy
evasion
```

### Pokémon slots — 348

Twelve slots:

```text
6 own
6 opponent
```

Each slot contains 29 values:

```text
5 scalar/binary
18 type one-hot
6 status one-hot
```

Unknown opponent slots contain 29 zeros.

### Active moves — 104

Four move slots × 26 values:

```text
present
legal
base_power_norm
accuracy
priority_norm
18 type one-hot
3 category one-hot
```

---

## Action mapping

```text
0 move_slot_1
1 move_slot_2
2 move_slot_3
3 move_slot_4
4 switch_team_slot_1
5 switch_team_slot_2
6 switch_team_slot_3
7 switch_team_slot_4
8 switch_team_slot_5
9 switch_team_slot_6
```

The six switch actions correspond to stable own-team slots.

The currently active slot is masked.

During a forced switch all four move actions are masked.

---

## Normalization

```text
generation       generation / 9
turn             min(turn / 100, 1)
HP               [0, 1]
level            level / 100
boost stage      stage / 6
base power       min(base_power / 250, 1)
accuracy         [0, 1]
priority         clip(priority / 7, -1, 1)
```

All UBR 1.0 vector values therefore lie in:

```text
[-1, 1]
```

---

## Deliberate exclusions

UBR 1.0 is a core representation rather than a complete competitive-battle
state.

The following are intentionally deferred:

```text
species identity
abilities
items
weather
terrain
entry hazards
screens
volatile effects
revealed opponent moves outside the active move interface
generation-specific special actions
```

Adding any of these changes the observation semantics and should create a new
schema version rather than silently modifying `ubr-1.0`.

---

## Validation invariants

For every observation:

```text
len(vector) == 472
len(mask)   == 10

all(vector finite)
all(vector in [-1, 1])
all(mask in {0, 1})

unknown opponent slot
    =>
all features in that slot == 0

forced switch
    =>
mask[0:4] == 0
```

For logged policy demonstrations:

```text
chosen_action_index >= 0
    =>
legal_action_mask[chosen_action_index] == 1
```

---

## Research role

UBR 1.0 is intended to be frozen before the first RL experiments.

This lets later experiments distinguish:

```text
change in learning algorithm
```

from:

```text
change in state representation
```

It also provides a fixed interface for the future transfer matrix:

```text
train Gen I  -> evaluate Gen IV
train Gen I  -> evaluate Gen IX
train Gen IV -> evaluate Gen I
...
```
