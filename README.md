# PokéBench30 v0.4.1 — UBR 1.1

PokéBench30 v0.4.1 is a corrective release of the Universal Battle
Representation introduced in v0.4.

The experimental design from v0.3/v0.4 is preserved:

```text
fixed teams
+ mirrored team assignment
+ partial information
+ TypeAwareAgent / SwitchAwareAgent baselines
+ Gen I / Gen IV / Gen IX
```

The change is focused on one issue discovered by the first real UBR smoke test:
**Gen I can expose transient legal commands such as `recharge` and `fight` that
are not part of the Pokémon's persistent four-move dictionary.**

UBR 1.0 could therefore produce an observation where the action selected by the
agent had no universal action index. UBR 1.1 fixes that while keeping the same
fixed interface:

```text
observation_dim = 472
action_dim      = 10
```

---

## 1. What changed from v0.4

The original smoke test produced:

```text
Records: 261
Chosen actions mapped into UBR: 253/261
```

The eight unmapped Gen I actions were:

```text
5 × fight
3 × recharge
```

The core problem was that UBR 1.0 started with the four persistent moves from
`active_pokemon.moves`, appended transient moves afterward, and then truncated
the list back to four slots.

In UBR 1.1, ordinary states keep the persistent move order, but any legal move
reported by Showdown is guaranteed to receive one of the four universal move
slots.

---

## 2. UBR 1.1 move-slot policy

For a normal state:

```text
slot 1 = persistent move 1
slot 2 = persistent move 2
slot 3 = persistent move 3
slot 4 = persistent move 4
```

If Showdown provides a transient legal move that is absent from those four:

```text
available_moves = [recharge]
```

UBR 1.1 replaces a persistent move that is currently unavailable:

```text
slot 1 = recharge      legal = 1
slot 2 = Rest          legal = 0
slot 3 = Earthquake    legal = 0
slot 4 = Body Slam     legal = 0
```

The important properties are:

```text
- every legal move is representable;
- legal persistent moves are never discarded;
- ordinary move ordering is preserved whenever possible;
- the action space remains 10-dimensional;
- no special Gen I-only action index is introduced.
```

If Showdown ever exposes more than four simultaneous legal moves, the encoder
now raises an explicit error instead of silently dropping one.

---

## 3. Universal observation

UBR 1.1 still produces 472 normalized features for Gen I, IV and IX.

```text
Global state                 6
Active Pokémon boosts       14
Own team: 6 × 29           174
Opponent: 6 × 29           174
Active moves: 4 × 26       104
──────────────────────────────
TOTAL                       472
```

Unknown opponent slots remain exactly zero-filled.

Species identity remains excluded.

The representation still does not reconstruct hidden opponent information.

---

## 4. Universal action space

The action space is unchanged:

```text
0 → move_slot_1
1 → move_slot_2
2 → move_slot_3
3 → move_slot_4

4 → switch_team_slot_1
5 → switch_team_slot_2
6 → switch_team_slot_3
7 → switch_team_slot_4
8 → switch_team_slot_5
9 → switch_team_slot_6
```

During forced switch:

```text
mask[0:4] = 0
```

The currently active team slot is also masked from the switch actions.

---

## 5. Schema version

Because move-slot semantics changed, the schema was intentionally bumped:

```text
UBR 1.0 → UBR 1.1
```

Current schema:

```text
schema_version  = ubr-1.1
observation_dim = 472
action_dim      = 10
```

The exact SHA-256 is stored in:

```text
config/observation_schema.json
```

Do not mix UBR 1.0 and UBR 1.1 datasets silently, even though both have the
same tensor dimensions.

---

## 6. Strict dataset validation

The v0.4 validator accepted `253/261` mapped actions because unmapped actions
were only reported informationally.

The v0.4.1 validator is strict.

It now fails if any of these are nonzero:

```text
schema mismatches
vector dimension errors
mask dimension errors
non-finite vectors
empty legal-action masks
unmapped chosen actions
chosen actions marked illegal
```

A successful dataset must therefore contain:

```text
Chosen actions mapped into UBR: X/X
```

for every generation.

If an unmapped action exists and `actions.csv` is present, the validator also
prints diagnostic examples with generation, turn, battle tag and the original
action name.

---

## 7. Installation

On Windows + VS Code:

```powershell
cd C:\Users\lucas\OneDrive\Desktop\PokeBench_versions\PokeBench30_v0.4.1
python -m pip install -r requirements.txt
```

Then verify the environment:

```powershell
python scripts\check_setup.py
```

---

## 8. Fixed teams

You can reuse the same fixed-team pool from v0.4/v0.3 by copying:

```text
teams\fixed
```

or regenerate it:

```powershell
python prepare_fixed_teams.py --showdown-dir "C:\Users\lucas\OneDrive\Desktop\pokemon-showdown" --force
```

Then:

```powershell
python scripts\check_fixed_pool.py
```

---

## 9. Offline UBR test

Before starting Showdown:

```powershell
python scripts\check_representation.py
```

Expected structure:

```text
[OK] UBR schema: ubr-1.1
[OK] Observation dimension: 472
[OK] Action dimension: 10
[OK] Gen 1: shape=(472,), ...
[OK] Gen 4: shape=(472,), ...
[OK] Gen 9: shape=(472,), ...

[OK] Universal representation offline checks passed.
```

---

## 10. Unit tests

Run:

```powershell
python -m unittest discover tests -v
```

The v0.4.1 suite adds explicit tests for:

```text
recharge with four persistent moves
fight with four persistent moves
normal move-order preservation
```

alongside the existing tests for:

```text
fixed-team configuration
mirrored metrics
partial observability
zero-filled hidden opponent slots
forced switch masking
schema dimensions
representation determinism
```

---

## 11. Real smoke test

With the local Pokémon Showdown server running:

```powershell
python run_controlled.py --generations 1 4 9 --matchup type_aware switch_aware --pairs pair01 --repeats 1 --log-universal-state
```

This creates six battles:

```text
3 generations × 1 team pair × 2 mirrored legs = 6 battles
```

The result directory should include:

```text
actions.csv
agent_summary.csv
battles.csv
controlled_summary.csv
metadata.json
mirror_cycles.csv
observation_schema.json
pair_summary.csv
summary.csv
universal_states.jsonl
```

---

## 12. Validate the new smoke test

After the run finishes:

```powershell
python scripts\validate_universal_states.py "results\controlled_YYYYMMDD_HHMMSS"
```

The target is now strictly:

```text
[INFO] Chosen actions mapped into UBR: X/X
[INFO] Gen 1: X/X actions mapped
[INFO] Gen 4: X/X actions mapped
[INFO] Gen 9: X/X actions mapped

[OK] schema mismatches: 0
[OK] dimension errors: 0
[OK] mask errors: 0
[OK] non-finite vectors: 0
[OK] empty legal-action masks: 0
[OK] unmapped chosen actions: 0
[OK] chosen actions marked illegal: 0

[OK] Universal-state dataset passed strict UBR validation.
```

Only after this passes should UBR 1.1 be frozen for the RL phase.

---

## 13. Project structure

```text
PokeBench30_v0.4.1/
│
├── pokebench30/
│   ├── agents/
│   ├── evaluation/
│   ├── experiments/
│   ├── representation/
│   │   ├── action_space.py
│   │   ├── encoder.py
│   │   ├── schema.py
│   │   └── validator.py
│   └── teams/
│
├── config/
│   ├── generations.yaml
│   ├── fixed_teams.yaml
│   ├── universal_representation.yaml
│   └── observation_schema.json
│
├── docs/
│   ├── UBR_1_0.md
│   └── UBR_1_1.md
│
├── scripts/
│   ├── check_setup.py
│   ├── check_fixed_pool.py
│   ├── check_representation.py
│   └── validate_universal_states.py
│
├── tests/
├── prepare_fixed_teams.py
├── run_controlled.py
└── run_pilot.py
```

---

## 14. Gate before v0.5

Do not begin the RL baseline until the following checklist is satisfied:

```text
[ ] offline representation test passes
[ ] all unit tests pass
[ ] live Gen I / IV / IX smoke test completes
[ ] all vectors have shape (472,)
[ ] all masks have shape (10,)
[ ] zero NaN / inf
[ ] zero hidden-opponent leakage
[ ] zero empty legal-action masks
[ ] zero unmapped chosen actions
[ ] zero chosen actions marked illegal
[ ] every generation reports mapped_actions == records
```

When these pass, UBR 1.1 is ready to be frozen and used as the input/action
interface for PokéBench30 v0.5.
