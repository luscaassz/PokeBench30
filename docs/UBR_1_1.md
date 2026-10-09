# UBR 1.1 — Universal Battle Representation

UBR 1.1 is the representation used by PokéBench30 v0.4.1.

It keeps the same tensor sizes introduced by UBR 1.0:

```text
observation_dim = 472
action_dim      = 10
```

The change is semantic and affects the four move slots.

## Why UBR 1.1 exists

A live v0.4 smoke test exposed eight Gen I decisions that were legal in
Pokémon Showdown but could not be mapped into UBR 1.0:

```text
5 × fight
3 × recharge
```

These are transient simulator-provided commands. They may appear in
`battle.available_moves` even when they are not part of the four persistent
moves stored in `active_pokemon.moves`.

UBR 1.0 appended such moves after the known moves and truncated the list to
four slots, which could silently discard the transient legal action.

## UBR 1.1 move-slot rule

Ordinary states preserve the ordering of `active_pokemon.moves`.

If Showdown exposes a legal transient move that is absent from those known
moves:

1. if a free move slot exists, the transient move is appended;
2. otherwise, the transient move replaces a known move that is currently
   unavailable;
3. a legal move is never replaced by another legal move;
4. if Showdown ever exposes more than four simultaneously legal moves, UBR
   fails loudly instead of silently dropping an action.

Example:

```text
persistent moves:
slot 1 = Hyper Beam
slot 2 = Rest
slot 3 = Earthquake
slot 4 = Body Slam

Showdown state after Hyper Beam:
available_moves = [recharge]
```

UBR 1.1 becomes:

```text
slot 1 = recharge     legal=1
slot 2 = Rest         legal=0
slot 3 = Earthquake   legal=0
slot 4 = Body Slam    legal=0
```

The universal action still uses index 0 (`move_slot_1`). No new action is
added and the 10-action interface remains fixed.

## Strict validation

The v0.4.1 dataset validator now requires all of the following:

```text
schema mismatches            = 0
dimension errors             = 0
mask errors                  = 0
non-finite vectors           = 0
empty legal-action masks     = 0
unmapped chosen actions      = 0
chosen actions marked illegal= 0
```

Therefore a result such as:

```text
Chosen actions mapped into UBR: 253/261
```

is now a validation failure rather than an informational message.

## Unchanged properties

UBR 1.1 keeps:

- 472-dimensional normalized observation vectors;
- 10 universal actions;
- six stable own-team switch slots;
- zero padding for unrevealed opponent slots;
- species-agnostic representation;
- no reconstruction of hidden opponent information;
- no Mega/Z/Dynamax/Tera actions in the scripted baseline action space.

The schema version bump changes the schema SHA-256 even though the feature
names and tensor dimensions are unchanged. This prevents UBR 1.0 and UBR 1.1
datasets from being mixed silently.
