# UBR 1.2 — Universal Battle Representation + Battle Context

UBR 1.2 is introduced in PokéBench30 v0.5.1.

```text
schema_version   = ubr-1.2
observation_dim  = 518
action_dim       = 10
schema_sha256    = d4b3858aab3a1c3bb1e17d9f34becf93814be05bfc949d52023651035f20e4c3
```

## Motivation

UBR 1.1 was structurally validated across Gen I, IV and IX and supported the
first MaskablePPO baseline. The v0.5 RL experiments exposed a representation
limitation: a memoryless policy could not observe whether several persistent
battle effects were already active.

Examples seen in the v0.5 baseline included repeated use of Sticky Web,
Stealth Rock and Taunt. More training alone cannot fully solve an observation
feature that is absent.

UBR 1.2 therefore adds a compact public battle-context block while preserving
the complete UBR 1.1 core.

## Compatibility rule

The first **472** values are exactly the UBR 1.1 feature ordering.

UBR 1.2 appends **46** values:

```text
472 core
+ 16 side-condition features
+  8 weather features
+  6 field features
+ 16 active volatile-effect features
= 518
```

This makes controlled UBR 1.1 vs UBR 1.2 ablations easier to interpret.

## Side conditions

The following are encoded for **both the player's side and opponent's side**:

```text
stealth_rock
spikes
toxic_spikes
sticky_web
reflect
light_screen
aurora_veil
tailwind
```

Encoding:

```text
Stealth Rock, Sticky Web, screens, Tailwind → 0/1
Spikes                                     → layers / 3
Toxic Spikes                               → layers / 2
```

The encoder reads only the battle state visible to that player.

## Weather

Eight weather indicators are appended:

```text
rain
sun
sandstorm
hail
snow
primordial_sea
desolate_land
delta_stream
```

The encoder accepts the normal poke-env/Showdown enum naming variants and
stores binary presence.

## Field / terrain context

```text
electric_terrain
grassy_terrain
misty_terrain
psychic_terrain
trick_room
gravity
```

All are binary presence indicators.

## Active volatile effects

For both currently active Pokémon:

```text
taunt
encore
substitute
confusion
leech_seed
disable
yawn
protect
```

These are binary presence indicators. They intentionally cover a small,
high-value set rather than attempting to serialize every Showdown effect.

## Partial observability

UBR 1.2 does **not** reveal hidden opponent Pokémon or reconstruct data the
agent has not observed.

The context extension uses public battle state already tracked by poke-env:

```text
battle.side_conditions
battle.opponent_side_conditions
battle.weather
battle.fields
battle.active_pokemon.effects
battle.opponent_active_pokemon.effects
```

## Still not encoded

UBR 1.2 remains a compact baseline, not a complete Markov state. It still
omits, among other things:

```text
species identity
items
abilities
move PP
known opponent move history
many rare side conditions / volatile effects
generation-specific gimmick actions
```

Those omissions should be versioned explicitly in later UBR revisions rather
than silently changing this schema.

## Action space

The 10-action space is unchanged:

```text
0..3  move slots 1..4
4..9  own team slots 1..6
```

The UBR 1.1 transient-move rule remains in force, so Gen I `recharge` and
`fight` stay representable without changing action dimensionality.

## Scientific comparison

The intended first v0.5.1 experiment keeps the learning setup fixed:

```text
MaskablePPO
Gen IX
training opponent = TypeAware
seed = 42
requested timesteps = 10,000
same reward
same network / PPO hyperparameters
```

and changes only:

```text
UBR 1.1 (472)
→
UBR 1.2 (518)
```

The v0.5 reference result was:

```text
Opponent        Wins / 60    Mirrored score
RandomAgent       45/60          0.750
TypeAwareAgent     5/60          0.083
```

These are a single-seed baseline and should not be treated as final statistical
claims.
