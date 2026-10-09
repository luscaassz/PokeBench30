# PokéBench30 v0.5 — Reinforcement Learning Baseline

## Scope

v0.5 introduces the first learned policy while freezing the representation
validated in v0.4.1:

```text
UBR 1.1
observation = 472
action space = 10
```

The initial algorithm is:

```text
MaskablePPO
```

The model is trained in a single generation and evaluated in that same
generation. Cross-generation transfer is intentionally reserved for v0.6.

---

## Why MaskablePPO

At a given turn, many of the ten UBR actions are illegal. Examples:

- disabled or unavailable moves;
- the currently active Pokémon cannot be switched into;
- fainted Pokémon cannot be selected;
- forced-switch requests forbid all move actions.

Action masking removes those actions from the policy's categorical
distribution instead of asking the network to learn legality by punishment.

---

## Training distribution

v0.5 trains on:

```text
gen1randombattle
gen4randombattle
or
gen9randombattle
```

rather than the three fixed benchmark pairs.

Reason:

```text
fixed pairs -> controlled evaluation
random teams -> training diversity
```

The fixed mirrored pool from v0.3 remains the evaluation benchmark.

---

## Reward

The first baseline uses poke-env's state-difference reward helper:

```text
victory:        30.0
fainted:         2.0
HP:              1.0
status:          0.5
```

The reward is deliberately simple and interpretable. v0.5 is meant to validate
the learning pipeline, not optimize reward engineering.

---

## Training opponent

Supported scripted opponents:

```text
random
max_damage
type_aware
switch_aware
```

The default is:

```text
type_aware
```

This provides a stronger curriculum than a purely random opponent without
already requiring self-play.

---

## Research separation

The versions now isolate different scientific factors:

```text
v0.3   experiment control
v0.4.1 state/action representation
v0.5   learning algorithm
v0.6   cross-generation transfer
```

The v0.5 model metadata stores the UBR version and SHA-256. A model whose
representation does not match the current code is rejected.

---

## Model artifact

A training run creates:

```text
models/
└── maskable_ppo_gen9_YYYYMMDD_HHMMSS/
    ├── maskable_ppo.zip
    ├── metadata.json
    ├── training_config.json
    ├── monitor.csv.monitor.csv
    ├── checkpoints/
    └── tensorboard/
```

`metadata.json` includes:

```text
training generation
training format
opponent
timesteps
seed
UBR version
UBR SHA-256
observation/action dimensions
reward weights
library versions
```

---

## Evaluation

The learned policy is wrapped in `RLPolicyAgent`, a normal poke-env Player.

Controlled evaluation uses the same fixed mirrored design as v0.3:

```text
LEG AB
RL + Team A
vs
baseline + Team B

LEG BA
RL + Team B
vs
baseline + Team A
```

The main metric remains:

```text
normalized mirrored score
```

The RL policy is always `agent_1` in the v0.5 evaluation runner.

---

## v0.5 gate

Before moving to v0.6:

1. training must complete without illegal actions;
2. reward curves must be saved;
3. model metadata must validate against UBR 1.1;
4. deterministic controlled evaluation must run against TypeAware;
5. evaluation should be repeated across several seeds before scientific claims.

A high win rate is not required to validate the pipeline. The milestone is that
a learned policy can consume UBR 1.1, respect the legal-action mask, train,
save, reload, and participate in mirrored controlled battles.
