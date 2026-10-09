# PokéBench30 v0.5 reference baseline

This file records the Gen IX single-seed result that motivated UBR 1.2.

## Training

```text
Algorithm: MaskablePPO
Training generation: IX
Training distribution: gen9randombattle
Training opponent: TypeAwareAgent
Requested timesteps: 10,000
Effective PPO rollout total: 10,240
Seed: 42
Representation: UBR 1.1
Observation dim: 472
Action dim: 10
```

## Fixed mirrored evaluation

Three fixed team pairs, 10 repeats per mirror leg:

```text
3 pairs × 2 legs × 10 = 60 battles / opponent
```

### Against RandomAgent

```text
pair01: 16/20 RL wins
pair02: 16/20 RL wins
pair03: 13/20 RL wins

Total: 45-15
Mirrored score: 0.750
```

### Against TypeAwareAgent

```text
pair01: 3/20 RL wins
pair02: 2/20 RL wins
pair03: 0/20 RL wins

Total: 5-55
Mirrored score: 0.083
```

## Interpretation

The v0.5 model showed non-random learned behavior, but inspection of its action
logs revealed repeated persistent-effect moves such as Sticky Web, Stealth Rock
and Taunt. UBR 1.1 did not expose whether those effects were already active.

v0.5.1 therefore keeps the learning algorithm fixed and changes the
representation to UBR 1.2 so battle-context information can be tested as an
ablation.
