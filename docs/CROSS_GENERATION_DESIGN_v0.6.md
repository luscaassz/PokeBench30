# PokéBench30 v0.6 — Cross-Generation Benchmark Design

## Primary research question

> How does the same literature-grounded reinforcement-learning agent learn and
> perform across Pokémon Generations I, IV and IX when representation,
> algorithm, hyperparameters, training budget and evaluation protocol are held
> constant?

The primary independent variable is now:

```text
Generation I
vs
Generation IV
vs
Generation IX
```

The v0.5.x experiments are retained as pilot / engineering validation.

## Training matrix

```text
3 generations
× 5 seeds
× 900k nominal PPO steps
=
15 independently trained models
```

The PettingZoo VecEnv exposes two player perspectives. With `n_steps=2048`,
each PPO rollout contains:

```text
2048 × 2 = 4096 policy transitions
```

The expected final counter is therefore:

```text
ceil(900000 / 4096) × 4096
= 901120 transitions/model
```

Expected total:

```text
15 × 901120
= 13,516,800 policy transitions
```

Every model starts from scratch.

There is no transfer of weights between generations in v0.6.0.

## Training environment

Each generation uses its native random-battle format:

```text
Gen I   gen1randombattle
Gen IV  gen4randombattle
Gen IX  gen9randombattle
```

Both players are controlled by the same current stochastic PPO policy, and
both player trajectories enter the rollout buffer.

This removes scripted-opponent choice from the main training comparison.

## Checkpoints

Nominal checkpoints:

```text
150k
300k
450k
600k
750k
900k
```

Checkpoints are written after the first completed PPO rollout that crosses the
nominal threshold. Therefore both nominal and actual step counts are saved.

## Evaluation pool

v0.6 expands the fixed pool from 3 to **10 independent deterministic team
pairs per generation**.

The first three historical pairs are retained, and seven additional pairs are
generated from deterministic Pokémon Showdown random-team seeds.

This is important because the pair, not each repeated battle, is the stronger
independent experimental unit.

## Mirrored final evaluation

For each model/opponent:

```text
10 team pairs
× 2 mirrored assignments
× 3 repeats
=
60 battles
```

Final opponents:

```text
Random
MaxDamage
TypeAware
SwitchAware
```

Across all 15 trained models:

```text
15 × 4 × 60
= 3600 final evaluation battles
```

## Diagnostic learning curve

At each non-final checkpoint, the model is evaluated against TypeAware using:

```text
10 pairs
× 2 mirrored assignments
× 1 repeat
=
20 battles/checkpoint
```

Five non-final checkpoints × 15 models:

```text
1500 diagnostic battles
```

Expected total controlled evaluation:

```text
3600 final
+1500 diagnostic
=
5100 battles
```

## Primary outputs

### `generation_summary.csv`

Final seed-level aggregate for each:

```text
generation × opponent
```

### `generation_comparison.csv`

Paired same-seed differences for:

```text
Gen IV - Gen I
Gen IX - Gen I
Gen IX - Gen IV
```

for each benchmark opponent.

### `pair_level_final.csv`

Preserves the 10 team-pair results instead of reducing everything to individual
battle counts.

This should be used for pair-aware bootstrap / hierarchical inference in the
paper.

### `learning_efficiency_typeaware.csv`

A descriptive area-under-learning-curve measure based on periodic TypeAware
evaluation.

It is not a replacement for the final mirrored benchmark.

## Interpretation boundary

v0.6.0 compares **independently trained same-method agents** across generations.

It does not yet ask:

- whether a Gen I model transfers zero-shot to Gen IV or IX;
- whether weights can be fine-tuned across generations;
- whether generation-specific mechanics such as Terastallization improve or
  reduce performance.

Those are separate extensions and should not be mixed into the primary
benchmark.
