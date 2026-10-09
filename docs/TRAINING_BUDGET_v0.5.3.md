# PokéBench30 v0.5.3 — Training Budget Study

## Research question

The v0.5.2 experiment found essentially identical mean external performance for
UBR 1.1 and UBR 1.2 at 10k timesteps.

v0.5.3 asks:

> Does the richer UBR 1.2 representation benefit more from a larger training
> budget than UBR 1.1?

The default comparison is:

```text
10k timesteps
vs
50k timesteps
```

for:

```text
UBR 1.1
UBR 1.2
```

with five seeds.

Because PPO collects full rollouts (`n_steps=2048` by default), the nominal
budgets correspond to approximately:

```text
10,000 requested → 10,240 collected timesteps
50,000 requested → 51,200 collected timesteps
```

The study labels conditions by the requested budget so the experiment
configuration remains explicit and reproducible.

## Constants held fixed

- Gen IX
- MaskablePPO
- MLP architecture
- learning rate
- reward
- action space
- TypeAware training opponent
- Random + TypeAware evaluation opponents
- fixed-team pool
- mirrored evaluation protocol
- five PPO seeds

Only the requested training budget changes.

## Important design choice

The 50k models are trained **from scratch**.

They are not warm-started from the saved 10k models.

This avoids turning the comparison into:

```text
fresh 10k
vs
10k checkpoint + another 40k under a restarted environment
```

and keeps each budget as an independent training condition.

Because Pokémon Showdown battle-event RNG is not common-random-number paired,
the 10k and 50k models are repeated stochastic runs rather than nested,
identical trajectories.

## Reusing the completed v0.5.2 baseline

If the 10k v0.5.2 experiment already exists, v0.5.3 can import its aggregate
records and train only the 50k models.

Example:

```powershell
python run_budget_study.py --baseline-experiment-dir "C:\...\PokeBench30_v0.5.2\multiseed_results\multiseed_gen9_YYYYMMDD_HHMMSS"
```

The importer verifies:

- generation
- 10k budget
- representations
- seeds
- train/evaluation opponents
- repeats
- PPO rollout/batch settings
- learning rate
- fixed-team config hash
- Pokémon Showdown commit

If these do not match, import is rejected.

## Default workload with imported 10k baseline

```text
2 representations
× 5 seeds
× 50k timesteps
=
10 new models

10 models
× 2 evaluation opponents
× 60 battles
=
1,200 new evaluation battles
```

Without baseline reuse, both 10k and 50k are trained:

```text
20 models
2,400 evaluation battles
```

## Main outputs

```text
aggregate/
├── training_summary.csv
├── per_model_budget_results.csv
├── aggregate_summary.csv
├── representation_paired_differences.csv
├── representation_comparison_by_budget.csv
├── budget_gain_by_seed.csv
├── budget_comparison.csv
├── budget_representation_interaction.csv
└── figures/
```

### `budget_comparison.csv`

For each representation/opponent:

```text
mean score at 10k
mean score at 50k
mean 50k - 10k gain
SD of seed-level gains
```

### `budget_representation_interaction.csv`

The key descriptive quantity is:

```text
(UBR 1.2 gain from 10k→50k)
-
(UBR 1.1 gain from 10k→50k)
```

Positive values suggest that the richer representation benefited more from
extra training. Negative values suggest the reverse.

This remains descriptive with five seeds; do not treat it as a definitive
causal or equivalence test.

## Resume

The runner is resumable. Re-run with:

```powershell
python run_budget_study.py --study-dir "budget_results\budget_gen9_..."
```

using the same configuration.

Completed models and evaluations are reused; incomplete directories are
recreated.
