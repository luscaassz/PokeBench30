# PokéBench30 v0.5.5 — Policy Architecture Study

## Research question

After the representation, training-budget and curriculum ablations, the main
remaining hypothesis is that the original feed-forward policy is too limited
for strategic Pokémon decisions.

v0.5.5 asks:

> With UBR, reward, PPO hyperparameters, training budget and curriculum held
> fixed, does changing the policy architecture improve generalization against
> Random, MaxDamage and TypeAware agents?

## Fixed experimental conditions

```text
Generation:       IX
Representation:   UBR 1.2
UBR dimension:    518
Action space:     10
Algorithm:        MaskablePPO
Reward:           unchanged v0.5 reward
Learning rate:    3e-4 constant
Training budget:  300k nominal timesteps
Seeds:            42, 123, 456, 789, 2026

Curriculum:
0–100k       Random
100–300k     TypeAware
```

Evaluation remains the same fixed/mirrored protocol used in v0.5.4.

## Architectures

### A — `mlp_baseline`

```text
518
 ↓
MLP [256, 128]
 ↓
actor / critic
```

This is the original v0.5-v0.5.4 policy and the comparison baseline.

### B — `mlp_large`

```text
518
 ↓
MLP [512, 256, 128]
 ↓
actor / critic
```

This tests whether the original policy was simply under-capacity.

### C — `history_mlp`

```text
4 recent UBR states
4 × 518 = 2072
 ↓
flatten
 ↓
MLP [512, 256, 128]
 ↓
actor / critic
```

This tests whether short temporal context helps even without an explicit
sequence model.

### D — `history_lstm`

```text
4 recent UBR states
      ↓
LSTM hidden=256
      ↓
MLP [128]
      ↓
actor / critic
```

This tests a temporal inductive bias using exactly the same four-state history
window as `history_mlp`.

## Why fixed-window LSTM instead of RecurrentPPO?

PokéBench requires strict invalid-action masking. The current sb3-contrib
MaskablePPO implementation does not natively support recurrent policies.

Using ordinary RecurrentPPO would force a different invalid-action treatment
and would confound the architecture ablation.

Therefore v0.5.5 uses a fixed-window LSTM feature extractor inside
MaskablePPO. This preserves:

- the same 10-action space;
- the same legal-action masks;
- the same PPO implementation;
- the same reward;
- the same evaluation protocol.

It tests short-term temporal sequence modeling without changing action
semantics.

A future version can evaluate a fully stateful recurrent masked PPO if a stable
implementation becomes available or is separately validated.

## Why include both history MLP and history LSTM?

If only baseline MLP and LSTM were compared, an LSTM improvement could come
from either:

1. access to previous observations; or
2. the LSTM sequence-processing inductive bias.

`history_mlp` provides a control condition.

Examples of possible outcomes:

```text
baseline ≈ large < history_mlp ≈ history_lstm
```

suggests that temporal information itself matters.

```text
baseline ≈ large ≈ history_mlp < history_lstm
```

suggests a stronger role for sequence modeling.

```text
baseline < large ≈ history models
```

suggests that raw model capacity was the main limitation.

```text
all similar
```

moves suspicion back toward UBR omissions, PPO optimization, opponent
distribution, or planning.

## Reusing the v0.5.4 MLP baseline

The completed `random_to_type_aware` regime from v0.5.4 is exactly the
`mlp_baseline` condition required here.

It can be imported instead of re-running five 300k trajectories.

Example:

```powershell
python run_policy_study.py --baseline-curriculum-dir "C:\...\PokeBench30_v0.5.4\curriculum_results\curriculum_gen9_YYYYMMDD_HHMMSS"
```

The importer validates the generation, representation, seeds, curriculum,
training budget, evaluation settings, PPO settings, fixed-team hash and
Pokémon Showdown commit.

With baseline import:

```text
3 new architectures
× 5 seeds
× 300k
=
15 new training trajectories
=
4.5M nominal new timesteps
```

Without import:

```text
4 architectures
× 5 seeds
× 300k
=
20 trajectories
=
6.0M nominal timesteps
```

## Checkpoints

The study evaluates:

```text
50k
100k
150k
200k
250k
300k
```

Intermediate checkpoints use the small diagnostic evaluation.
The 300k checkpoint uses the full fixed/mirrored evaluation.

## Main outputs

```text
policy_results/
└── policy_gen9_.../
    ├── study_manifest.json
    ├── imported_mlp_baseline/
    ├── runs/
    └── aggregate/
        ├── learning_curve.csv
        ├── aggregate_learning_curve.csv
        ├── training_progress.csv
        ├── final_summary.csv
        ├── paired_architecture_differences.csv
        ├── architecture_comparison.csv
        └── figures/
```

The primary result file is:

```text
architecture_comparison.csv
```

The primary comparison against TypeAware is the paired seed difference between
each architecture and `mlp_baseline`.
