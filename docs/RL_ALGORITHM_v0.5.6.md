# PokéBench30 v0.5.6 — RL Algorithm Study

## Research question

The previous ablations indicate that simply increasing the feed-forward network
size or adding a short four-decision history does not improve performance
against TypeAware.

v0.5.6 therefore asks:

> With representation, reward, policy architecture, curriculum, seeds,
> environment-interaction budget and evaluation held fixed, how does the RL
> algorithm affect performance?

## Fixed conditions

```text
Generation        IX
Representation    UBR 1.2
Observation dim   518
Action dim        10
Policy network    MLP [256, 128]
Reward            unchanged v0.5 dense reward
Seeds             42, 123, 456, 789, 2026

Curriculum
0–100k            Random
100–300k          TypeAware
```

The fixed/mirrored evaluation protocol is unchanged.

## Algorithms

### MaskablePPO

The current PokéBench30 baseline.

It learns an actor-critic policy directly and excludes illegal actions through
the action mask.

### Masked DQN

The network outputs one Q-value for each of the 10 universal actions:

```text
UBR 518
  ↓
MLP [256, 128]
  ↓
Q(s,a0) ... Q(s,a9)
```

Illegal actions are excluded in both places where they matter:

1. epsilon-greedy behavior/action selection;
2. the next-state bootstrap maximum.

Therefore an illegal action cannot become the target simply because it has a
large Q-value.

### Masked Double DQN

Identical to Masked DQN except for the bootstrap rule:

```text
online network:
    selects the best legal next action

target network:
    evaluates that selected action
```

This isolates the DQN-vs-Double-DQN target rule while keeping the replay,
epsilon schedule, network and all other Q-learning settings identical.

## Shared DQN / DDQN configuration

```text
network                 [256, 128]
learning rate           2.5e-4
gamma                   0.99
batch size              128
replay capacity         50,000 transitions
learning starts         5,000
train frequency         every 4 environment steps
gradient steps          1
target update interval  2,000 environment steps
epsilon                 1.00 -> 0.05
epsilon decay window    first 50% of training
loss                    Huber
max grad norm           10
```

These values are a prespecified engineering baseline, not a claim that DQN or
DDQN has been optimally tuned.

## Environment-interaction budget

The earlier PPO studies label checkpoints as 50k, 100k, ..., 300k, but
MaskablePPO collects complete rollouts of 2048 steps.

Thus each nominal 50k block actually contains:

```text
ceil(50,000 / 2,048) × 2,048
= 51,200 environment steps
```

and the final model has:

```text
6 × 51,200
= 307,200 environment steps
```

To avoid giving DQN/DDQN a smaller experience budget, v0.5.6 gives them exactly
the same **307,200 environment interactions**, while retaining the familiar
nominal checkpoint labels.

## Replay across the curriculum transition

DQN/DDQN keep their replay buffer when the training opponent changes from
Random to TypeAware.

This is intentional: replay is part of the off-policy algorithm. Random-stage
experience therefore remains available temporarily during TypeAware
fine-tuning and is naturally replaced as the finite replay buffer fills.

## Crash-safe resume

For DQN/DDQN, the latest checkpoint includes a separate resume snapshot with:

- online network;
- target network;
- optimizer;
- RNG state;
- exact environment-step count;
- replay buffer.

Only one large replay snapshot is kept per active seed. After the final 300k
checkpoint completes, it is deleted because the final model itself is
sufficient for evaluation.

The runner refuses to resume a partially completed DQN/DDQN trajectory if the
matching replay snapshot is missing. It will not silently continue with an
empty replay buffer.

## Reusing the PPO baseline

The completed v0.5.5 policy study already contains the required
`mlp_baseline`, trained with the exact Random -> TypeAware curriculum.

Example:

```powershell
python run_algorithm_study.py `
  --ppo-baseline-dir "C:\...\PokeBench30_v0.5.5\policy_results\policy_gen9_20261002_194318"
```

With this import only 10 new trajectories are needed:

```text
2 new algorithms
× 5 seeds
× 307,200 effective environment steps
=
3,072,000 new environment interactions
```

## Main outputs

```text
algorithm_results/
└── algorithm_gen9_.../
    ├── study_manifest.json
    ├── imported_maskable_ppo/
    ├── runs/
    │   ├── masked_dqn/
    │   └── masked_double_dqn/
    └── aggregate/
        ├── learning_curve.csv
        ├── aggregate_learning_curve.csv
        ├── training_progress.csv
        ├── final_summary.csv
        ├── paired_algorithm_differences.csv
        ├── algorithm_comparison.csv
        └── figures/
```

`algorithm_comparison.csv` is the primary final comparison against
MaskablePPO.

## Interpretation

This is an algorithm-level comparison, not a claim that every hyperparameter is
identical across algorithms. PPO and DQN necessarily have different
optimization machinery.

The controlled elements are the representation, policy-network shape, reward,
training-opponent sequence, seeds, effective environment-interaction budget,
action legality and final evaluation protocol.
