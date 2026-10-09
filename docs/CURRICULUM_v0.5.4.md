# PokéBench30 v0.5.4 — Curriculum Training Study

## Research question

The previous RL milestones established that:

- PPO learns enough to outperform a Random baseline;
- increasing the budget from 10k to 50k does not reliably improve fixed/mirrored evaluation;
- UBR 1.2 is conceptually richer than UBR 1.1 but does not yield a large systematic gain at these small budgets.

v0.5.4 therefore changes **only the training-opponent regime** while holding the representation, reward and PPO configuration fixed.

The question is:

> For the same 300k nominal training budget, does a curriculum of progressively stronger opponents improve the final policy compared with training directly against TypeAwareAgent?

## Literature motivation

This design is informed by prior Pokémon RL work:

- Jett Wang, *Winning at Pokémon Random Battles Using Reinforcement Learning* (MIT, 2024): PPO with self-play, regular validation against a heuristic opponent, action masking, and very large-scale training.
- Rossi et al., *I Choose You, Reinforcement Learning! Trained RL Agents For Pokémon Battles* (SBGames 2024): DRL agents trained at 300k and 900k steps against Random/MaxDamage-style opponents.
- Hamish Ivison, *PokéRL* (2021): experiments suggesting that training against a simpler/random opponent can sometimes generalize better to a stronger deterministic opponent than training only against that stronger opponent.
- Keval Vivek, *Pokémon Showdown with Reinforcement Learning*: poke-env examples using a dense HP/faint/victory reward similar in spirit to the current PokéBench reward.

The v0.5.4 experiment does **not** claim these works used identical environments or protocols. They motivate a controlled curriculum ablation inside PokéBench30.

## Constants held fixed

```text
Generation:        IX
Representation:    UBR 1.2 (518 features)
Action space:      10
Algorithm:         MaskablePPO
Reward:            unchanged from v0.5
Learning rate:     3e-4 constant
PPO architecture:  unchanged
Seeds:             42, 123, 456, 789, 2026
Nominal budget:    300k per regime/seed
```

The learning rate is intentionally kept constant in this version. Literature suggests that annealing may matter, but changing both curriculum and the LR schedule in the same experiment would confound the curriculum ablation. LR scheduling should be tested separately after this study.

## Training regimes

### A — `direct_type_aware`

```text
0 ─────────────────────────────── 300k
             TypeAware
```

### B — `random_to_type_aware`

```text
0 ─────── 100k ────────────────── 300k
   Random          TypeAware
```

### C — `progressive`

```text
0 ─────── 100k ─────── 200k ───── 300k
   Random       MaxDamage     TypeAware
```

### D — `random_only`

```text
0 ─────────────────────────────── 300k
               Random
```

The Random-only arm is a control for whether diversity from a weak stochastic opponent produces a more general policy than direct training against TypeAware.

## Checkpoint evaluation

Models are saved at nominal:

```text
50k
100k
150k
200k
250k
300k
```

PPO collects full rollouts, so with the default `n_steps=2048` each 50k segment normally executes 51,200 actual steps. The nominal checkpoint label is retained as the experimental condition, while `curriculum_metadata.json` records the actual `model.num_timesteps`.

Intermediate checkpoints are **diagnostic**, with one repeat per mirrored leg by default:

```text
3 team pairs × 2 mirror legs × 1 repeat = 6 battles/opponent
```

Each checkpoint is tested against:

```text
Random
MaxDamage
TypeAware
```

The 300k final checkpoint uses 10 repeats per leg:

```text
3 pairs × 2 legs × 10 = 60 battles/opponent
```

Only the final 300k evaluation should be used for the primary cross-regime comparison. Intermediate curves are intended to show learning dynamics and transition effects.

## Primary comparison

The primary reference is `direct_type_aware`.

For every seed and evaluation opponent, v0.5.4 computes:

```text
curriculum final mirror score
-
direct TypeAware final mirror score
```

Outputs:

```text
aggregate/
├── learning_curve.csv
├── aggregate_learning_curve.csv
├── training_progress.csv
├── final_summary.csv
├── paired_regime_differences.csv
├── curriculum_comparison.csv
└── figures/
```

`curriculum_comparison.csv` is the main table for deciding whether a curriculum produced a consistent improvement over direct TypeAware training.

## Resume behavior

The runner is resumable. A checkpoint is considered complete when it contains both:

```text
maskable_ppo.zip
metadata.json
```

Missing evaluations are recreated without retraining that checkpoint. If training stopped after a completed checkpoint, the PPO model is loaded with its optimizer state and training continues from that checkpoint.

Because Pokémon Showdown's battle RNG is not common-random-number controlled, resumed trajectories are not guaranteed to be bitwise identical to an uninterrupted run.

## What v0.5.4 does not test

This version does not change:

- reward coefficients;
- UBR features;
- recurrent memory;
- learning-rate schedule;
- MCTS/planning;
- self-play.

Those remain separate future ablations if curriculum alone is insufficient.
