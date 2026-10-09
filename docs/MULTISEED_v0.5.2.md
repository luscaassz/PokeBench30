# v0.5.2 — Multi-seed UBR ablation

## Research question

Does adding the UBR 1.2 battle-context block improve the learned MaskablePPO
policy consistently, or were the single-seed v0.5/v0.5.1 results mostly run
variance?

The experiment changes only the representation profile:

| Component | UBR 1.1 arm | UBR 1.2 arm |
|---|---:|---:|
| observation | 472 | 518 |
| actions | 10 | 10 |
| algorithm | MaskablePPO | MaskablePPO |
| policy network | same | same |
| reward | same | same |
| training opponent | same | same |
| timesteps | same | same |
| evaluation teams | same | same |

The 46 additional UBR 1.2 values cover selected hazards/side conditions,
weather, field effects, and volatile effects.

## Default seeds

```text
42, 123, 456, 789, 2026
```

Five seeds are a first stability check, not a large-sample statistical study.
The output emphasizes seed-wise scores, mean, standard deviation, and paired
representation differences rather than treating every battle as an independent
model replicate.

## Default workload

For two UBRs and five seeds:

```text
10 trained models
2 evaluation opponents/model
60 controlled battles/evaluation
= 1,200 controlled evaluation battles
```

Training contributes an additional 100,000 requested PPO timesteps in total.

## Output tree

```text
multiseed_results/
└── multiseed_gen9_YYYYMMDD_HHMMSS/
    ├── experiment_manifest.json
    ├── models/
    │   ├── ubr_1_1_seed_42/
    │   ├── ...
    │   └── ubr_1_2_seed_2026/
    ├── evaluations/
    │   ├── ubr_1_1_seed_42_vs_random/
    │   ├── ubr_1_1_seed_42_vs_type_aware/
    │   └── ...
    └── aggregate/
        ├── training_summary.csv
        ├── per_model_results.csv
        ├── aggregate_summary.csv
        ├── paired_differences.csv
        ├── representation_comparison.csv
        └── figures/
```

## Resume support

The runner writes partial aggregate files after every completed model. If an
experiment is interrupted, rerun the same command with:

```powershell
python run_multiseed.py --experiment-dir "multiseed_results\multiseed_gen9_..."
```

Completed models/evaluations are checked and reused. The runner refuses to mix
a different configuration into the same experiment directory.

## RNG interpretation

`--seed` controls the learning stack used by Stable-Baselines3/PyTorch and is
stored with each model. It does not make the Pokémon Showdown battle-event RNG
identical between UBR arms. Consequently, `paired_differences.csv` is useful
for matched seed bookkeeping but is not a strict common-random-numbers design.
