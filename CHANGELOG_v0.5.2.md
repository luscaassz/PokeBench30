# PokéBench30 v0.5.2

## Goal

v0.5.2 turns the v0.5 / v0.5.1 comparison into a reproducible multi-seed
ablation instead of relying on one PPO run per representation.

## Main changes

- One codebase can now train and evaluate both validated representations:
  - `ubr-1.1`: 472 observations, 10 actions
  - `ubr-1.2`: 518 observations, 10 actions
- The validated schema hashes are preserved exactly.
- `train_rl.py` accepts `--ubr ubr-1.1|ubr-1.2`.
- Saved-model inference automatically reconstructs the representation recorded
  in `metadata.json`.
- `run_multiseed.py` automates the complete representation ablation:
  - multiple seeds;
  - both UBR profiles;
  - training against the same scripted opponent;
  - fixed-team mirrored evaluation against selected baselines;
  - resume support;
  - per-model, aggregate, and paired-difference CSVs;
  - aggregate comparison plots.
- `run_rl_controlled.py` now accepts deterministic `--output-root` and
  `--run-name` paths for experiment orchestration.
- `scripts/check_rl_setup.py` validates both schema hashes and confirms that
  UBR 1.2 preserves the full UBR 1.1 prefix.
- Added `scripts/check_multiseed_setup.py`.

## Default experiment

```text
Generation: Gen IX
Algorithm: MaskablePPO
Representations: UBR 1.1, UBR 1.2
Seeds: 42, 123, 456, 789, 2026
Training opponent: TypeAwareAgent
Requested timesteps/model: 10,000
Evaluation opponents: RandomAgent, TypeAwareAgent
Evaluation: 3 team pairs × 2 mirror legs × 10 repeats = 60 battles/opponent/model
Models: 10
Controlled evaluations: 20
Total controlled evaluation battles: 1,200
```

## Important statistical limitation

The PPO seed is controlled, but PokéBench30 does not currently impose common
random numbers on Pokémon Showdown battle-event RNG. Therefore the same seed
across UBR 1.1 and UBR 1.2 should be interpreted as repeated model runs with a
matched algorithm seed, not as identical stochastic battle trajectories.
