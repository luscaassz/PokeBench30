# PokéBench30 v0.6.0

## Cross-Generation Benchmark

v0.6.0 changes the central project question.

The v0.5.x line investigated engineering choices around representation,
training budget, curriculum, policy architecture and RL algorithm.

v0.6.0 freezes a literature-grounded agent and makes **generation** the primary
experimental variable.

## Frozen agent

- PPO with custom invalid-action masking
- UBR 1.2
- shared-policy two-player PettingZoo self-play
- sparse terminal reward `{-1,0,+1}`
- Wang learning-rate annealing
- Wang PPO optimization coefficients
- Wang-inspired shared 3×256 trunk + 2×256 actor/critic heads
- common-core 10-action space

## Main benchmark

```text
Gen I × Gen IV × Gen IX
5 seeds/generation
900k nominal steps/model
```

## Evaluation improvements

- fixed pool expanded from 3 to 10 team pairs per generation;
- final mirrored evaluation remains 60 battles/model/opponent:
  `10 pairs × 2 legs × 3 repeats`;
- pair-level outputs are preserved for statistical analysis;
- diagnostic TypeAware learning curves are recorded every 150k nominal steps.

## New files

- `run_crossgen_benchmark.py`
- `pokebench30/rl/crossgen_study.py`
- `pokebench30/rl/wang_protocol.py`
- `pokebench30/rl/wang_style.py`
- `config/crossgen_benchmark.yaml`
- `scripts/check_crossgen_setup.py`
- `docs/LITERATURE_PROTOCOL_v0.6.md`
- `docs/CROSS_GENERATION_DESIGN_v0.6.md`

## Important scope decisions

- v0.6.0 is Wang-inspired, not an exact reproduction.
- MCTS is not used.
- both player trajectories are collected through the two-agent PettingZoo VecEnv.
- Wang's 3725-dimensional Gen-IV representation is replaced by UBR 1.2.
- Gen IX Terastallization is excluded from the common-core action space.
- primary models are trained independently within each generation; cross-gen
  transfer is reserved for a later experiment.

## Infrastructure inherited

The v0.5.5 legal-order hotfix remains included:

```text
valid_orders supplements available_moves
```

The JSON manifest canonicalization fix also remains included for reliable
resume.
