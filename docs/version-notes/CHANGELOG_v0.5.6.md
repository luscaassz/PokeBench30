# PokéBench30 v0.5.6

## RL Algorithm Study

v0.5.6 compares:

- MaskablePPO
- Masked DQN
- Masked Double DQN

while fixing:

- Generation IX
- UBR 1.2
- MLP `[256, 128]`
- dense v0.5 reward
- Random 100k -> TypeAware 200k curriculum
- five seeds
- effective environment-interaction budget
- fixed/mirrored evaluation

## New

- `run_algorithm_study.py`
- `pokebench30/rl/algorithm_study.py`
- `pokebench30/rl/masked_dqn.py`
- strict action masking for DQN behavior and bootstrap targets
- Double DQN masked target selection/evaluation
- replay-buffer resume snapshots
- optional import of the v0.5.5/v0.5.4 PPO baseline
- paired seed-level algorithm comparisons
- `algorithm_comparison.csv`

## Incorporated v0.5.5 infrastructure hotfixes

### Legal-action mapping

Ordinary legal moves exposed by `battle.valid_orders` now supplement
`battle.available_moves`.

This fixes the rare v0.5.5 failure where a valid order such as Earthquake was
accepted by Showdown but could not be represented by the UBR action mapper.

### Resume manifest canonicalization

Tuple/list differences introduced by JSON serialization are normalized before
study manifests are compared.

This fixes false "different configuration" errors when resuming a valid
v0.5.5-style experiment.

## Budget normalization

A nominal 50k PPO segment actually collects 51,200 environment interactions
with `n_steps=2048`.

v0.5.6 gives DQN and DDQN the same effective 51,200 interactions per nominal
checkpoint block, for 307,200 total environment interactions.
