# PokéBench30 v0.5.5

## Policy Architecture Study

v0.5.5 freezes the best curriculum candidate from v0.5.4 and varies policy
architecture.

### Fixed

- Generation IX
- UBR 1.2
- 10-action universal action space
- MaskablePPO
- dense v0.5 reward
- constant learning rate
- 300k nominal training budget
- five seeds
- Random 100k -> TypeAware 200k curriculum
- fixed/mirrored evaluation

### New architectures

- `mlp_baseline`: `[256, 128]`
- `mlp_large`: `[512, 256, 128]`
- `history_mlp`: four UBR states -> flattened MLP
- `history_lstm`: four UBR states -> LSTM(256) -> MLP(128)

### Why no fully stateful RecurrentPPO?

Strict invalid-action masking is a benchmark invariant. Native sb3-contrib
MaskablePPO does not support recurrent policies in the current stack.

v0.5.5 therefore uses a fixed-window LSTM encoder while keeping MaskablePPO
and identical action masking across all conditions.

### New

- `run_policy_study.py`
- `pokebench30/rl/architecture_study.py`
- `pokebench30/rl/policy_specs.py`
- `pokebench30/rl/temporal_policy.py`
- architecture-aware `RLPolicyAgent`
- backward-compatible architecture metadata
- optional import of the v0.5.4 `random_to_type_aware` baseline
- policy learning curves and paired seed comparisons
- non-interactive Matplotlib `Agg` backend

### Main scientific comparison

```text
mlp_baseline
vs
mlp_large
vs
history_mlp
vs
history_lstm
```

with all non-architecture experimental factors held fixed.
