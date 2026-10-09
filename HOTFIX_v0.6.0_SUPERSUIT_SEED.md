# PokéBench30 v0.6.0 — SuperSuit/SB3 seed compatibility hotfix

Observed on Windows during:

```powershell
python run_crossgen_benchmark.py --smoke
```

Error:

```text
AttributeError: 'ConcatVecEnv' object has no attribute 'seed'
```

## Cause

The Stable-Baselines3-compatible SuperSuit wrapper delegates `VecEnv.seed()`
to its underlying `ConcatVecEnv`.

The Gymnasium-style `ConcatVecEnv` exposes `reset(seed=...)`, but does not
expose the legacy `seed()` method expected by the SB3 wrapper during PPO model
construction.

## Fix

`crossgen_study.py` now installs a small compatibility shim after the SuperSuit
vector environment is created.

The shim:

1. implements the missing `seed()` method;
2. stores the requested seed;
3. forwards that seed to the next `reset(seed=...)`;
4. avoids starting and discarding an extra Pokémon battle during model setup.

This is an infrastructure compatibility fix only. It does not alter:

- PPO hyperparameters;
- UBR representation;
- reward;
- self-play protocol;
- training budget;
- evaluation protocol.
