# PokéBench30 v0.5.3

## Training Budget Study

v0.5.3 does not change the UBR feature sets, action space, PPO network or
reward. It adds an automated training-budget experiment.

### New

- `run_budget_study.py`
- `pokebench30/rl/budget_study.py`
- `scripts/check_budget_study_setup.py`
- `docs/TRAINING_BUDGET_v0.5.3.md`
- `docs/V0_5_2_RESULTS.md`
- default 10k vs 50k study
- five-seed paired descriptive comparisons
- import of compatible completed v0.5.2 10k experiments
- safe resume of 50k model/evaluation runs
- budget-level aggregate CSVs
- representation × budget descriptive interaction output
- budget-performance figures

### Experimental design

Default:

```text
UBR 1.1 vs UBR 1.2
10k vs 50k
5 seeds
Gen IX
MaskablePPO
train vs TypeAware
evaluate vs Random + TypeAware
60 battles/model/opponent
```

When a compatible v0.5.2 baseline is imported, v0.5.3 trains only the 50k
condition.

### Important

50k models are trained from scratch. They are not continued from 10k
checkpoints.

Pokémon Showdown battle-event RNG is still not common-random-number controlled,
so paired seed comparisons are repeated stochastic training runs rather than
identical battle trajectories.
