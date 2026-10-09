# PokéBench30 v0.5.4

## Curriculum Training Study

v0.5.4 keeps the selected RL representation, action space, reward and MaskablePPO architecture fixed and adds an automated training-opponent curriculum experiment.

### New

- `run_curriculum_study.py`
- `pokebench30/rl/curriculum_study.py`
- `scripts/check_curriculum_setup.py`
- `docs/CURRICULUM_v0.5.4.md`
- `docs/LITERATURE_RATIONALE_v0.5.4.md`
- four 300k curriculum regimes
- five-seed experiment automation
- 50k checkpoint saving
- diagnostic checkpoint evaluation vs Random, MaxDamage and TypeAware
- full 300k fixed/mirrored final evaluation
- safe resume from completed checkpoints
- seed-level paired comparisons against direct TypeAware training
- curriculum learning-curve figures
- `--smoke` end-to-end mode

### Default regimes

```text
direct_type_aware:
  300k TypeAware

random_to_type_aware:
  100k Random + 200k TypeAware

progressive:
  100k Random + 100k MaxDamage + 100k TypeAware

random_only:
  300k Random
```

### Scientific isolation

The following remain unchanged:

- UBR 1.2 (518 observations)
- 10-action universal action space
- dense reward from v0.5
- PPO network and hyperparameters
- constant learning rate
- fixed/mirrored final evaluation protocol

The version therefore isolates the effect of **training-opponent curriculum** as much as possible within the current framework.
