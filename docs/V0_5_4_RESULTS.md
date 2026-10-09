# PokéBench30 v0.5.4 — observed curriculum results

The completed v0.5.4 experiment used:

- Generation IX
- UBR 1.2
- MaskablePPO
- 300k nominal training timesteps
- five seeds: 42, 123, 456, 789, 2026
- final fixed/mirrored evaluation: 60 battles per seed/opponent

Final mean mirrored scores:

| Training regime | Random | MaxDamage | TypeAware |
|---|---:|---:|---:|
| Direct TypeAware | 0.7900 | 0.3700 | 0.1100 |
| Random -> TypeAware | 0.8267 | 0.2867 | 0.1367 |
| Progressive | 0.7967 | 0.3533 | 0.0567 |
| Random only | 0.9000 | 0.4933 | 0.0767 |

The Random -> TypeAware curriculum produced the highest mean score against
TypeAware, although the improvement over direct training was small and variable
across seeds. The progressive Random -> MaxDamage -> TypeAware schedule was
worse than direct training against TypeAware.

The v0.5.5 architecture study therefore fixes the curriculum to:

```text
0–100k:   Random
100–300k: TypeAware
```

and changes only the policy architecture.
