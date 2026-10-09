# PokéBench30 v0.5.5 — observed policy-architecture results

The completed v0.5.5 experiment used:

- Generation IX
- UBR 1.2
- Random 100k -> TypeAware 200k curriculum
- 300k nominal training timesteps
- five seeds: 42, 123, 456, 789, 2026
- final fixed/mirrored evaluation: 60 battles per seed/opponent

Final mean mirrored scores:

| Policy | Random | MaxDamage | TypeAware |
|---|---:|---:|---:|
| MLP baseline `[256,128]` | 0.8267 | 0.2867 | **0.1367** |
| MLP large `[512,256,128]` | 0.7733 | 0.2933 | 0.0600 |
| History MLP | **0.8467** | **0.3033** | 0.0367 |
| History LSTM | 0.6600 | 0.2367 | 0.0700 |

Against TypeAware, every alternative architecture finished below the baseline
mean.

Paired mean differences versus the baseline:

```text
history_lstm   -0.0667
history_mlp    -0.1000
mlp_large      -0.0767
```

These results do not prove that recurrent or larger policies are generally
unsuitable for Pokémon. They show that the specific tested changes — a larger
feed-forward network and a fixed four-decision history encoded by either an MLP
or LSTM — did not improve the current benchmark setting.

This motivates v0.5.6: keep the best-performing policy architecture fixed and
change the reinforcement-learning algorithm instead.
