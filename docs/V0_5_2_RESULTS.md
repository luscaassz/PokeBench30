# PokéBench30 v0.5.2 — observed 10k multi-seed baseline

Experimental setup:

- Gen IX
- MaskablePPO
- 10,000 requested timesteps
- training opponent: TypeAwareAgent
- seeds: 42, 123, 456, 789, 2026
- representations: UBR 1.1 and UBR 1.2
- evaluation: three fixed pairs, mirrored AB/BA, 10 repeats per leg
- 60 battles/model/opponent

Observed aggregate mirrored scores:

| Representation | RandomAgent | TypeAwareAgent |
|---|---:|---:|
| UBR 1.1 | 0.7033 ± 0.1114 | 0.0600 ± 0.0723 |
| UBR 1.2 | 0.7033 ± 0.0828 | 0.0600 ± 0.0480 |

Aggregate records:

- UBR 1.1 vs Random: 211-89 over 300 battles
- UBR 1.2 vs Random: 211-89 over 300 battles
- UBR 1.1 vs TypeAware: 18-282 over 300 battles
- UBR 1.2 vs TypeAware: 18-282 over 300 battles

Interpretation used to motivate v0.5.3:

> At a 10k training budget, no systematic mean performance advantage was
> observed for UBR 1.2 over UBR 1.1. The next question is whether the richer
> representation benefits from a larger training budget.

These are descriptive results from five training seeds. They are not an
equivalence claim.
