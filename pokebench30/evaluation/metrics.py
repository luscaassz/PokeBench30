from __future__ import annotations

import math


def wilson_interval(
    successes: int,
    trials: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if trials <= 0:
        return 0.0, 0.0

    p_hat = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = (p_hat + z2 / (2.0 * trials)) / denominator
    margin = (
        z
        * math.sqrt(
            (p_hat * (1.0 - p_hat) / trials)
            + z2 / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def exact_binomial_two_sided(successes: int, trials: int, p: float = 0.5) -> float:
    """Small pure-Python two-sided exact binomial p-value.

    It sums outcomes whose probability is no larger than that of the observed
    outcome. This is appropriate for the benchmark's simple win/loss checks.
    """
    if trials <= 0:
        return 1.0
    if not 0 <= successes <= trials:
        raise ValueError("successes must be between 0 and trials")
    if not 0 < p < 1:
        raise ValueError("p must be between 0 and 1")

    def prob(k: int) -> float:
        return math.comb(trials, k) * (p**k) * ((1 - p) ** (trials - k))

    observed = prob(successes)
    total = sum(prob(k) for k in range(trials + 1) if prob(k) <= observed + 1e-15)
    return min(1.0, total)
