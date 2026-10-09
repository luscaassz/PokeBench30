from __future__ import annotations

import math


def wilson_interval(
    successes: int,
    trials: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Returns (0, 0) when no trials are available.
    """
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
