"""Seeded paired comparisons over a fixed question set.

The bootstrap resamples questions with replacement and reports a percentile
interval for the mean paired delta. That interval describes uncertainty from
this query sample. It does not model retrieval randomness, corpus sampling, or
hardware noise. The sign-flip test is a two-sided paired permutation test of
the null that the mean delta is zero: under that null each question's sign is
exchangeable. Both procedures use ``random.Random``, so identical seeds,
sample counts, and CPython versions reproduce the same numbers.
"""

import random
from collections.abc import Sequence
from statistics import fmean

from ragstat.evaluation.timing import percentile


def bootstrap_mean_ci(
    values: Sequence[float], *, samples: int, seed: str, confidence: float
) -> tuple[float, float, float]:
    """Return the sample mean and the percentile confidence interval of the mean."""
    if not values:
        raise ValueError("Bootstrap requires at least one observation")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
        raise ValueError("Bootstrap sample count must be a positive integer")
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be strictly between zero and one")
    rng = random.Random(seed)
    size = len(values)
    means = [sum(values[rng.randrange(size)] for _ in range(size)) / size for _ in range(samples)]
    alpha = (1.0 - confidence) / 2.0
    return fmean(values), percentile(means, alpha), percentile(means, 1.0 - alpha)


def signflip_p_value(deltas: Sequence[float], *, samples: int, seed: str) -> float:
    """Two-sided paired sign-flip p-value, with the observed statistic included.

    The +1 numerator and denominator follow the permutation convention that the
    observed sample is one of the possible assignments, so the p-value cannot be zero.
    """
    if not deltas:
        raise ValueError("Permutation test requires at least one observation")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
        raise ValueError("Permutation sample count must be a positive integer")
    observed = abs(fmean(deltas))
    rng = random.Random(seed)
    size = len(deltas)
    extreme = 0
    for _ in range(samples):
        total = sum(value if rng.randrange(2) else -value for value in deltas)
        if abs(total / size) >= observed:
            extreme += 1
    return (extreme + 1) / (samples + 1)
