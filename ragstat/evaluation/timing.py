"""Timing summaries in milliseconds, using linear-interpolated percentiles."""

import math
from collections.abc import Sequence


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values or not 0 <= fraction <= 1:
        raise ValueError("A nonempty sample and fraction between zero and one are required")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
