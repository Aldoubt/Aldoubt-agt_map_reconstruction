"""Generate simple agricultural traversability maps."""

import numpy as np


def compute_traversability(relative_height, max_step=0.15):
    """Classify cells by relative height.

    0 unknown
    1 traversable
    2 obstacle
    """
    result = np.zeros_like(relative_height, dtype=np.uint8)

    valid = ~np.isnan(relative_height)
    result[valid] = 1
    result[valid & (relative_height > max_step)] = 2

    return result
