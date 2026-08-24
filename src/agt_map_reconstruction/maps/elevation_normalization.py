"""Local elevation normalization for agricultural LiDAR maps.

Convert absolute height into relative height by estimating a local ground
surface from a height grid.
"""

import numpy as np


def estimate_ground_surface(height_grid, kernel_size=5):
    """Estimate local ground surface using neighborhood minimum filtering."""
    from scipy.ndimage import minimum_filter

    valid = np.nan_to_num(height_grid, nan=np.nanmax(height_grid))
    return minimum_filter(valid, size=kernel_size)


def normalize_height(height_grid, kernel_size=5):
    """Return height above local ground."""
    ground = estimate_ground_surface(height_grid, kernel_size)
    relative = height_grid - ground
    relative[np.isnan(height_grid)] = np.nan
    return relative
