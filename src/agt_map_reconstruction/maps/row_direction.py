"""Estimate agricultural row direction from geometric map structure.

This module provides a geometry-only baseline for agricultural scenes.
The goal is not semantic recognition, but recovering dominant field/crop
orientation before corridor extraction.
"""

import numpy as np


def estimate_row_direction(points_or_grid):
    """Estimate dominant 2D direction using PCA.

    Args:
        points_or_grid:
            Either Nx2 points or a 2D grid. For grids, valid cells are
            converted into image coordinates.

    Returns:
        angle_rad: dominant direction angle in radians.
        direction: unit vector [dx, dy].
    """
    data = np.asarray(points_or_grid)

    if data.ndim == 2 and data.shape[1] == 2:
        points = data
    elif data.ndim == 2:
        valid = ~np.isnan(data)
        y, x = np.where(valid)
        points = np.column_stack((x, y))
    else:
        raise ValueError("input must be Nx2 points or 2D grid")

    if len(points) < 2:
        return 0.0, np.array([1.0, 0.0])

    centered = points - points.mean(axis=0)
    covariance = centered.T @ centered / len(points)

    values, vectors = np.linalg.eigh(covariance)
    direction = vectors[:, np.argmax(values)]

    angle = float(np.arctan2(direction[1], direction[0]))
    return angle, direction
