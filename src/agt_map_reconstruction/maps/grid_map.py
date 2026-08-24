import numpy as np

from .elevation_normalization import normalize_height
from .traversability import compute_traversability


def points_to_height_grid(points, resolution=0.05):
    points = np.asarray(points)
    x_min, y_min = points[:, :2].min(axis=0)
    idx = np.floor((points[:, :2] - [x_min, y_min]) / resolution).astype(int)
    size = idx.max(axis=0) + 1
    grid = np.full((size[1], size[0]), np.nan)
    for p, cell in zip(points[:, 2], idx):
        x, y = cell
        if np.isnan(grid[y, x]):
            grid[y, x] = p
        else:
            grid[y, x] = min(grid[y, x], p)
    return grid


def build_traversability_map(points, resolution=0.05, kernel_size=5):
    height_grid = points_to_height_grid(points, resolution)
    relative_height = normalize_height(height_grid, kernel_size)
    traversability = compute_traversability(relative_height)
    return {
        "height": height_grid,
        "relative_height": relative_height,
        "traversability": traversability,
    }
