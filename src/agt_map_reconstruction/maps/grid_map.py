import numpy as np


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


def traversability_from_height(height_grid, max_height_difference=0.15):
    valid = ~np.isnan(height_grid)
    result = np.zeros_like(height_grid, dtype=np.uint8)
    result[valid] = 1
    return result
