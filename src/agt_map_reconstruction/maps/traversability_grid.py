"""Generate simple 2D traversability grids from segmented point clouds."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def points_to_grid(points, resolution=0.05):
    if len(points) == 0:
        return np.zeros((1, 1), dtype=np.float32)

    xy = points[:, :2]
    min_xy = xy.min(axis=0)
    indices = np.floor((xy - min_xy) / resolution).astype(np.int32)

    size = indices.max(axis=0) + 1
    grid = np.full((size[1], size[0]), np.nan, dtype=np.float32)

    z = points[:, 2]
    for idx, height in zip(indices, z):
        x, y = idx
        if np.isnan(grid[y, x]) or height > grid[y, x]:
            grid[y, x] = height

    return grid


def save_grid(grid, path, title="traversability"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.imshow(grid, origin="lower")
    plt.colorbar(label="height")
    plt.title(title)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
