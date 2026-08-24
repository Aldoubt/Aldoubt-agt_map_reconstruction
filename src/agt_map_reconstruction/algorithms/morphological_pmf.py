"""Progressive morphological ground filter on a tiled elevation raster.

This is a 2.5D PMF implementation for a globally registered PCD.  It follows
the standard PMF idea: progressively larger morphological openings estimate a
low ground surface while a slope-dependent threshold decides ground points.
Tiles include a halo so morphology does not stop at arbitrary tile borders.
"""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from agt_map_reconstruction.maps.grid_map import points_to_height_grid


@dataclass(frozen=True)
class PMFConfig:
    resolution: float = 0.05
    chunk_size: int = 1_000_000
    tile_size: int = 256
    max_window_m: float = 1.00
    initial_distance_m: float = 0.05
    slope: float = 0.20
    height_threshold_m: float = 0.15

    def __post_init__(self):
        if self.resolution <= 0 or self.chunk_size <= 0 or self.tile_size <= 0:
            raise ValueError("resolution, chunk_size, and tile_size must be positive")
        if self.max_window_m <= 0 or self.initial_distance_m < 0 or self.slope < 0:
            raise ValueError("PMF distances and slope must be non-negative")
        if self.height_threshold_m <= 0:
            raise ValueError("height_threshold_m must be positive")


def _fill_missing_nearest(grid):
    valid = np.isfinite(grid)
    if valid.all():
        return grid
    if not valid.any():
        raise ValueError("elevation grid contains no observed cells")
    distances, indices = ndimage.distance_transform_edt(~valid, return_distances=True, return_indices=True)
    del distances
    return grid[tuple(indices)]


def _window_sizes(config):
    maximum = int(np.floor(config.max_window_m / config.resolution))
    sizes = []
    for size in range(1, max(1, maximum) + 1, 2):
        sizes.append(size)
    return sizes or [1]


def _pmf_tile(tile, config):
    if not np.isfinite(tile).any():
        return np.full(tile.shape, np.nan, dtype=np.float32)
    filled = _fill_missing_nearest(tile)
    current = filled
    sizes = _window_sizes(config)
    for size in sizes:
        opened = ndimage.grey_opening(filled, size=(size, size))
        window_m = size * config.resolution
        threshold = max(
            config.initial_distance_m + config.slope * window_m,
            config.height_threshold_m,
        )
        # The opening removes features smaller than the current window. The
        # threshold is retained as the PMF progressive acceptance bound.
        current = opened
        filled = np.where(filled - opened <= threshold, opened, filled)
    return current


def progressive_morphological_filter(points, config=None):
    """Return point masks and the PMF ground surface for finite XYZ points."""
    config = config if isinstance(config, PMFConfig) else PMFConfig(**(config or {}))
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] < 3 or len(points) == 0:
        raise ValueError("points must be a non-empty Nx3 array")
    finite = np.isfinite(points[:, :3]).all(axis=1)
    if not finite.any():
        raise ValueError("points contain no finite XYZ samples")
    height, origin = points_to_height_grid(
        points[finite, :3], config.resolution, config.chunk_size, return_origin=True
    )
    max_radius = max(size // 2 for size in _window_sizes(config))
    ground_surface = np.full_like(height, np.nan, dtype=np.float32)
    tile = config.tile_size
    for row in range(0, height.shape[0], tile):
        for column in range(0, height.shape[1], tile):
            row_stop, column_stop = min(row + tile, height.shape[0]), min(column + tile, height.shape[1])
            row_start, column_start = max(0, row - max_radius), max(0, column - max_radius)
            filtered = _pmf_tile(height[row_start:row_stop + max_radius, column_start:column_stop + max_radius], config)
            row_offset, column_offset = row - row_start, column - column_start
            ground_surface[row:row_stop, column:column_stop] = filtered[
                row_offset:row_offset + row_stop - row,
                column_offset:column_offset + column_stop - column,
            ]

    indices = np.floor((points[:, :2] - origin) / config.resolution).astype(np.int64)
    valid_index = finite & (indices[:, 0] >= 0) & (indices[:, 1] >= 0)
    valid_index &= indices[:, 0] < height.shape[1]
    valid_index &= indices[:, 1] < height.shape[0]
    ground_mask = np.zeros(len(points), dtype=bool)
    point_threshold = config.height_threshold_m + config.slope * config.resolution
    ground_mask[valid_index] = points[valid_index, 2] <= (
        ground_surface[indices[valid_index, 1], indices[valid_index, 0]] + point_threshold
    )
    observed_grid = np.isfinite(height)
    ground_grid = np.zeros_like(observed_grid)
    ground_grid[indices[valid_index, 1], indices[valid_index, 0]] |= ground_mask[valid_index]
    return {
        "ground": points[ground_mask],
        "non_ground": points[~ground_mask],
        "ground_surface": ground_surface,
        "observed_grid": observed_grid,
        "ground_grid": ground_grid,
        "origin_xy": origin,
        "resolution": config.resolution,
        "name": "morphological_pmf",
    }


def segment(points: np.ndarray, config=None):
    """Compatibility entry point for the real tiled PMF implementation."""
    return progressive_morphological_filter(points, config)
