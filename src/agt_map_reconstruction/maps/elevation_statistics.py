"""Chunked, robust per-cell elevation statistics for EXP003."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class ElevationStatistics:
    """Rasterized elevation measurements and their grid metadata.

    Empty cells have a point count of zero and ``NaN`` elevation values.
    ``low_height`` is the lower edge of the fixed histogram bin containing
    the requested lower cumulative quantile.
    """

    low_height: np.ndarray
    point_count: np.ndarray
    minimum_height: np.ndarray
    maximum_height: np.ndarray
    q10_height: np.ndarray
    q50_height: np.ndarray
    q90_height: np.ndarray
    origin_xy: np.ndarray
    resolution: float


def _grid_indices(xy, origin_xy, resolution):
    scaled = (xy - origin_xy) / resolution
    scaled = np.nextafter(scaled, np.asarray(np.inf, dtype=scaled.dtype))
    return np.floor(scaled).astype(np.int64, copy=False)


def _validate_inputs(points, resolution, chunk_size, low_quantile, histogram_bins):
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] < 3 or len(points) == 0:
        raise ValueError("points must be a non-empty Nx3 array")
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("resolution must be finite and positive")
    if not isinstance(chunk_size, Integral) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if not np.isfinite(low_quantile) or not 0.0 <= low_quantile <= 1.0:
        raise ValueError("low_quantile must be finite and between 0 and 1")
    if (
        not isinstance(histogram_bins, Integral)
        or isinstance(histogram_bins, bool)
        or histogram_bins <= 0
    ):
        raise ValueError("histogram_bins must be a positive integer")
    return points, int(chunk_size), int(histogram_bins)


def points_to_elevation_statistics(
    points,
    resolution=0.05,
    chunk_size=1_000_000,
    low_quantile=0.10,
    histogram_bins=64,
):
    """Rasterize finite XYZ points into robust, bounded-memory statistics.

    The first chunked scan establishes finite XY bounds.  The second fills
    per-cell counts and extrema, and the third fills fixed-bin histograms over
    each cell's own observed Z range.  All point-index arrays are scoped to a
    single chunk.
    """
    points, chunk_size, histogram_bins = _validate_inputs(
        points, resolution, chunk_size, low_quantile, histogram_bins
    )

    min_xy = np.array([np.inf, np.inf], dtype=np.float64)
    max_xy = np.array([-np.inf, -np.inf], dtype=np.float64)
    finite_point_count = 0
    for start in range(0, len(points), chunk_size):
        chunk = points[start:start + chunk_size]
        finite = np.isfinite(chunk[:, :3]).all(axis=1)
        if not finite.any():
            continue
        xyz = chunk[finite, :3]
        min_xy = np.minimum(min_xy, xyz[:, :2].min(axis=0))
        max_xy = np.maximum(max_xy, xyz[:, :2].max(axis=0))
        finite_point_count += int(finite.sum())
    if finite_point_count == 0:
        raise ValueError("points contain no finite XYZ samples")

    origin_xy = min_xy.astype(np.float64, copy=False)
    size_xy = _grid_indices(max_xy[None, :], origin_xy, resolution)[0] + 1
    width, height = int(size_xy[0]), int(size_xy[1])
    cell_count = width * height
    point_count = np.zeros(cell_count, dtype=np.int64)
    minimum_height = np.full(cell_count, np.inf, dtype=np.float64)
    maximum_height = np.full(cell_count, -np.inf, dtype=np.float64)
    histogram = np.zeros((cell_count, histogram_bins), dtype=np.int64)

    for start in range(0, len(points), chunk_size):
        chunk = points[start:start + chunk_size]
        finite = np.isfinite(chunk[:, :3]).all(axis=1)
        if not finite.any():
            continue
        xyz = chunk[finite, :3]
        indices = _grid_indices(xyz[:, :2], origin_xy, resolution)
        flat_indices = indices[:, 1] * width + indices[:, 0]

        np.add.at(point_count, flat_indices, 1)
        np.minimum.at(minimum_height, flat_indices, xyz[:, 2])
        np.maximum.at(maximum_height, flat_indices, xyz[:, 2])

    for start in range(0, len(points), chunk_size):
        chunk = points[start:start + chunk_size]
        finite = np.isfinite(chunk[:, :3]).all(axis=1)
        if not finite.any():
            continue
        xyz = chunk[finite, :3]
        indices = _grid_indices(xyz[:, :2], origin_xy, resolution)
        flat_indices = indices[:, 1] * width + indices[:, 0]
        cell_minimum = minimum_height[flat_indices]
        cell_range = maximum_height[flat_indices] - cell_minimum
        bins = np.zeros(len(xyz), dtype=np.int64)
        varying_range = cell_range > 0.0
        bins[varying_range] = np.floor(
            (xyz[varying_range, 2] - cell_minimum[varying_range])
            * histogram_bins
            / cell_range[varying_range]
        ).astype(np.int64, copy=False)
        bins = np.clip(bins, 0, histogram_bins - 1)
        np.add.at(histogram, (flat_indices, bins), 1)

    target_count = np.maximum(1, np.ceil(point_count * low_quantile).astype(np.int64))
    cumulative = np.cumsum(histogram, axis=1)
    def quantile_height(quantile):
        targets = np.maximum(1, np.ceil(point_count * quantile).astype(np.int64))
        selected = (cumulative >= targets[:, None]).argmax(axis=1)
        values = np.full(cell_count, np.nan, dtype=np.float64)
        values[observed] = minimum_height[observed] + (
            selected[observed]
            * (maximum_height[observed] - minimum_height[observed])
            / histogram_bins
        )
        return values

    selected_bin = (cumulative >= target_count[:, None]).argmax(axis=1)
    empty = point_count == 0
    low_height = np.full(cell_count, np.nan, dtype=np.float64)
    observed = ~empty
    low_height[observed] = minimum_height[observed] + (
        selected_bin[observed]
        * (maximum_height[observed] - minimum_height[observed])
        / histogram_bins
    )
    q10_height = quantile_height(0.10)
    q50_height = quantile_height(0.50)
    q90_height = quantile_height(0.90)
    minimum_height[empty] = np.nan
    maximum_height[empty] = np.nan
    grid_shape = (height, width)
    return ElevationStatistics(
        low_height=low_height.reshape(grid_shape),
        point_count=point_count.reshape(grid_shape),
        minimum_height=minimum_height.reshape(grid_shape),
        maximum_height=maximum_height.reshape(grid_shape),
        q10_height=q10_height.reshape(grid_shape),
        q50_height=q50_height.reshape(grid_shape),
        q90_height=q90_height.reshape(grid_shape),
        origin_xy=origin_xy,
        resolution=float(resolution),
    )
