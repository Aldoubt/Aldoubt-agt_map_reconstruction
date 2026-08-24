import numpy as np

from .elevation_normalization import normalize_height
from .traversability import compute_traversability


def _grid_indices(xy, min_xy, resolution):
    scaled = (xy - min_xy) / resolution
    scaled = np.nextafter(scaled, np.asarray(np.inf, dtype=scaled.dtype))
    return np.floor(scaled).astype(np.int64, copy=False)


def points_to_height_grid(points, resolution=0.05, chunk_size=1_000_000,
                          return_origin=False):
    """Rasterize the minimum observed Z without allocating all indices.

    Chunked ``minimum.at`` keeps the operation practical for the 85M-point
    FAST-LIVO2 map used by EXP002 and preserves the original minimum-height
    grid semantics.
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] < 3 or len(points) == 0:
        raise ValueError("points must be a non-empty Nx3 array")
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    min_xy = np.array([np.inf, np.inf], dtype=np.float64)
    max_xy = np.array([-np.inf, -np.inf], dtype=np.float64)
    valid_points = 0
    for start in range(0, len(points), chunk_size):
        chunk = points[start:start + chunk_size]
        finite = np.isfinite(chunk[:, :3]).all(axis=1)
        if not finite.any():
            continue
        xy = chunk[finite, :2]
        min_xy = np.minimum(min_xy, xy.min(axis=0))
        max_xy = np.maximum(max_xy, xy.max(axis=0))
        valid_points += int(finite.sum())
    if valid_points == 0:
        raise ValueError("points contain no finite XYZ samples")

    coordinate_dtype = points.dtype if np.issubdtype(points.dtype, np.floating) else np.float64
    min_xy = min_xy.astype(coordinate_dtype, copy=False)
    max_xy = max_xy.astype(coordinate_dtype, copy=False)
    size_xy = _grid_indices(max_xy[None, :], min_xy, resolution)[0] + 1
    width, height = int(size_xy[0]), int(size_xy[1])
    flat = np.full(width * height, np.inf, dtype=np.float32)

    for start in range(0, len(points), chunk_size):
        chunk = points[start:start + chunk_size]
        finite = np.isfinite(chunk[:, :3]).all(axis=1)
        if not finite.any():
            continue
        chunk = chunk[finite]
        indices = _grid_indices(chunk[:, :2], min_xy, resolution)
        flat_indices = indices[:, 1] * width + indices[:, 0]
        np.minimum.at(flat, flat_indices, chunk[:, 2])

    grid = flat.reshape(height, width)
    grid[np.isinf(grid)] = np.nan
    if return_origin:
        return grid, min_xy.astype(float, copy=False)
    return grid


def build_traversability_map(points, resolution=0.05, kernel_size=5,
                             chunk_size=1_000_000):
    height_grid, origin_xy = points_to_height_grid(
        points,
        resolution,
        chunk_size,
        return_origin=True,
    )
    relative_height = normalize_height(height_grid, kernel_size)
    traversability = compute_traversability(relative_height)
    return {
        "height": height_grid,
        "relative_height": relative_height,
        "traversability": traversability,
        "origin_xy": origin_xy,
        "resolution": float(resolution),
    }
