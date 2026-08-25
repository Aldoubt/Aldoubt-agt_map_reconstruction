"""Recover crop rows, bounded aisles, and wall-like structures from a height grid."""

from dataclasses import dataclass, asdict
import json

import numpy as np
from scipy import ndimage, signal


@dataclass(frozen=True)
class RowStructureConfig:
    resolution: float = 0.05
    ridge_height_threshold_m: float = 0.15
    min_row_length_m: float = 2.0
    min_row_width_m: float = 0.10
    max_row_width_m: float = 1.0
    min_row_spacing_m: float = 0.80
    min_aisle_width_m: float = 0.60
    vehicle_width_m: float = 0.60
    repair_gap_m: float = 0.30
    wall_height_threshold_m: float = 0.25
    wall_min_length_m: float = 0.50

    def __post_init__(self):
        for name in ("resolution", "ridge_height_threshold_m", "min_row_length_m", "min_row_width_m", "max_row_width_m", "min_row_spacing_m", "min_aisle_width_m", "vehicle_width_m", "wall_height_threshold_m", "wall_min_length_m"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.repair_gap_m < 0 or not np.isfinite(self.repair_gap_m):
            raise ValueError("repair_gap_m must be finite and non-negative")


def _row_frame(array, angle_rad):
    return ndimage.rotate(array, np.degrees(angle_rad), reshape=False, order=0, mode="constant", cval=np.nan, prefilter=False)


def _original_frame(array, angle_rad):
    return ndimage.rotate(array, -np.degrees(angle_rad), reshape=False, order=0, mode="constant", cval=0, prefilter=False).astype(bool)


def _original_values(array, angle_rad):
    return ndimage.rotate(array, -np.degrees(angle_rad), reshape=False, order=0, mode="constant", cval=np.nan, prefilter=False)


def _fill_along_rows(mask, gap_cells):
    """Repair only short gaps along columns, never across row bands."""
    result = np.asarray(mask, dtype=bool).copy()
    if gap_cells <= 0:
        return result
    structure = np.ones((1, 2 * gap_cells + 1), dtype=bool)
    return ndimage.binary_closing(result, structure=structure)


def analyze_row_structure(height_grid, row_direction=(1.0, 0.0), config=None, origin_xy=(0.0, 0.0), obstacle_grid=None):
    """Return row segments, bounded aisles, wall candidates, and overlays.

    The input is expected to be a ground-height raster.  The direction is the
    row axis in image coordinates; all morphology is performed after rotating
    into that frame, where rows run along columns.
    """
    config = config if isinstance(config, RowStructureConfig) else RowStructureConfig(**(config or {}))
    height = np.asarray(height_grid, dtype=np.float64)
    if height.ndim != 2:
        raise ValueError("height_grid must be two-dimensional")
    if obstacle_grid is not None:
        obstacle_grid = np.asarray(obstacle_grid, dtype=bool)
        if obstacle_grid.shape != height.shape:
            raise ValueError("obstacle_grid must have the same shape as height_grid")
    direction = np.asarray(row_direction, dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm == 0 or not np.isfinite(norm):
        raise ValueError("row_direction must be finite and non-zero")
    angle = float(np.arctan2(direction[1], direction[0]))
    rotated = _row_frame(height, angle)
    valid = np.isfinite(rotated)
    if not valid.any():
        raise ValueError("height_grid contains no finite values")
    filled = rotated.copy()
    nearest = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    filled[~valid] = rotated[tuple(nearest)][~valid]
    profile = np.percentile(filled, 75, axis=1)
    baseline = float(np.percentile(profile, 25))
    smooth = ndimage.gaussian_filter1d(profile, sigma=1.5)
    peak_distance = max(1, int(np.ceil(config.min_row_spacing_m / config.resolution)))
    peaks, _ = signal.find_peaks(smooth, height=baseline + config.ridge_height_threshold_m, distance=peak_distance)
    rows = []
    ridge_mask = np.zeros_like(rotated, dtype=bool)
    for row_id, peak in enumerate(peaks):
        band = smooth >= smooth[peak] - config.ridge_height_threshold_m * 0.5
        start, stop = int(peak), int(peak)
        while start > 0 and band[start - 1]:
            start -= 1
        while stop + 1 < len(band) and band[stop + 1]:
            stop += 1
        width_m = (stop - start + 1) * config.resolution
        if not config.min_row_width_m <= width_m <= config.max_row_width_m:
            continue
        support = valid[start:stop + 1].any(axis=0)
        columns = np.flatnonzero(support)
        if not columns.size or (columns[-1] - columns[0] + 1) * config.resolution < config.min_row_length_m:
            continue
        ridge_mask[start:stop + 1, columns[0]:columns[-1] + 1] = True
        start_xy = np.asarray(origin_xy) + np.array([columns[0] * config.resolution, start * config.resolution])
        end_xy = np.asarray(origin_xy) + np.array([columns[-1] * config.resolution, stop * config.resolution])
        rows.append({"id": row_id, "start_xy": start_xy.tolist(), "end_xy": end_xy.tolist(), "width_m": width_m, "length_m": float((columns[-1] - columns[0] + 1) * config.resolution)})

    aisle = np.zeros_like(rotated, dtype=bool)
    accepted_aisles = []
    for left, right in zip(rows, rows[1:]):
        left_stop = int(round(left["end_xy"][1] / config.resolution - origin_xy[1] / config.resolution))
        right_start = int(round(right["start_xy"][1] / config.resolution - origin_xy[1] / config.resolution))
        if right_start <= left_stop:
            continue
        width_m = (right_start - left_stop - 1) * config.resolution
        if width_m < max(config.min_aisle_width_m, config.vehicle_width_m):
            continue
        supported = valid[left_stop + 1:right_start].any(axis=0)
        supported = _fill_along_rows(supported[None, :], int(np.floor(config.repair_gap_m / config.resolution)))[0]
        columns = np.flatnonzero(supported)
        if not columns.size:
            continue
        aisle[left_stop + 1:right_start, columns[0]:columns[-1] + 1] = True
        accepted_aisles.append({"between_rows": [left["id"], right["id"]], "width_m": width_m, "length_m": float((columns[-1] - columns[0] + 1) * config.resolution), "vehicle_width_fits": width_m >= config.vehicle_width_m})

    residual = filled - ndimage.grey_opening(filled, size=(5, 5))
    wall = (residual >= config.wall_height_threshold_m) & valid & ~ridge_mask
    if obstacle_grid is not None:
        external_wall = _row_frame(obstacle_grid.astype(float), angle) >= 0.5
        # Keep externally supplied occupied structures separate from the
        # height-residual blobs so a long ridge cannot merge with a support.
        wall = external_wall & valid & ~ridge_mask
    labels, count = ndimage.label(wall, structure=np.ones((3, 3), dtype=bool))
    wall_mask = np.zeros_like(wall)
    walls = []
    for label in range(1, count + 1):
        yy, xx = np.where(labels == label)
        if len(xx) < 2:
            continue
        # Row-like components are already represented as ridges; wall
        # candidates must have a clear cross-row (normal) extent.
        if np.ptp(xx) > np.ptp(yy) * 0.75:
            continue
        length = max(np.ptp(xx), np.ptp(yy)) * config.resolution
        if length < config.wall_min_length_m:
            continue
        wall_mask[labels == label] = True
        walls.append({"id": len(walls), "start_xy": (np.asarray(origin_xy) + np.array([xx.min(), yy.min()]) * config.resolution).tolist(), "end_xy": (np.asarray(origin_xy) + np.array([xx.max(), yy.max()]) * config.resolution).tolist(), "length_m": float(length)})

    row_widths = [row["width_m"] for row in rows]
    return {
        "row_direction": direction.tolist(),
        "row_angle_rad": angle,
        "row_count": len(rows),
        "mean_row_width_m": float(np.mean(row_widths)) if row_widths else None,
        "row_width_range_m": [float(min(row_widths)), float(max(row_widths))] if row_widths else None,
        "vehicle_fitting_aisle_count": sum(item["vehicle_width_fits"] for item in accepted_aisles),
        "rows": rows,
        "aisles": accepted_aisles,
        "wall_count": len(walls),
        "walls": walls,
        "ridge_mask": _original_frame(ridge_mask, angle),
        "aisle_mask": _original_frame(aisle, angle),
        "wall_mask": _original_frame(wall_mask, angle),
        "filled_height": _original_values(filled, angle),
    }


def save_structure_json(result, path):
    serializable = {key: value for key, value in result.items() if not isinstance(value, np.ndarray)}
    serializable["config"] = asdict(RowStructureConfig())
    from pathlib import Path
    Path(path).write_text(json.dumps(serializable, indent=2) + "\n", encoding="utf-8")
