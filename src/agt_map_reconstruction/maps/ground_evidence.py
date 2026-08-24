"""Conservative four-state ground evidence and navigation costs for EXP003."""

from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from numbers import Integral, Real

import numpy as np
from scipy import interpolate, ndimage
from scipy.spatial import QhullError


class EvidenceClass(IntEnum):
    """Stable evidence labels stored in EXP003's evidence grid."""

    UNKNOWN = 0
    FREE_CONFIRMED = 1
    OCCUPIED_CONFIRMED = 2
    GROUND_INTERPOLATED = 3


@dataclass(frozen=True)
class GroundEvidenceConfig:
    """Metric thresholds used to convert elevation statistics into evidence."""

    resolution: float = 0.05
    min_points_per_cell: int = 3
    min_ground_support_cells: int = 2
    ground_window_m: float = 0.50
    ground_percentile: float = 20.0
    ground_seed_percentile: float = 10.0
    max_ground_step_m: float = 0.20
    max_interpolation_gap_m: float = 0.25
    obstacle_height_m: float = 0.15
    obstacle_inflation_radius_m: float = 0.25
    interpolated_ground_cost: int = 64


@dataclass(frozen=True)
class GroundEvidenceResult:
    """Consistent ground model, measured clearance, and evidence labels."""

    ground_surface: np.ndarray
    clearance: np.ndarray
    ground_model_support: np.ndarray
    evidence: np.ndarray


def _validate_config(config):
    if not isinstance(config, GroundEvidenceConfig):
        raise TypeError("config must be a GroundEvidenceConfig")
    if not isinstance(config.resolution, Real) or not np.isfinite(config.resolution) or config.resolution <= 0:
        raise ValueError("resolution must be finite and positive")
    for name in ("min_points_per_cell", "min_ground_support_cells"):
        value = getattr(config, name)
        if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name in (
        "ground_window_m",
        "max_ground_step_m",
        "max_interpolation_gap_m",
        "obstacle_height_m",
        "obstacle_inflation_radius_m",
    ):
        value = getattr(config, name)
        if (
            not isinstance(value, Real)
            or isinstance(value, bool)
            or not np.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{name} must be finite and non-negative")
    if config.ground_window_m == 0:
        raise ValueError("ground_window_m must be positive")
    for name in ("ground_percentile", "ground_seed_percentile"):
        value = getattr(config, name)
        if (
            not isinstance(value, Real)
            or isinstance(value, bool)
            or not np.isfinite(value)
            or not 0.0 <= value <= 100.0
        ):
            raise ValueError(f"{name} must be finite and between 0 and 100")
    if (
        not isinstance(config.interpolated_ground_cost, Integral)
        or isinstance(config.interpolated_ground_cost, bool)
        or not 1 <= config.interpolated_ground_cost <= 253
    ):
        raise ValueError("interpolated_ground_cost must be an integer between 1 and 253")


def _validate_measurements(low_height, point_count):
    low_height = np.asarray(low_height, dtype=np.float64)
    point_count = np.asarray(point_count)
    if low_height.ndim != 2 or point_count.ndim != 2:
        raise ValueError("low_height and point_count must be two-dimensional")
    if low_height.shape != point_count.shape:
        raise ValueError("low_height and point_count must have matching shapes")
    if np.any(~np.isfinite(point_count)) or np.any(point_count < 0):
        raise ValueError("point_count must contain finite, non-negative values")
    return low_height, point_count


def _disk_footprint(radius_cells):
    coordinates = np.arange(-radius_cells, radius_cells + 1)
    rows, columns = np.meshgrid(coordinates, coordinates, indexing="ij")
    return rows * rows + columns * columns <= radius_cells * radius_cells


def _nan_percentile(values, percentile):
    finite_values = values[np.isfinite(values)]
    if not len(finite_values):
        return np.nan
    return np.percentile(finite_values, percentile)


def _ground_footprint(config):
    radius_cells = int(np.ceil(config.ground_window_m / config.resolution))
    footprint = _disk_footprint(radius_cells)
    footprint[radius_cells, radius_cells] = False
    return footprint


def _estimate_ground_surface(measured_height, config, footprint):
    return ndimage.generic_filter(
        measured_height,
        _nan_percentile,
        footprint=footprint,
        mode="constant",
        cval=np.nan,
        extra_arguments=(config.ground_percentile,),
    )


def _propagate_ground_support(measured_height, measured, footprint, config):
    neighbor_count = ndimage.convolve(
        measured.astype(np.int32),
        footprint.astype(np.int32),
        mode="constant",
        cval=0,
    )
    eligible = measured & (neighbor_count >= config.min_ground_support_cells)
    reachable = np.zeros(measured.shape, dtype=bool)
    if not eligible.any():
        return reachable

    seed_ceiling = np.percentile(
        measured_height[eligible], config.ground_seed_percentile
    )
    seeds = eligible & (measured_height <= seed_ceiling)
    queue = deque(map(tuple, np.column_stack(np.nonzero(seeds))))
    reachable[seeds] = True
    row_count, column_count = measured.shape
    while queue:
        row, column = queue.popleft()
        height = measured_height[row, column]
        for row_offset in (-1, 0, 1):
            for column_offset in (-1, 0, 1):
                if row_offset == 0 and column_offset == 0:
                    continue
                neighbor_row = row + row_offset
                neighbor_column = column + column_offset
                if not (
                    0 <= neighbor_row < row_count
                    and 0 <= neighbor_column < column_count
                ):
                    continue
                if reachable[neighbor_row, neighbor_column] or not eligible[
                    neighbor_row, neighbor_column
                ]:
                    continue
                if abs(measured_height[neighbor_row, neighbor_column] - height) <= (
                    config.max_ground_step_m
                ):
                    reachable[neighbor_row, neighbor_column] = True
                    queue.append((neighbor_row, neighbor_column))
    return reachable


def _interpolate_bounded_components(ground_surface, source, unsupported, config):
    interpolated_surface = ground_surface.copy()
    filled = np.zeros(unsupported.shape, dtype=bool)
    labels, component_count = ndimage.label(
        unsupported,
        structure=np.ones((3, 3), dtype=bool),
    )
    for component_label in range(1, component_count + 1):
        component = labels == component_label
        coordinates = np.column_stack(np.nonzero(component))
        if (
            np.any(coordinates[:, 0] == 0)
            or np.any(coordinates[:, 0] == component.shape[0] - 1)
            or np.any(coordinates[:, 1] == 0)
            or np.any(coordinates[:, 1] == component.shape[1] - 1)
        ):
            continue
        boundary = ndimage.binary_dilation(
            component,
            structure=np.ones((3, 3), dtype=bool),
        ) & ~component
        if not boundary.any() or not np.all(source[boundary]):
            continue
        distance_to_boundary = ndimage.distance_transform_edt(
            ~boundary,
            sampling=config.resolution,
        )
        if np.max(distance_to_boundary[component]) > config.max_interpolation_gap_m:
            continue
        span_cells = np.ptp(coordinates, axis=0) + 1
        if np.max(span_cells * config.resolution) > (
            2 * config.max_interpolation_gap_m + config.resolution
        ):
            continue

        boundary_coordinates = np.column_stack(np.nonzero(boundary))
        component_values = []
        supported = True
        for coordinate in coordinates:
            offsets = (
                boundary_coordinates - coordinate
            ) * config.resolution
            local = np.linalg.norm(offsets, axis=1) <= config.max_interpolation_gap_m
            local_coordinates = boundary_coordinates[local]
            local_offsets = offsets[local]
            if len(local_coordinates) < 3:
                supported = False
                break
            vertical_support = (
                np.any(local_offsets[:, 0] < 0)
                and np.any(local_offsets[:, 0] > 0)
            )
            horizontal_support = (
                np.any(local_offsets[:, 1] < 0)
                and np.any(local_offsets[:, 1] > 0)
            )
            if not (vertical_support or horizontal_support):
                supported = False
                break
            try:
                value = interpolate.griddata(
                    local_coordinates,
                    ground_surface[boundary][local],
                    coordinate[None, :],
                    method="linear",
                )[0]
            except QhullError:
                supported = False
                break
            if not np.isfinite(value):
                supported = False
                break
            component_values.append(value)
        if supported:
            interpolated_surface[coordinates[:, 0], coordinates[:, 1]] = component_values
            filled[component] = True
    return interpolated_surface, filled


def build_ground_evidence_details(low_height, point_count, config):
    """Build one consistent bounded ground surface and evidence classification.

    The ground model uses only cells meeting the measurement-density threshold.
    Unsupported cells are labeled as interpolated only when they are within the
    metric gap bound and linear interpolation from confirmed ground succeeds.
    """
    _validate_config(config)
    low_height, point_count = _validate_measurements(low_height, point_count)

    measured = (point_count >= config.min_points_per_cell) & np.isfinite(low_height)
    measured_height = np.where(measured, low_height, np.nan)
    footprint = _ground_footprint(config)
    propagated_ground = _propagate_ground_support(
        measured_height, measured, footprint, config
    )
    propagated_height = np.where(propagated_ground, low_height, np.nan)
    ground_surface = _estimate_ground_surface(propagated_height, config, footprint)
    propagated_neighbor_count = ndimage.convolve(
        propagated_ground.astype(np.int32),
        footprint.astype(np.int32),
        mode="constant",
        cval=0,
    )
    ground_model_support = (
        measured
        & np.isfinite(ground_surface)
        & (propagated_neighbor_count >= config.min_ground_support_cells)
    )
    confirmed_free = (
        ground_model_support
        & np.isfinite(ground_surface)
        & (low_height - ground_surface <= config.obstacle_height_m)
    )
    confirmed_occupied = (
        ground_model_support
        & np.isfinite(ground_surface)
        & ~confirmed_free
    )

    evidence = np.full(low_height.shape, EvidenceClass.UNKNOWN, dtype=np.uint8)
    evidence[confirmed_free] = EvidenceClass.FREE_CONFIRMED
    evidence[confirmed_occupied] = EvidenceClass.OCCUPIED_CONFIRMED

    if confirmed_free.any() and config.max_interpolation_gap_m > 0:
        ground_surface, interpolated = _interpolate_bounded_components(
            ground_surface,
            confirmed_free,
            ~measured,
            config,
        )
        evidence[interpolated] = EvidenceClass.GROUND_INTERPOLATED
    ground_surface = np.where(evidence != EvidenceClass.UNKNOWN, ground_surface, np.nan)
    clearance = np.where(measured, low_height - ground_surface, np.nan)
    return GroundEvidenceResult(
        ground_surface=ground_surface,
        clearance=clearance,
        ground_model_support=ground_model_support,
        evidence=evidence,
    )


def build_ground_evidence(low_height, point_count, config):
    """Return the stable evidence grid from the detailed Task 2 result."""
    return build_ground_evidence_details(low_height, point_count, config).evidence


def build_navigation_costmap(evidence, config):
    """Return a conservative uint8 costmap from four-state ground evidence."""
    _validate_config(config)
    evidence = np.asarray(evidence)
    if evidence.ndim != 2:
        raise ValueError("evidence must be two-dimensional")
    valid_labels = {int(label) for label in EvidenceClass}
    if not np.isin(evidence, tuple(valid_labels)).all():
        raise ValueError("evidence contains an unknown label")

    evidence = evidence.astype(np.uint8, copy=False)
    costmap = np.full(evidence.shape, 255, dtype=np.uint8)
    free = evidence == EvidenceClass.FREE_CONFIRMED
    interpolated = evidence == EvidenceClass.GROUND_INTERPOLATED
    occupied = evidence == EvidenceClass.OCCUPIED_CONFIRMED
    costmap[free] = 0
    costmap[interpolated] = config.interpolated_ground_cost
    costmap[occupied] = 254

    if occupied.any() and config.obstacle_inflation_radius_m > 0:
        distance_to_obstacle = ndimage.distance_transform_edt(
            ~occupied,
            sampling=config.resolution,
        )
        inflated = distance_to_obstacle <= config.obstacle_inflation_radius_m
        costmap[inflated & (evidence != EvidenceClass.UNKNOWN)] = 254
    return costmap
