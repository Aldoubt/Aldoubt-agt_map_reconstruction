"""Conservative 3D ray support for observed-free ground evidence.

A ray supports a grid cell only when the ray segment inside that cell lies in a
configured low-height band above the local ground surface. The return/hit cell
is never marked free. This module produces support counts only; it does not edit
the static map or promote semantics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GroundAwareRayConfig:
    min_ground_relative_height_m: float
    max_ground_relative_height_m: float
    min_support_rays: int
    min_ray_range_m: float = 0.0
    max_ray_range_m: float | None = None

    def __post_init__(self):
        values = (
            self.min_ground_relative_height_m,
            self.max_ground_relative_height_m,
            self.min_ray_range_m,
        )
        if not all(np.isfinite(float(value)) for value in values):
            raise ValueError("ray evidence thresholds must be finite")
        if float(self.min_ground_relative_height_m) < 0.0:
            raise ValueError("min_ground_relative_height_m must be >= 0")
        if (
            float(self.max_ground_relative_height_m)
            <= float(self.min_ground_relative_height_m)
        ):
            raise ValueError(
                "max_ground_relative_height_m must be greater than the minimum"
            )
        if int(self.min_support_rays) < 1:
            raise ValueError("min_support_rays must be >= 1")
        if float(self.min_ray_range_m) < 0.0:
            raise ValueError("min_ray_range_m must be >= 0")
        if self.max_ray_range_m is not None:
            if not np.isfinite(float(self.max_ray_range_m)):
                raise ValueError("max_ray_range_m must be finite when supplied")
            if float(self.max_ray_range_m) <= float(self.min_ray_range_m):
                raise ValueError("max_ray_range_m must exceed min_ray_range_m")


def _world_to_grid_continuous(xyz, metadata):
    xyz = np.asarray(xyz, dtype=np.float64)
    dx = xyz[..., 0] - float(metadata.origin_x)
    dy = xyz[..., 1] - float(metadata.origin_y)
    c = float(np.cos(metadata.origin_yaw))
    s = float(np.sin(metadata.origin_yaw))
    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy
    return np.stack(
        (local_x / float(metadata.resolution), local_y / float(metadata.resolution)),
        axis=-1,
    )


def _clip_segment_to_grid(p0, p1, width, height):
    """Return original-ray t interval inside [0,width] x [0,height]."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    direction = p1 - p0
    t_enter = 0.0
    t_exit = 1.0
    for coordinate, delta, limit in (
        (p0[0], direction[0], float(width)),
        (p0[1], direction[1], float(height)),
    ):
        if abs(float(delta)) <= 1e-15:
            if coordinate < 0.0 or coordinate >= limit:
                return None
            continue
        a = (0.0 - coordinate) / delta
        b = (limit - coordinate) / delta
        near = min(a, b)
        far = max(a, b)
        t_enter = max(t_enter, float(near))
        t_exit = min(t_exit, float(far))
        if t_enter >= t_exit - 1e-15:
            return None
    return max(0.0, t_enter), min(1.0, t_exit)


def _traverse_clipped_cells(p0, p1, width, height, t0, t1):
    """Yield (x, y, original_t_enter, original_t_exit) via exact 2D DDA."""
    direction = np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)
    q0 = np.asarray(p0, dtype=float) + float(t0) * direction
    q1 = np.asarray(p0, dtype=float) + float(t1) * direction

    upper_x = np.nextafter(float(width), -np.inf)
    upper_y = np.nextafter(float(height), -np.inf)
    q0[0] = np.clip(q0[0], 0.0, upper_x)
    q0[1] = np.clip(q0[1], 0.0, upper_y)
    q1[0] = np.clip(q1[0], 0.0, upper_x)
    q1[1] = np.clip(q1[1], 0.0, upper_y)

    delta = q1 - q0
    x = int(np.floor(q0[0]))
    y = int(np.floor(q0[1]))
    end_x = int(np.floor(q1[0]))
    end_y = int(np.floor(q1[1]))

    if delta[0] > 0.0:
        step_x = 1
        t_max_x = (x + 1.0 - q0[0]) / delta[0]
        t_delta_x = 1.0 / delta[0]
    elif delta[0] < 0.0:
        step_x = -1
        t_max_x = (q0[0] - x) / (-delta[0])
        t_delta_x = 1.0 / (-delta[0])
    else:
        step_x = 0
        t_max_x = np.inf
        t_delta_x = np.inf

    if delta[1] > 0.0:
        step_y = 1
        t_max_y = (y + 1.0 - q0[1]) / delta[1]
        t_delta_y = 1.0 / delta[1]
    elif delta[1] < 0.0:
        step_y = -1
        t_max_y = (q0[1] - y) / (-delta[1])
        t_delta_y = 1.0 / (-delta[1])
    else:
        step_y = 0
        t_max_y = np.inf
        t_delta_y = np.inf

    local_enter = 0.0
    while 0 <= x < width and 0 <= y < height:
        local_exit = min(float(t_max_x), float(t_max_y), 1.0)
        original_enter = float(t0) + local_enter * (float(t1) - float(t0))
        original_exit = float(t0) + local_exit * (float(t1) - float(t0))
        yield x, y, original_enter, original_exit
        if x == end_x and y == end_y:
            break
        if local_exit >= 1.0 - 1e-15:
            break

        cross_x = abs(float(t_max_x) - local_exit) <= 1e-12
        cross_y = abs(float(t_max_y) - local_exit) <= 1e-12
        if cross_x:
            x += step_x
            t_max_x += t_delta_x
        if cross_y:
            y += step_y
            t_max_y += t_delta_y
        local_enter = local_exit


def accumulate_ground_aware_ray_support(bundle, ground_surface, metadata, config):
    """Accumulate low-height line-of-sight support per grid cell.

    ``support_count`` counts supporting rays. When ``bundle.scan_index`` is
    present, ``scan_support_count`` counts at most one support per physical scan
    per cell. Scan indices must be non-decreasing so same-scan rays remain one
    contiguous temporal group. The hit cell is never free and NaN ground cells
    cannot receive support.
    """
    from .observation_ray_bundle import ObservationRayBundle

    if not isinstance(bundle, ObservationRayBundle):
        raise TypeError("bundle must be an ObservationRayBundle")
    if not isinstance(config, GroundAwareRayConfig):
        raise TypeError("config must be a GroundAwareRayConfig")
    if bundle.frame_id != str(metadata.frame_id):
        raise ValueError(
            f"ray frame_id={bundle.frame_id!r} does not match grid frame "
            f"{metadata.frame_id!r}"
        )

    ground = np.asarray(ground_surface, dtype=np.float64)
    expected_shape = (int(metadata.height), int(metadata.width))
    if ground.shape != expected_shape:
        raise ValueError(
            f"ground_surface shape {ground.shape} does not match grid {expected_shape}"
        )

    origins = np.asarray(bundle.ray_origin_xyz_m, dtype=np.float64)
    endpoints = np.asarray(bundle.ray_endpoint_xyz_m, dtype=np.float64)
    origins_grid = _world_to_grid_continuous(origins, metadata)
    endpoints_grid = _world_to_grid_continuous(endpoints, metadata)

    scan_indices = None
    scan_support = None
    last_scan_seen = None
    if bundle.scan_index is not None:
        scan_indices = np.asarray(bundle.scan_index, dtype=np.int64)
        if np.any(np.diff(scan_indices) < 0):
            raise ValueError("scan_index must be non-decreasing for scan support counting")
        scan_support = np.zeros(expected_shape, dtype=np.uint32)
        last_scan_seen = np.full(expected_shape, -1, dtype=np.int64)

    support = np.zeros(expected_shape, dtype=np.uint32)
    accepted_ray_count = 0
    traversed_cell_visits = 0
    supported_cell_visits = 0
    scan_supported_cell_visits = 0

    for index in range(bundle.ray_count):
        origin = origins[index]
        endpoint = endpoints[index]
        ray_range = float(np.linalg.norm(endpoint - origin))
        if ray_range + 1e-12 < float(config.min_ray_range_m):
            continue
        if (
            config.max_ray_range_m is not None
            and ray_range - 1e-12 > float(config.max_ray_range_m)
        ):
            continue

        p0 = origins_grid[index]
        p1 = endpoints_grid[index]
        clipped = _clip_segment_to_grid(
            p0, p1, int(metadata.width), int(metadata.height)
        )
        if clipped is None:
            continue
        accepted_ray_count += 1

        endpoint_inside = (
            0.0 <= p1[0] < float(metadata.width)
            and 0.0 <= p1[1] < float(metadata.height)
        )
        hit_cell = None
        if endpoint_inside:
            hit_cell = (int(np.floor(p1[0])), int(np.floor(p1[1])))

        scan_id = None if scan_indices is None else int(scan_indices[index])
        for x, y, cell_t0, cell_t1 in _traverse_clipped_cells(
            p0,
            p1,
            int(metadata.width),
            int(metadata.height),
            clipped[0],
            clipped[1],
        ):
            traversed_cell_visits += 1
            if hit_cell is not None and (x, y) == hit_cell:
                continue
            ground_z = float(ground[y, x])
            if not np.isfinite(ground_z):
                continue
            t_mid = 0.5 * (float(cell_t0) + float(cell_t1))
            ray_z = float(origin[2] + t_mid * (endpoint[2] - origin[2]))
            relative_height = ray_z - ground_z
            if (
                relative_height + 1e-12
                < float(config.min_ground_relative_height_m)
                or relative_height - 1e-12
                > float(config.max_ground_relative_height_m)
            ):
                continue
            support[y, x] += np.uint32(1)
            supported_cell_visits += 1
            if scan_id is not None and last_scan_seen[y, x] != scan_id:
                scan_support[y, x] += np.uint32(1)
                last_scan_seen[y, x] = scan_id
                scan_supported_cell_visits += 1

    support_mask = support >= int(config.min_support_rays)
    result = {
        "support_count": support,
        "support_mask": support_mask,
        "summary": {
            "schema_version": 1,
            "input_ray_count": bundle.ray_count,
            "accepted_ray_count": int(accepted_ray_count),
            "traversed_cell_visits": int(traversed_cell_visits),
            "supported_cell_visits": int(supported_cell_visits),
            "supported_cell_count": int(np.count_nonzero(support_mask)),
            "finite_ground_cell_count": int(np.count_nonzero(np.isfinite(ground))),
            "scan_support_available": scan_support is not None,
            "scan_supported_cell_visits": int(scan_supported_cell_visits),
            "semantic_promotion": False,
        },
    }
    if scan_support is not None:
        result["scan_support_count"] = scan_support
    return result
