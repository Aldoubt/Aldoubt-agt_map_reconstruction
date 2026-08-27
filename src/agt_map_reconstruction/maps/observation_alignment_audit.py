"""Audit whether exported trajectory-aware rays share the frozen P1 map gauge.

This module is diagnostic only. It does not transform rays, change the map, or
promote any semantic class. Navigation PGM images are flipped back to the
repository's lower-left-origin (y, x) grid convention before cell lookup.
"""

from __future__ import annotations

from pathlib import Path
import math

import cv2
import numpy as np
import yaml


OCCUPIED_VALUE = 0
UNKNOWN_VALUE = 205
FREE_VALUE = 254


def load_navigation_grid(map_yaml_path):
    path = Path(map_yaml_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"navigation map YAML not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("navigation map YAML must be a mapping")

    image_value = payload.get("image")
    resolution = payload.get("resolution")
    origin = payload.get("origin")
    if not isinstance(image_value, str) or not image_value:
        raise ValueError("navigation map YAML missing image")
    if resolution is None or float(resolution) <= 0.0:
        raise ValueError("navigation map YAML resolution must be > 0")
    if not isinstance(origin, list) or len(origin) != 3:
        raise ValueError("navigation map YAML origin must contain [x, y, yaw]")

    image_path = Path(image_value).expanduser()
    if not image_path.is_absolute():
        image_path = (path.parent / image_path).resolve()
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"navigation map image not readable: {image_path}")
    if image.ndim == 3:
        image = image[..., 0]
    if image.ndim != 2:
        raise ValueError("navigation map image must be 2D")

    # Map-server image rows are top-down; repository arrays are lower-left-origin.
    grid = np.flipud(np.asarray(image, dtype=np.uint8))
    return {
        "yaml_path": str(path),
        "image_path": str(image_path),
        "grid": grid,
        "resolution": float(resolution),
        "origin": [float(value) for value in origin],
        "width": int(grid.shape[1]),
        "height": int(grid.shape[0]),
    }


def _world_to_grid(points_xy, *, resolution, origin):
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_xy must have shape (N, 2)")
    origin_x, origin_y, yaw = [float(value) for value in origin]
    dx = points[:, 0] - origin_x
    dy = points[:, 1] - origin_y
    c = math.cos(yaw)
    s = math.sin(yaw)
    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy
    gx = np.floor(local_x / float(resolution)).astype(np.int64)
    gy = np.floor(local_y / float(resolution)).astype(np.int64)
    return gx, gy


def _cell_stats(grid, gx, gy):
    height, width = grid.shape
    in_bounds = (gx >= 0) & (gy >= 0) & (gx < width) & (gy < height)
    values = np.full(gx.shape, -1, dtype=np.int16)
    if np.any(in_bounds):
        values[in_bounds] = grid[gy[in_bounds], gx[in_bounds]].astype(np.int16)

    in_count = int(np.count_nonzero(in_bounds))
    total = int(gx.size)
    classes = {}
    for name, value in (
        ("occupied", OCCUPIED_VALUE),
        ("unknown", UNKNOWN_VALUE),
        ("free", FREE_VALUE),
    ):
        count = int(np.count_nonzero(values == value))
        classes[name] = {
            "count": count,
            "fraction_of_all": (count / total) if total else 0.0,
            "fraction_of_in_bounds": (count / in_count) if in_count else 0.0,
        }
    other_count = int(np.count_nonzero(in_bounds & ~np.isin(values, [0, 205, 254])))
    classes["other"] = {
        "count": other_count,
        "fraction_of_all": (other_count / total) if total else 0.0,
        "fraction_of_in_bounds": (other_count / in_count) if in_count else 0.0,
    }
    return {
        "in_bounds_count": in_count,
        "out_of_bounds_count": total - in_count,
        "in_bounds_fraction": (in_count / total) if total else 0.0,
        "classes": classes,
    }


def _xyz_extent(points):
    array = np.asarray(points, dtype=np.float64)
    return {
        "min": [float(value) for value in np.min(array, axis=0)],
        "max": [float(value) for value in np.max(array, axis=0)],
    }


def _map_world_corners(width, height, resolution, origin):
    origin_x, origin_y, yaw = [float(value) for value in origin]
    local = np.asarray(
        [
            [0.0, 0.0],
            [width * resolution, 0.0],
            [width * resolution, height * resolution],
            [0.0, height * resolution],
        ],
        dtype=np.float64,
    )
    c = math.cos(yaw)
    s = math.sin(yaw)
    world = np.empty_like(local)
    world[:, 0] = origin_x + c * local[:, 0] - s * local[:, 1]
    world[:, 1] = origin_y + s * local[:, 0] + c * local[:, 1]
    return world


def audit_observation_ray_alignment(bundle, navigation_grid):
    """Return map-overlap diagnostics for a validated ObservationRayBundle."""
    origins = np.asarray(bundle.ray_origin_xyz_m, dtype=np.float64)
    endpoints = np.asarray(bundle.ray_endpoint_xyz_m, dtype=np.float64)
    if origins.shape != endpoints.shape or origins.ndim != 2 or origins.shape[1] != 3:
        raise ValueError("ray bundle origins/endpoints must have matching shape (N, 3)")
    if origins.shape[0] == 0:
        raise ValueError("ray bundle must contain at least one ray")

    grid = np.asarray(navigation_grid["grid"], dtype=np.uint8)
    resolution = float(navigation_grid["resolution"])
    origin = navigation_grid["origin"]
    endpoint_gx, endpoint_gy = _world_to_grid(
        endpoints[:, :2], resolution=resolution, origin=origin
    )
    origin_gx, origin_gy = _world_to_grid(
        origins[:, :2], resolution=resolution, origin=origin
    )

    ray_lengths = np.linalg.norm(endpoints - origins, axis=1)
    corners = _map_world_corners(
        navigation_grid["width"],
        navigation_grid["height"],
        resolution,
        origin,
    )
    return {
        "schema_version": 1,
        "ray_count": int(origins.shape[0]),
        "ray_frame_id": str(bundle.frame_id),
        "map": {
            "yaml_path": navigation_grid["yaml_path"],
            "image_path": navigation_grid["image_path"],
            "resolution_m": resolution,
            "width": int(navigation_grid["width"]),
            "height": int(navigation_grid["height"]),
            "origin": [float(value) for value in origin],
            "world_corners_xy_m": corners.tolist(),
            "world_aabb_xy_m": {
                "min": [float(value) for value in np.min(corners, axis=0)],
                "max": [float(value) for value in np.max(corners, axis=0)],
            },
        },
        "origins": {
            **_cell_stats(grid, origin_gx, origin_gy),
            "xyz_extent_m": _xyz_extent(origins),
        },
        "endpoints": {
            **_cell_stats(grid, endpoint_gx, endpoint_gy),
            "xyz_extent_m": _xyz_extent(endpoints),
        },
        "ray_length_m": {
            "median": float(np.median(ray_lengths)),
            "p95": float(np.quantile(ray_lengths, 0.95)),
            "max": float(np.max(ray_lengths)),
        },
        "automatic_alignment_acceptance": False,
        "semantic_promotion": False,
    }
