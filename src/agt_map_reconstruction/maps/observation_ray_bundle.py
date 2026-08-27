"""Stable, ROS-independent input contract for trajectory-aware observation rays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 1
REQUIRED_ARRAY_KEYS = (
    "schema_version",
    "frame_id",
    "ray_origin_xyz_m",
    "ray_endpoint_xyz_m",
)


@dataclass(frozen=True)
class ObservationRayBundle:
    """One map-frame sensor-origin / first-return pair per LiDAR ray."""

    ray_origin_xyz_m: np.ndarray
    ray_endpoint_xyz_m: np.ndarray
    frame_id: str = "map"
    timestamp_s: np.ndarray | None = None
    scan_index: np.ndarray | None = None

    @property
    def ray_count(self):
        return int(self.ray_origin_xyz_m.shape[0])


def _as_xyz(name, values):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _optional_vector(name, values, count, dtype):
    if values is None:
        return None
    array = np.asarray(values, dtype=dtype).reshape(-1)
    if array.shape != (count,):
        raise ValueError(f"{name} must have shape (N,)")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def validate_observation_ray_bundle(
    ray_origin_xyz_m,
    ray_endpoint_xyz_m,
    frame_id="map",
    timestamp_s=None,
    scan_index=None,
):
    """Validate and normalize a ray bundle without changing its geometry."""
    origins = _as_xyz("ray_origin_xyz_m", ray_origin_xyz_m)
    endpoints = _as_xyz("ray_endpoint_xyz_m", ray_endpoint_xyz_m)
    if origins.shape != endpoints.shape:
        raise ValueError("ray origins and endpoints must have matching shapes")
    if origins.shape[0] == 0:
        raise ValueError("observation ray bundle must contain at least one ray")
    frame_id = str(frame_id)
    if not frame_id:
        raise ValueError("frame_id must be non-empty")

    timestamps = _optional_vector(
        "timestamp_s", timestamp_s, origins.shape[0], np.float64
    )
    if timestamps is not None and np.any(np.diff(timestamps) < 0.0):
        raise ValueError("timestamp_s must be non-decreasing")
    scans = _optional_vector("scan_index", scan_index, origins.shape[0], np.int64)
    if scans is not None and np.any(scans < 0):
        raise ValueError("scan_index must be non-negative")

    return ObservationRayBundle(
        ray_origin_xyz_m=origins,
        ray_endpoint_xyz_m=endpoints,
        frame_id=frame_id,
        timestamp_s=timestamps,
        scan_index=scans,
    )


def write_observation_ray_bundle(path, bundle):
    """Persist the stable NPZ contract with pickle disabled on read."""
    if not isinstance(bundle, ObservationRayBundle):
        raise TypeError("bundle must be an ObservationRayBundle")
    bundle = validate_observation_ray_bundle(
        bundle.ray_origin_xyz_m,
        bundle.ray_endpoint_xyz_m,
        frame_id=bundle.frame_id,
        timestamp_s=bundle.timestamp_s,
        scan_index=bundle.scan_index,
    )
    payload = {
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int16),
        "frame_id": np.asarray(bundle.frame_id),
        "ray_origin_xyz_m": bundle.ray_origin_xyz_m,
        "ray_endpoint_xyz_m": bundle.ray_endpoint_xyz_m,
    }
    if bundle.timestamp_s is not None:
        payload["timestamp_s"] = bundle.timestamp_s
    if bundle.scan_index is not None:
        payload["scan_index"] = bundle.scan_index

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path


def load_observation_ray_bundle(path, expected_frame_id="map"):
    """Load a schema-v1 bundle and reject ambiguous/object arrays."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        missing = [key for key in REQUIRED_ARRAY_KEYS if key not in payload]
        if missing:
            raise ValueError("ray bundle missing keys: " + ", ".join(missing))
        schema_version = int(np.asarray(payload["schema_version"]).reshape(()))
        if schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported ray bundle schema_version={schema_version}; "
                f"expected {SCHEMA_VERSION}"
            )
        frame_id = str(np.asarray(payload["frame_id"]).reshape(()))
        if expected_frame_id is not None and frame_id != str(expected_frame_id):
            raise ValueError(
                f"ray bundle frame_id={frame_id!r} does not match "
                f"expected {expected_frame_id!r}"
            )
        timestamp_s = payload["timestamp_s"] if "timestamp_s" in payload else None
        scan_index = payload["scan_index"] if "scan_index" in payload else None
        return validate_observation_ray_bundle(
            payload["ray_origin_xyz_m"],
            payload["ray_endpoint_xyz_m"],
            frame_id=frame_id,
            timestamp_s=timestamp_s,
            scan_index=scan_index,
        )
