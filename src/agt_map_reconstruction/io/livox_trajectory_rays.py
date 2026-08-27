"""ROS-independent geometry core for Livox CustomMsg trajectory rays.

This module does not read rosbag2. It only validates pose trajectories, applies
FAST-LIVO-style Livox return filtering, interpolates IMU/body poses without
extrapolation, and transforms LiDAR first returns into a common world frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PoseTrajectory:
    timestamp_s: np.ndarray
    position_xyz_m: np.ndarray
    quaternion_xyzw: np.ndarray
    parent_frame: str
    child_frame: str

    @property
    def pose_count(self):
        return int(self.timestamp_s.shape[0])


def _normalize_quaternions(values):
    quaternions = np.asarray(values, dtype=np.float64)
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError("quaternion_xyzw must have shape (N, 4)")
    if not np.isfinite(quaternions).all():
        raise ValueError("quaternion_xyzw must contain only finite values")
    norm = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norm <= 1e-12):
        raise ValueError("quaternion_xyzw contains a zero-length quaternion")
    return quaternions / norm


def validate_pose_trajectory(
    timestamp_s,
    position_xyz_m,
    quaternion_xyzw,
    parent_frame,
    child_frame,
):
    timestamps = np.asarray(timestamp_s, dtype=np.float64).reshape(-1)
    positions = np.asarray(position_xyz_m, dtype=np.float64)
    quaternions = _normalize_quaternions(quaternion_xyzw)
    if timestamps.ndim != 1 or timestamps.shape[0] < 2:
        raise ValueError("pose trajectory requires at least two timestamps")
    if positions.shape != (timestamps.shape[0], 3):
        raise ValueError("position_xyz_m must have shape (N, 3)")
    if quaternions.shape[0] != timestamps.shape[0]:
        raise ValueError("pose arrays must have matching lengths")
    if not np.isfinite(timestamps).all() or not np.isfinite(positions).all():
        raise ValueError("pose trajectory must contain only finite values")
    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("pose timestamps must be strictly increasing")
    parent_frame = str(parent_frame)
    child_frame = str(child_frame)
    if not parent_frame or not child_frame:
        raise ValueError("trajectory parent/child frames must be non-empty")
    return PoseTrajectory(
        timestamp_s=timestamps,
        position_xyz_m=positions,
        quaternion_xyzw=quaternions,
        parent_frame=parent_frame,
        child_frame=child_frame,
    )


def _slerp_xyzw(q0, q1, alpha):
    q0 = _normalize_quaternions(q0)
    q1 = _normalize_quaternions(q1)
    alpha = np.asarray(alpha, dtype=np.float64).reshape(-1)
    if q0.shape != q1.shape or q0.shape[0] != alpha.shape[0]:
        raise ValueError("slerp inputs must have matching lengths")

    dot = np.sum(q0 * q1, axis=1)
    negative = dot < 0.0
    q1 = q1.copy()
    q1[negative] *= -1.0
    dot = np.clip(np.abs(dot), 0.0, 1.0)

    result = np.empty_like(q0)
    linear = dot > 0.9995
    if np.any(linear):
        local = q0[linear] + alpha[linear, None] * (q1[linear] - q0[linear])
        result[linear] = _normalize_quaternions(local)
    if np.any(~linear):
        theta = np.arccos(dot[~linear])
        sin_theta = np.sin(theta)
        local_alpha = alpha[~linear]
        weight0 = np.sin((1.0 - local_alpha) * theta) / sin_theta
        weight1 = np.sin(local_alpha * theta) / sin_theta
        local = weight0[:, None] * q0[~linear] + weight1[:, None] * q1[~linear]
        result[~linear] = _normalize_quaternions(local)
    return result


def interpolate_pose_trajectory(trajectory, query_timestamp_s, max_pose_gap_s):
    """Interpolate T_parent_child at query times, never extrapolating.

    Position is linearly interpolated and orientation uses shortest-arc SLERP.
    A query is valid only when it lies inside the trajectory time span and the
    bracketing pose interval is no larger than ``max_pose_gap_s``.
    """
    if not isinstance(trajectory, PoseTrajectory):
        raise TypeError("trajectory must be a PoseTrajectory")
    if not np.isfinite(float(max_pose_gap_s)) or float(max_pose_gap_s) <= 0.0:
        raise ValueError("max_pose_gap_s must be finite and > 0")

    query = np.asarray(query_timestamp_s, dtype=np.float64).reshape(-1)
    if not np.isfinite(query).all():
        raise ValueError("query timestamps must be finite")
    times = trajectory.timestamp_s
    inside = (query >= times[0]) & (query <= times[-1])

    right = np.searchsorted(times, query, side="right")
    right = np.clip(right, 1, times.shape[0] - 1)
    left = right - 1
    interval = times[right] - times[left]
    alpha = (query - times[left]) / interval
    valid = inside & (interval <= float(max_pose_gap_s) + 1e-12)

    positions = (
        trajectory.position_xyz_m[left]
        + alpha[:, None]
        * (trajectory.position_xyz_m[right] - trajectory.position_xyz_m[left])
    )
    quaternions = _slerp_xyzw(
        trajectory.quaternion_xyzw[left],
        trajectory.quaternion_xyzw[right],
        alpha,
    )
    return {
        "position_xyz_m": positions,
        "quaternion_xyzw": quaternions,
        "valid_mask": valid,
        "bracketing_interval_s": interval,
        "before_trajectory_mask": query < times[0],
        "after_trajectory_mask": query > times[-1],
        "gap_rejected_mask": inside & (interval > float(max_pose_gap_s) + 1e-12),
    }


def rotate_vectors_xyzw(quaternion_xyzw, vectors_xyz):
    quaternions = _normalize_quaternions(quaternion_xyzw)
    vectors = np.asarray(vectors_xyz, dtype=np.float64)
    if vectors.shape != (quaternions.shape[0], 3):
        raise ValueError("vectors_xyz must have shape (N, 3)")
    q_vec = quaternions[:, :3]
    q_w = quaternions[:, 3:4]
    twice_cross = 2.0 * np.cross(q_vec, vectors)
    return vectors + q_w * twice_cross + np.cross(q_vec, twice_cross)


def select_livox_returns(
    points_xyz_m,
    offset_time_ns,
    tags,
    lines,
    *,
    scan_line_count,
    blind_range_m,
    preprocess_point_filter_num=1,
    export_point_stride=1,
    max_return_range_m=None,
):
    """Apply the conservative FAST-LIVO CustomMsg return contract.

    The quality class check mirrors the customary Livox handler:
    ``(tag & 0x30) in {0x00, 0x10}``. Index 0 is skipped, matching the handler's
    loop convention. ``export_point_stride`` is an additional deterministic
    evidence-sampling stride and is not a FAST-LIVO parameter.
    """
    points = np.asarray(points_xyz_m, dtype=np.float64)
    offsets = np.asarray(offset_time_ns, dtype=np.int64).reshape(-1)
    tags = np.asarray(tags, dtype=np.uint8).reshape(-1)
    lines = np.asarray(lines, dtype=np.int64).reshape(-1)
    count = points.shape[0]
    if points.shape != (count, 3):
        raise ValueError("points_xyz_m must have shape (N, 3)")
    if offsets.shape != (count,) or tags.shape != (count,) or lines.shape != (count,):
        raise ValueError("Livox point attribute arrays must have matching lengths")
    if int(scan_line_count) < 1:
        raise ValueError("scan_line_count must be >= 1")
    if not np.isfinite(float(blind_range_m)) or float(blind_range_m) < 0.0:
        raise ValueError("blind_range_m must be finite and >= 0")
    if int(preprocess_point_filter_num) < 1 or int(export_point_stride) < 1:
        raise ValueError("point filter/stride values must be >= 1")
    if max_return_range_m is not None:
        if not np.isfinite(float(max_return_range_m)) or float(max_return_range_m) <= float(blind_range_m):
            raise ValueError("max_return_range_m must exceed blind_range_m")

    indices = np.arange(1, count, dtype=np.int64)
    if int(preprocess_point_filter_num) > 1:
        indices = indices[indices % int(preprocess_point_filter_num) == 0]
    if indices.size == 0:
        return indices, {"candidate_count": 0, "accepted_before_export_stride": 0}

    local_points = points[indices]
    finite = np.isfinite(local_points).all(axis=1)
    line_ok = (lines[indices] >= 0) & (lines[indices] < int(scan_line_count))
    tag_class = tags[indices] & np.uint8(0x30)
    tag_ok = (tag_class == np.uint8(0x00)) | (tag_class == np.uint8(0x10))
    range_m = np.linalg.norm(local_points, axis=1)
    range_ok = range_m > float(blind_range_m) + 1e-12
    if max_return_range_m is not None:
        range_ok &= range_m <= float(max_return_range_m) + 1e-12
    offset_ok = offsets[indices] >= 0
    keep = finite & line_ok & tag_ok & range_ok & offset_ok
    accepted = indices[keep]
    accepted_before_stride = int(accepted.size)
    accepted = accepted[:: int(export_point_stride)]
    return accepted, {
        "candidate_count": int(indices.size),
        "accepted_before_export_stride": accepted_before_stride,
        "accepted_after_export_stride": int(accepted.size),
        "removed_nonfinite": int(np.count_nonzero(~finite)),
        "removed_line": int(np.count_nonzero(finite & ~line_ok)),
        "removed_tag": int(np.count_nonzero(finite & line_ok & ~tag_ok)),
        "removed_range": int(np.count_nonzero(finite & line_ok & tag_ok & ~range_ok)),
    }


def build_livox_ray_chunk(
    points_xyz_lidar_m,
    point_timestamp_s,
    trajectory,
    rotation_lidar_to_imu,
    translation_lidar_to_imu_m,
    *,
    max_pose_gap_s,
):
    """Transform LiDAR first returns into the trajectory parent frame.

    ``rotation_lidar_to_imu`` and ``translation_lidar_to_imu_m`` define
    ``T_imu_lidar``: ``p_imu = R_imu_lidar @ p_lidar + t_imu_lidar``.
    The trajectory is ``T_parent_imu``. Each return uses the interpolated pose at
    its own point timestamp, so origin and endpoint share the same physical time.
    """
    points = np.asarray(points_xyz_lidar_m, dtype=np.float64)
    timestamps = np.asarray(point_timestamp_s, dtype=np.float64).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] != timestamps.shape[0]:
        raise ValueError("points/timestamps must have shapes (N,3) and (N,)")
    if not np.isfinite(points).all() or not np.isfinite(timestamps).all():
        raise ValueError("ray input points/timestamps must be finite")

    rotation = np.asarray(rotation_lidar_to_imu, dtype=np.float64)
    translation = np.asarray(translation_lidar_to_imu_m, dtype=np.float64).reshape(-1)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("LiDAR-to-IMU extrinsic must be a 3x3 rotation and 3-vector")
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ValueError("LiDAR-to-IMU extrinsic must be finite")
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError("rotation_lidar_to_imu must be a proper rotation matrix")

    pose = interpolate_pose_trajectory(trajectory, timestamps, max_pose_gap_s)
    valid = pose["valid_mask"]
    if not np.any(valid):
        return {
            "ray_origin_xyz_m": np.empty((0, 3), dtype=np.float64),
            "ray_endpoint_xyz_m": np.empty((0, 3), dtype=np.float64),
            "timestamp_s": np.empty((0,), dtype=np.float64),
            "valid_mask": valid,
            "pose": pose,
        }

    local_points_imu = points @ rotation.T + translation[None, :]
    local_origins_imu = np.broadcast_to(translation, points.shape)
    q = pose["quaternion_xyzw"][valid]
    p = pose["position_xyz_m"][valid]
    origins = rotate_vectors_xyzw(q, local_origins_imu[valid]) + p
    endpoints = rotate_vectors_xyzw(q, local_points_imu[valid]) + p
    return {
        "ray_origin_xyz_m": origins,
        "ray_endpoint_xyz_m": endpoints,
        "timestamp_s": timestamps[valid],
        "valid_mask": valid,
        "pose": pose,
    }
