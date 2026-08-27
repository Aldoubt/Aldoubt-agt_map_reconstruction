import numpy as np
import pytest

from agt_map_reconstruction.io.livox_trajectory_rays import (
    build_livox_ray_chunk,
    interpolate_pose_trajectory,
    select_livox_returns,
    validate_pose_trajectory,
)


def _trajectory():
    return validate_pose_trajectory(
        [10.0, 11.0],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, np.sin(np.pi / 4.0), np.cos(np.pi / 4.0)],
        ],
        "camera_init",
        "aft_mapped",
    )


def test_pose_interpolation_uses_slerp_and_never_extrapolates():
    trajectory = _trajectory()
    result = interpolate_pose_trajectory(
        trajectory,
        [9.9, 10.5, 11.1],
        max_pose_gap_s=1.1,
    )

    assert result["valid_mask"].tolist() == [False, True, False]
    np.testing.assert_allclose(result["position_xyz_m"][1], [0.5, 0.0, 0.0])

    q = result["quaternion_xyzw"][1]
    # Halfway between 0 and 90 deg around +Z is 45 deg.
    np.testing.assert_allclose(
        q,
        [0.0, 0.0, np.sin(np.pi / 8.0), np.cos(np.pi / 8.0)],
        atol=1e-7,
    )
    assert result["before_trajectory_mask"].tolist() == [True, False, False]
    assert result["after_trajectory_mask"].tolist() == [False, False, True]


def test_pose_interpolation_rejects_large_bracketing_gap():
    trajectory = _trajectory()
    result = interpolate_pose_trajectory(
        trajectory,
        [10.5],
        max_pose_gap_s=0.2,
    )
    assert not result["valid_mask"][0]
    assert result["gap_rejected_mask"][0]


def test_livox_filter_matches_tag_line_blind_and_export_stride_contract():
    points = np.array(
        [
            [10.0, 0.0, 0.0],  # index 0 intentionally skipped
            [1.0, 0.0, 0.0],   # valid
            [2.0, 0.0, 0.0],   # bad tag
            [3.0, 0.0, 0.0],   # bad line
            [0.1, 0.0, 0.0],   # inside blind range
            [5.0, 0.0, 0.0],   # valid
            [6.0, 0.0, 0.0],   # valid
        ]
    )
    offsets = np.arange(points.shape[0], dtype=np.int64) * 100
    tags = np.array([0x10, 0x10, 0x30, 0x10, 0x00, 0x00, 0x10], dtype=np.uint8)
    lines = np.array([0, 0, 0, 4, 0, 1, 2])

    indices, summary = select_livox_returns(
        points,
        offsets,
        tags,
        lines,
        scan_line_count=4,
        blind_range_m=0.5,
        preprocess_point_filter_num=1,
        export_point_stride=2,
    )

    # Accepted before export stride: indices 1, 5, 6. Every second accepted point -> 1, 6.
    np.testing.assert_array_equal(indices, [1, 6])
    assert summary["accepted_before_export_stride"] == 3
    assert summary["accepted_after_export_stride"] == 2


def test_ray_chunk_applies_lidar_to_imu_extrinsic_and_per_point_pose():
    trajectory = validate_pose_trajectory(
        [0.0, 1.0],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
        "camera_init",
        "aft_mapped",
    )
    points_lidar = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    timestamps = np.array([0.25, 0.75])
    result = build_livox_ray_chunk(
        points_lidar,
        timestamps,
        trajectory,
        np.eye(3),
        [0.1, 0.2, 0.3],
        max_pose_gap_s=1.1,
    )

    np.testing.assert_allclose(
        result["ray_origin_xyz_m"],
        [[0.35, 0.2, 0.3], [0.85, 0.2, 0.3]],
    )
    np.testing.assert_allclose(
        result["ray_endpoint_xyz_m"],
        [[1.35, 0.2, 0.3], [1.85, 0.2, 0.3]],
    )
    np.testing.assert_allclose(result["timestamp_s"], timestamps)


def test_invalid_rotation_is_rejected():
    with pytest.raises(ValueError, match="proper rotation"):
        build_livox_ray_chunk(
            [[1.0, 0.0, 0.0]],
            [10.5],
            _trajectory(),
            np.diag([1.0, 1.0, 2.0]),
            [0.0, 0.0, 0.0],
            max_pose_gap_s=1.1,
        )
