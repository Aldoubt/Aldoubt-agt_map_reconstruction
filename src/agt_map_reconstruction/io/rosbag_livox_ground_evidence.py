"""Stream Livox CustomMsg observations directly into ground-aware grid support.

Unlike the NPZ exporter used for QA, this adapter keeps only bounded ray batches
in memory and accumulates global support-count grids. It preserves the physical
scan index so downstream evidence can distinguish raw ray density from unique
scan support. It never writes a full ray bundle and never modifies semantic or
navigation-map cells.
"""

from __future__ import annotations

import numpy as np

from .livox_trajectory_rays import build_livox_ray_chunk, select_livox_returns
from .rosbag_livox_ray_export import (
    _open_reader,
    _ros_runtime,
    _stamp_seconds,
    load_odometry_pose_trajectory,
)
from agt_map_reconstruction.maps.ground_aware_ray_stream import (
    accumulate_ground_aware_ray_batches,
)
from agt_map_reconstruction.maps.observation_ray_bundle import (
    validate_observation_ray_bundle,
)


def stream_livox_ground_aware_evidence(
    contract,
    ground_surface,
    metadata,
    config,
    *,
    output_frame_id,
    allow_parent_frame_alias,
    max_pose_gap_s,
    export_point_stride,
    scan_stride=1,
    max_return_range_m=None,
    max_rays=None,
    batch_ray_limit=250000,
):
    """Accumulate map-frame ray and unique-scan support without a full ray NPZ."""
    if int(scan_stride) < 1 or int(export_point_stride) < 1:
        raise ValueError("scan_stride/export_point_stride must be >= 1")
    if int(batch_ray_limit) < 1:
        raise ValueError("batch_ray_limit must be >= 1")
    if max_rays is not None and int(max_rays) < 1:
        raise ValueError("max_rays must be >= 1 when supplied")

    trajectory = load_odometry_pose_trajectory(contract)
    output_frame_id = str(output_frame_id)
    if output_frame_id != str(metadata.frame_id):
        raise ValueError(
            f"output frame {output_frame_id!r} does not match grid frame "
            f"{metadata.frame_id!r}"
        )
    if output_frame_id != trajectory.parent_frame and not bool(allow_parent_frame_alias):
        raise ValueError(
            f"trajectory parent is {trajectory.parent_frame!r}, output frame is "
            f"{output_frame_id!r}; explicit parent-frame alias approval is required"
        )

    _, deserialize_message, get_message = _ros_runtime()
    source = contract["lidar"]
    message_type = get_message(source["type"])
    preprocess = contract["preprocess"]
    rotation = np.asarray(
        contract["extrinsic"]["rotation_lidar_to_imu_row_major"], dtype=np.float64
    ).reshape(3, 3)
    translation = np.asarray(
        contract["extrinsic"]["translation_lidar_to_imu_m"], dtype=np.float64
    )

    stats = {
        "lidar_metadata_scan_count": int(source["message_count"]),
        "lidar_scans_read": 0,
        "selected_scan_count": 0,
        "input_point_count": 0,
        "accepted_before_export_stride": 0,
        "sampled_point_count": 0,
        "pose_supported_ray_count": 0,
        "pose_rejected_before_trajectory": 0,
        "pose_rejected_after_trajectory": 0,
        "pose_rejected_large_gap": 0,
        "max_offset_time_ns": 0,
        "point_count_mismatch_scans": 0,
        "header_timebase_abs_delta_max_s": 0.0,
        "stopped_at_max_rays": False,
    }

    def batches():
        reader = _open_reader(source["bag"], source["storage_identifier"], source["topic"])
        scan_index = -1
        origins = []
        endpoints = []
        scan_ids = []
        buffered = 0

        def flush():
            nonlocal origins, endpoints, scan_ids, buffered
            if buffered == 0:
                return None
            bundle = validate_observation_ray_bundle(
                np.concatenate(origins, axis=0),
                np.concatenate(endpoints, axis=0),
                frame_id=output_frame_id,
                scan_index=np.concatenate(scan_ids, axis=0),
            )
            origins = []
            endpoints = []
            scan_ids = []
            buffered = 0
            return bundle

        while reader.has_next():
            topic, data, _ = reader.read_next()
            if topic != source["topic"]:
                continue
            scan_index += 1
            stats["lidar_scans_read"] = scan_index + 1
            message = deserialize_message(data, message_type)
            point_count = len(message.points)
            stats["input_point_count"] += point_count
            if int(message.point_num) != point_count:
                stats["point_count_mismatch_scans"] += 1
            header_time_s = _stamp_seconds(message.header.stamp)
            timebase_s = float(message.timebase) * 1e-9
            stats["header_timebase_abs_delta_max_s"] = max(
                float(stats["header_timebase_abs_delta_max_s"]),
                abs(header_time_s - timebase_s),
            )
            if scan_index % int(scan_stride) != 0:
                continue
            stats["selected_scan_count"] += 1

            points = np.empty((point_count, 3), dtype=np.float64)
            offsets = np.empty((point_count,), dtype=np.int64)
            tags = np.empty((point_count,), dtype=np.uint8)
            lines = np.empty((point_count,), dtype=np.int64)
            for index, point in enumerate(message.points):
                points[index] = (point.x, point.y, point.z)
                offsets[index] = int(point.offset_time)
                tags[index] = int(point.tag)
                lines[index] = int(point.line)
            if offsets.size:
                stats["max_offset_time_ns"] = max(
                    int(stats["max_offset_time_ns"]), int(np.max(offsets))
                )

            selected, selection = select_livox_returns(
                points,
                offsets,
                tags,
                lines,
                scan_line_count=preprocess["scan_line_count"],
                blind_range_m=preprocess["blind_range_m"],
                preprocess_point_filter_num=preprocess["preprocess_point_filter_num"],
                export_point_stride=export_point_stride,
                max_return_range_m=max_return_range_m,
            )
            stats["accepted_before_export_stride"] += int(
                selection["accepted_before_export_stride"]
            )
            stats["sampled_point_count"] += int(selected.size)
            if selected.size == 0:
                continue

            order = np.argsort(offsets[selected], kind="stable")
            selected = selected[order]
            point_timestamps = (
                int(message.timebase) + offsets[selected].astype(np.int64)
            ).astype(np.float64) * 1e-9
            chunk = build_livox_ray_chunk(
                points[selected],
                point_timestamps,
                trajectory,
                rotation,
                translation,
                max_pose_gap_s=max_pose_gap_s,
            )
            pose = chunk["pose"]
            stats["pose_rejected_before_trajectory"] += int(
                np.count_nonzero(pose["before_trajectory_mask"])
            )
            stats["pose_rejected_after_trajectory"] += int(
                np.count_nonzero(pose["after_trajectory_mask"])
            )
            stats["pose_rejected_large_gap"] += int(
                np.count_nonzero(pose["gap_rejected_mask"])
            )

            chunk_origins = chunk["ray_origin_xyz_m"]
            chunk_endpoints = chunk["ray_endpoint_xyz_m"]
            valid_count = int(chunk_origins.shape[0])
            if valid_count == 0:
                continue

            if max_rays is not None:
                remaining = int(max_rays) - int(stats["pose_supported_ray_count"])
                if remaining <= 0:
                    stats["stopped_at_max_rays"] = True
                    break
                if valid_count > remaining:
                    chunk_origins = chunk_origins[:remaining]
                    chunk_endpoints = chunk_endpoints[:remaining]
                    valid_count = remaining
                    stats["stopped_at_max_rays"] = True

            origins.append(chunk_origins)
            endpoints.append(chunk_endpoints)
            scan_ids.append(np.full((valid_count,), scan_index, dtype=np.int64))
            buffered += valid_count
            stats["pose_supported_ray_count"] += valid_count

            # Flush only after the complete physical scan has been appended. This
            # guarantees one scan never appears in two batches, so scan-support
            # counts can be summed across batches without double counting.
            if buffered >= int(batch_ray_limit):
                bundle = flush()
                if bundle is not None:
                    yield bundle
            if stats["stopped_at_max_rays"]:
                break

        bundle = flush()
        if bundle is not None:
            yield bundle

    result = accumulate_ground_aware_ray_batches(
        batches(),
        ground_surface,
        metadata,
        config,
    )
    result["summary"].update(stats)
    result["summary"].update(
        {
            "trajectory_pose_count": int(trajectory.pose_count),
            "trajectory_parent_frame": trajectory.parent_frame,
            "trajectory_child_frame": trajectory.child_frame,
            "output_frame_id": output_frame_id,
            "parent_frame_alias_without_transform": bool(
                output_frame_id != trajectory.parent_frame
            ),
            "scan_stride": int(scan_stride),
            "export_point_stride": int(export_point_stride),
            "batch_ray_limit": int(batch_ray_limit),
            "max_pose_gap_s": float(max_pose_gap_s),
            "platform_self_filter_reproduced": False,
            "semantic_promotion": False,
        }
    )
    return result
