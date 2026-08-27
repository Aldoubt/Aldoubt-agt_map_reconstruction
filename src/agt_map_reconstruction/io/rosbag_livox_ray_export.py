"""Rosbag2 adapter for trajectory-aware Livox observation-ray export.

The geometry is delegated to :mod:`livox_trajectory_rays`. ROS imports are lazy
so offline unit tests and metadata inspection remain usable without a sourced
ROS 2 environment.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from .livox_trajectory_rays import (
    build_livox_ray_chunk,
    select_livox_returns,
    validate_pose_trajectory,
)
from .rosbag_observation_inventory import inventory_rosbag2_metadata


def _metadata_payload(bag_dir):
    bag = Path(bag_dir).expanduser().resolve()
    path = bag / "metadata.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"rosbag2 metadata not found: {path}")
    return bag, yaml.safe_load(path.read_text(encoding="utf-8"))


def _topic_record(inventory, topic_name):
    matches = [item for item in inventory["topics"] if item["name"] == str(topic_name)]
    if len(matches) != 1:
        raise ValueError(f"topic {topic_name!r} not found exactly once in rosbag metadata")
    return matches[0]


def _unique_livox_custom_topic(inventory):
    matches = [
        item for item in inventory["topics"]
        if item["message_family"] == "livox_custom" and item["message_count"] > 0
    ]
    if len(matches) != 1:
        names = [item["name"] for item in matches]
        raise ValueError(
            "expected exactly one Livox CustomMsg topic; explicit --lidar-topic required "
            f"when ambiguous, candidates={names}"
        )
    return matches[0]["name"]


def _load_fast_livo_preprocess(path):
    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("FAST-LIVO config must be a YAML mapping")
    root = payload.get("/**", payload)
    if isinstance(root, dict) and "ros__parameters" in root:
        root = root["ros__parameters"]
    if not isinstance(root, dict):
        raise ValueError("FAST-LIVO config missing ros__parameters")
    preprocess = root.get("preprocess")
    if not isinstance(preprocess, dict):
        raise ValueError("FAST-LIVO config missing preprocess mapping")
    required = ("scan_line", "blind", "point_filter_num")
    missing = [key for key in required if key not in preprocess]
    if missing:
        raise ValueError("FAST-LIVO preprocess missing: " + ", ".join(missing))
    return {
        "source": str(config_path),
        "scan_line_count": int(preprocess["scan_line"]),
        "blind_range_m": float(preprocess["blind"]),
        "preprocess_point_filter_num": int(preprocess["point_filter_num"]),
    }


def resolve_benchmark_ray_export_contract(
    benchmark_run,
    *,
    fast_livo_config,
    lidar_bag=None,
    lidar_topic=None,
    trajectory_bag=None,
    trajectory_topic=None,
):
    """Resolve the selected full-bag benchmark run without choosing hidden sources."""
    run = Path(benchmark_run).expanduser().resolve()
    manifest_path = run / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"benchmark manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    dataset = manifest.get("dataset")
    algorithms = manifest.get("algorithms")
    calibration = manifest.get("calibration")
    if not isinstance(dataset, dict) or not isinstance(algorithms, dict):
        raise ValueError("benchmark manifest missing dataset/algorithms mappings")
    fast_livo = algorithms.get("fast_livo2")
    if not isinstance(fast_livo, dict):
        raise ValueError("benchmark manifest missing algorithms.fast_livo2")
    if not isinstance(calibration, dict):
        raise ValueError("benchmark manifest missing calibration mapping")

    manifest_bag = Path(str(dataset.get("bag_dir", ""))).expanduser().resolve()
    selected_lidar_bag = (
        manifest_bag if lidar_bag is None else Path(lidar_bag).expanduser().resolve()
    )
    if selected_lidar_bag != manifest_bag:
        raise ValueError(
            "selected lidar bag differs from benchmark manifest dataset.bag_dir; "
            "use the run that actually replayed this bag"
        )

    selected_trajectory_bag = (
        run / "raw" / "fast_livo2" / "fast_livo_trajectory"
        if trajectory_bag is None
        else Path(trajectory_bag).expanduser().resolve()
    )

    lidar_dir, lidar_metadata = _metadata_payload(selected_lidar_bag)
    lidar_inventory = inventory_rosbag2_metadata(lidar_metadata)
    selected_lidar_topic = (
        _unique_livox_custom_topic(lidar_inventory)
        if lidar_topic is None
        else str(lidar_topic)
    )
    lidar_record = _topic_record(lidar_inventory, selected_lidar_topic)
    if lidar_record["message_family"] != "livox_custom":
        raise ValueError("selected lidar topic is not a Livox CustomMsg source")

    trajectory_dir, trajectory_metadata = _metadata_payload(selected_trajectory_bag)
    trajectory_inventory = inventory_rosbag2_metadata(trajectory_metadata)
    configured_trajectory = (
        fast_livo.get("topics", {}).get("outputs", {}).get("trajectory")
        if isinstance(fast_livo.get("topics"), dict)
        else None
    )
    selected_trajectory_topic = str(
        trajectory_topic or configured_trajectory or "/aft_mapped_to_init"
    )
    trajectory_record = _topic_record(trajectory_inventory, selected_trajectory_topic)
    if trajectory_record["type"] != "nav_msgs/msg/Odometry":
        raise ValueError("selected FAST-LIVO2 trajectory topic must be nav_msgs/msg/Odometry")

    rotation = calibration.get("rotation_lidar_to_imu_row_major")
    translation = calibration.get("translation_lidar_to_imu_m")
    if not isinstance(rotation, list) or len(rotation) != 9:
        raise ValueError("benchmark calibration requires 9-value LiDAR-to-IMU rotation")
    if not isinstance(translation, list) or len(translation) != 3:
        raise ValueError("benchmark calibration requires 3-value LiDAR-to-IMU translation")

    trajectory_contract = fast_livo.get("trajectory_contract", {})
    if trajectory_contract:
        if trajectory_contract.get("pose_semantics") != "T_PARENT_TRACKED":
            raise ValueError("unsupported FAST-LIVO2 pose_semantics")
        if trajectory_contract.get("tracked_frame_physical") != "IMU_BODY":
            raise ValueError("FAST-LIVO2 trajectory must track the IMU/body frame")

    return {
        "schema_version": 1,
        "benchmark_run": str(run),
        "benchmark_manifest": str(manifest_path),
        "run_id": str(manifest.get("run_id", run.name)),
        "lidar": {
            "bag": str(lidar_dir),
            "topic": selected_lidar_topic,
            "type": lidar_record["type"],
            "message_count": int(lidar_record["message_count"]),
            "storage_identifier": lidar_inventory["storage_identifier"],
        },
        "trajectory": {
            "bag": str(trajectory_dir),
            "topic": selected_trajectory_topic,
            "type": trajectory_record["type"],
            "message_count": int(trajectory_record["message_count"]),
            "storage_identifier": trajectory_inventory["storage_identifier"],
            "declared_contract": trajectory_contract,
        },
        "extrinsic": {
            "convention": "LIDAR_TO_IMU",
            "rotation_lidar_to_imu_row_major": [float(value) for value in rotation],
            "translation_lidar_to_imu_m": [float(value) for value in translation],
        },
        "preprocess": _load_fast_livo_preprocess(fast_livo_config),
        "point_source_stage": "raw_custom_msg_pre_platform_self_filter",
        "platform_self_filter_reproduced": False,
        "automatic_source_selection": False,
        "semantic_promotion": False,
    }


def _ros_runtime():
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as exc:
        raise RuntimeError(
            "ROS 2 Python runtime is required for message export; source Humble and the "
            "workspace containing livox_ros_driver2 before running this command"
        ) from exc
    return rosbag2_py, deserialize_message, get_message


def _stamp_seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _open_reader(bag_dir, storage_identifier, topic_name):
    rosbag2_py, _, _ = _ros_runtime()
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id=str(storage_identifier)),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )
    try:
        reader.set_filter(rosbag2_py.StorageFilter(topics=[str(topic_name)]))
    except (AttributeError, TypeError):
        pass
    return reader


def load_odometry_pose_trajectory(contract):
    """Deserialize the selected /aft_mapped_to_init Odometry trajectory."""
    _, deserialize_message, get_message = _ros_runtime()
    source = contract["trajectory"]
    message_type = get_message(source["type"])
    reader = _open_reader(source["bag"], source["storage_identifier"], source["topic"])

    timestamps = []
    positions = []
    quaternions = []
    parent_frames = set()
    child_frames = set()
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic != source["topic"]:
            continue
        message = deserialize_message(data, message_type)
        timestamps.append(_stamp_seconds(message.header.stamp))
        parent_frames.add(str(message.header.frame_id))
        child_frames.add(str(message.child_frame_id))
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        positions.append([position.x, position.y, position.z])
        quaternions.append([orientation.x, orientation.y, orientation.z, orientation.w])

    if len(parent_frames) != 1 or len(child_frames) != 1:
        raise ValueError(
            "FAST-LIVO2 odometry must use one stable parent/child frame; "
            f"parents={sorted(parent_frames)}, children={sorted(child_frames)}"
        )
    if len(timestamps) != int(source["message_count"]):
        raise ValueError(
            f"trajectory read count {len(timestamps)} != metadata count {source['message_count']}"
        )
    return validate_pose_trajectory(
        timestamps,
        positions,
        quaternions,
        next(iter(parent_frames)),
        next(iter(child_frames)),
    )


def export_livox_observation_rays(
    contract,
    *,
    output_frame_id,
    allow_parent_frame_alias,
    max_pose_gap_s,
    export_point_stride,
    scan_stride=1,
    max_return_range_m=None,
    max_rays=None,
):
    """Export a sampled schema-v1 observation bundle and a detailed QA summary."""
    from agt_map_reconstruction.maps.observation_ray_bundle import (
        validate_observation_ray_bundle,
    )

    if int(scan_stride) < 1 or int(export_point_stride) < 1:
        raise ValueError("scan_stride/export_point_stride must be >= 1")
    if max_rays is not None and int(max_rays) < 1:
        raise ValueError("max_rays must be >= 1 when supplied")

    trajectory = load_odometry_pose_trajectory(contract)
    output_frame_id = str(output_frame_id)
    if output_frame_id != trajectory.parent_frame and not bool(allow_parent_frame_alias):
        raise ValueError(
            f"trajectory parent is {trajectory.parent_frame!r}, output frame is "
            f"{output_frame_id!r}; pass --allow-parent-frame-alias only when the P1 map "
            "uses the same numeric world gauge under a canonical frame name"
        )

    _, deserialize_message, get_message = _ros_runtime()
    source = contract["lidar"]
    message_type = get_message(source["type"])
    reader = _open_reader(source["bag"], source["storage_identifier"], source["topic"])
    preprocess = contract["preprocess"]
    rotation = np.asarray(
        contract["extrinsic"]["rotation_lidar_to_imu_row_major"], dtype=np.float64
    ).reshape(3, 3)
    translation = np.asarray(
        contract["extrinsic"]["translation_lidar_to_imu_m"], dtype=np.float64
    )

    origins = []
    endpoints = []
    timestamps_out = []
    scans_out = []
    scan_index = -1
    selected_scan_count = 0
    input_point_count = 0
    accepted_pre_stride = 0
    sampled_point_count = 0
    pose_supported_count = 0
    pose_before_count = 0
    pose_after_count = 0
    pose_gap_count = 0
    timebase_header_delta_s = []
    max_offset_time_ns = 0
    point_count_mismatch_scans = 0
    stopped_at_max_rays = False

    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic != source["topic"]:
            continue
        scan_index += 1
        message = deserialize_message(data, message_type)
        input_point_count += len(message.points)
        if int(message.point_num) != len(message.points):
            point_count_mismatch_scans += 1
        header_time_s = _stamp_seconds(message.header.stamp)
        timebase_s = float(message.timebase) * 1e-9
        timebase_header_delta_s.append(abs(header_time_s - timebase_s))
        if scan_index % int(scan_stride) != 0:
            continue
        selected_scan_count += 1

        point_count = len(message.points)
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
            max_offset_time_ns = max(max_offset_time_ns, int(np.max(offsets)))

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
        accepted_pre_stride += int(selection["accepted_before_export_stride"])
        sampled_point_count += int(selected.size)
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
        pose_before_count += int(np.count_nonzero(pose["before_trajectory_mask"]))
        pose_after_count += int(np.count_nonzero(pose["after_trajectory_mask"]))
        pose_gap_count += int(np.count_nonzero(pose["gap_rejected_mask"]))
        valid_count = int(chunk["ray_origin_xyz_m"].shape[0])
        if valid_count == 0:
            continue

        if max_rays is not None:
            remaining = int(max_rays) - pose_supported_count
            if remaining <= 0:
                stopped_at_max_rays = True
                break
            if valid_count > remaining:
                valid_count = remaining
                for key in ("ray_origin_xyz_m", "ray_endpoint_xyz_m", "timestamp_s"):
                    chunk[key] = chunk[key][:remaining]
                stopped_at_max_rays = True

        origins.append(chunk["ray_origin_xyz_m"])
        endpoints.append(chunk["ray_endpoint_xyz_m"])
        timestamps_out.append(chunk["timestamp_s"])
        scans_out.append(np.full((valid_count,), scan_index, dtype=np.int64))
        pose_supported_count += valid_count
        if stopped_at_max_rays:
            break

    if not origins:
        raise ValueError("no pose-supported Livox rays were exported")
    origin_array = np.concatenate(origins, axis=0)
    endpoint_array = np.concatenate(endpoints, axis=0)
    timestamp_array = np.concatenate(timestamps_out, axis=0)
    scan_array = np.concatenate(scans_out, axis=0)
    if np.any(np.diff(timestamp_array) < -1e-9):
        raise ValueError("exported point timestamps are not globally non-decreasing")
    bundle = validate_observation_ray_bundle(
        origin_array,
        endpoint_array,
        frame_id=output_frame_id,
        timestamp_s=timestamp_array,
        scan_index=scan_array,
    )

    pose_intervals = np.diff(trajectory.timestamp_s)
    deltas = np.asarray(timebase_header_delta_s, dtype=np.float64)
    summary = {
        "trajectory_pose_count": trajectory.pose_count,
        "trajectory_parent_frame": trajectory.parent_frame,
        "trajectory_child_frame": trajectory.child_frame,
        "trajectory_start_s": float(trajectory.timestamp_s[0]),
        "trajectory_end_s": float(trajectory.timestamp_s[-1]),
        "trajectory_interval_median_s": float(np.median(pose_intervals)),
        "trajectory_interval_p95_s": float(np.quantile(pose_intervals, 0.95)),
        "trajectory_interval_max_s": float(np.max(pose_intervals)),
        "lidar_metadata_scan_count": int(source["message_count"]),
        "lidar_scans_read": int(scan_index + 1),
        "selected_scan_count": int(selected_scan_count),
        "input_point_count": int(input_point_count),
        "accepted_before_export_stride": int(accepted_pre_stride),
        "sampled_point_count": int(sampled_point_count),
        "pose_supported_ray_count": int(pose_supported_count),
        "pose_rejected_before_trajectory": int(pose_before_count),
        "pose_rejected_after_trajectory": int(pose_after_count),
        "pose_rejected_large_gap": int(pose_gap_count),
        "max_offset_time_ns": int(max_offset_time_ns),
        "point_count_mismatch_scans": int(point_count_mismatch_scans),
        "header_timebase_abs_delta_median_s": float(np.median(deltas)) if deltas.size else None,
        "header_timebase_abs_delta_max_s": float(np.max(deltas)) if deltas.size else None,
        "output_first_timestamp_s": float(timestamp_array[0]),
        "output_last_timestamp_s": float(timestamp_array[-1]),
        "stopped_at_max_rays": bool(stopped_at_max_rays),
        "semantic_promotion": False,
    }
    return bundle, summary
