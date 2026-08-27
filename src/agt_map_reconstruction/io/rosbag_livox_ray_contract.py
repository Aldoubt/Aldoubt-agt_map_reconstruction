"""Resolve P1-E1 Livox/FAST-LIVO2 replay provenance without hidden defaults.

The benchmark run freezes the raw bag and FAST-LIVO2 trajectory contract, but
some newer benchmark manifests no longer duplicate numeric LiDAR-to-IMU
calibration at the top level. In that case the exact replay YAML supplied by the
caller is the authoritative numeric extrinsic source.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

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
        item
        for item in inventory["topics"]
        if item["message_family"] == "livox_custom" and item["message_count"] > 0
    ]
    if len(matches) != 1:
        names = [item["name"] for item in matches]
        raise ValueError(
            "expected exactly one Livox CustomMsg topic; explicit --lidar-topic required "
            f"when ambiguous, candidates={names}"
        )
    return matches[0]["name"]


def _fast_livo_parameter_root(path):
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"FAST-LIVO config not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("FAST-LIVO config must be a YAML mapping")
    root = payload.get("/**", payload)
    if isinstance(root, dict) and "ros__parameters" in root:
        root = root["ros__parameters"]
    if not isinstance(root, dict):
        raise ValueError("FAST-LIVO config missing ros__parameters")
    return config_path, root


def _load_fast_livo_preprocess(path):
    config_path, root = _fast_livo_parameter_root(path)
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


def _load_fast_livo_extrinsic(path):
    config_path, root = _fast_livo_parameter_root(path)
    extrinsic = root.get("extrin_calib")
    if not isinstance(extrinsic, dict):
        raise ValueError(
            "benchmark manifest has no numeric calibration and FAST-LIVO config "
            "is missing extrin_calib"
        )
    rotation = extrinsic.get("extrinsic_R")
    translation = extrinsic.get("extrinsic_T")
    if not isinstance(rotation, list) or len(rotation) != 9:
        raise ValueError("FAST-LIVO extrin_calib.extrinsic_R must contain 9 values")
    if not isinstance(translation, list) or len(translation) != 3:
        raise ValueError("FAST-LIVO extrin_calib.extrinsic_T must contain 3 values")
    return {
        "convention": "LIDAR_TO_IMU",
        "rotation_lidar_to_imu_row_major": [float(value) for value in rotation],
        "translation_lidar_to_imu_m": [float(value) for value in translation],
        "source": str(config_path),
        "source_field": "ros__parameters.extrin_calib",
    }


def _resolve_extrinsic(manifest, fast_livo, fast_livo_config):
    declared_convention = fast_livo.get("extrinsic_convention")
    if declared_convention not in (None, "", "LIDAR_TO_IMU"):
        raise ValueError(
            "unsupported FAST-LIVO2 extrinsic_convention="
            f"{declared_convention!r}; expected LIDAR_TO_IMU"
        )

    calibration = manifest.get("calibration")
    if isinstance(calibration, dict):
        rotation = calibration.get("rotation_lidar_to_imu_row_major")
        translation = calibration.get("translation_lidar_to_imu_m")
        if isinstance(rotation, list) and len(rotation) == 9 and isinstance(translation, list) and len(translation) == 3:
            return {
                "convention": "LIDAR_TO_IMU",
                "rotation_lidar_to_imu_row_major": [float(value) for value in rotation],
                "translation_lidar_to_imu_m": [float(value) for value in translation],
                "source": "benchmark_manifest",
                "source_field": "calibration",
            }
        raise ValueError(
            "benchmark calibration mapping exists but does not contain a valid "
            "9-value rotation and 3-value translation"
        )

    return _load_fast_livo_extrinsic(fast_livo_config)


def resolve_benchmark_ray_export_contract(
    benchmark_run,
    *,
    fast_livo_config,
    lidar_bag=None,
    lidar_topic=None,
    trajectory_bag=None,
    trajectory_topic=None,
):
    """Resolve the selected full-bag replay and its explicit numeric extrinsic."""
    run = Path(benchmark_run).expanduser().resolve()
    manifest_path = run / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"benchmark manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    dataset = manifest.get("dataset")
    algorithms = manifest.get("algorithms")
    if not isinstance(dataset, dict) or not isinstance(algorithms, dict):
        raise ValueError("benchmark manifest missing dataset/algorithms mappings")
    fast_livo = algorithms.get("fast_livo2")
    if not isinstance(fast_livo, dict):
        raise ValueError("benchmark manifest missing algorithms.fast_livo2")

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
        "extrinsic": _resolve_extrinsic(manifest, fast_livo, fast_livo_config),
        "preprocess": _load_fast_livo_preprocess(fast_livo_config),
        "point_source_stage": "raw_custom_msg_pre_platform_self_filter",
        "platform_self_filter_reproduced": False,
        "automatic_source_selection": False,
        "semantic_promotion": False,
    }
