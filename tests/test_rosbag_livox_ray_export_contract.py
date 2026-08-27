import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from agt_map_reconstruction.io.rosbag_livox_ray_contract import (
    resolve_benchmark_ray_export_contract,
)


def _write_metadata(path, topics):
    path.mkdir(parents=True)
    payload = {
        "rosbag2_bagfile_information": {
            "version": 5,
            "storage_identifier": "sqlite3",
            "message_count": sum(item[2] for item in topics),
            "topics_with_message_count": [
                {
                    "topic_metadata": {
                        "name": name,
                        "type": type_name,
                        "serialization_format": "cdr",
                    },
                    "message_count": count,
                }
                for name, type_name, count in topics
            ],
        }
    }
    (path / "metadata.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")


def _fixture(tmp_path):
    raw = tmp_path / "green-house"
    _write_metadata(
        raw,
        [
            ("/agt/sensors/lidar/custom", "livox_ros_driver2/msg/CustomMsg", 6230),
            ("/agt/sensors/imu/data", "sensor_msgs/msg/Imu", 124600),
        ],
    )
    run = tmp_path / "samebag_v1_full"
    trajectory = run / "raw" / "fast_livo2" / "fast_livo_trajectory"
    _write_metadata(
        trajectory,
        [
            ("/aft_mapped_to_init", "nav_msgs/msg/Odometry", 6215),
            ("/path", "nav_msgs/msg/Path", 6215),
        ],
    )
    manifest = {
        "run_id": "samebag_v1_full_20260817_162851",
        "dataset": {"bag_dir": str(raw)},
        "calibration": {
            "rotation_lidar_to_imu_row_major": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            "translation_lidar_to_imu_m": [0.011, 0.02329, -0.04412],
        },
        "algorithms": {
            "fast_livo2": {
                "extrinsic_convention": "LIDAR_TO_IMU",
                "topics": {"outputs": {"trajectory": "/aft_mapped_to_init"}},
                "trajectory_contract": {
                    "pose_semantics": "T_PARENT_TRACKED",
                    "tracked_frame_physical": "IMU_BODY",
                },
            }
        },
    }
    run.mkdir(parents=True, exist_ok=True)
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    config = tmp_path / "mid360_lio_only.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "/**": {
                    "ros__parameters": {
                        "preprocess": {
                            "scan_line": 4,
                            "blind": 0.5,
                            "point_filter_num": 1,
                        },
                        "extrin_calib": {
                            "extrinsic_R": [
                                1.0, 0.0, 0.0,
                                0.0, 1.0, 0.0,
                                0.0, 0.0, 1.0,
                            ],
                            "extrinsic_T": [0.011, 0.02329, -0.04412],
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return raw, run, config


def test_contract_selects_unique_raw_livox_and_full_bag_odometry(tmp_path):
    raw, run, config = _fixture(tmp_path)
    contract = resolve_benchmark_ray_export_contract(
        run,
        fast_livo_config=config,
    )

    assert contract["lidar"]["bag"] == str(raw.resolve())
    assert contract["lidar"]["topic"] == "/agt/sensors/lidar/custom"
    assert contract["lidar"]["message_count"] == 6230
    assert contract["trajectory"]["topic"] == "/aft_mapped_to_init"
    assert contract["trajectory"]["message_count"] == 6215
    assert contract["preprocess"]["scan_line_count"] == 4
    assert contract["preprocess"]["blind_range_m"] == pytest.approx(0.5)
    assert contract["extrinsic"]["source"] == "benchmark_manifest"
    assert contract["platform_self_filter_reproduced"] is False
    assert contract["semantic_promotion"] is False


def test_contract_uses_explicit_fast_livo_config_when_manifest_has_no_calibration(tmp_path):
    raw, run, config = _fixture(tmp_path)
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("calibration")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    contract = resolve_benchmark_ray_export_contract(
        run,
        fast_livo_config=config,
    )

    assert contract["lidar"]["bag"] == str(raw.resolve())
    assert contract["extrinsic"]["source"] == str(config.resolve())
    assert contract["extrinsic"]["source_field"] == "ros__parameters.extrin_calib"
    assert contract["extrinsic"]["rotation_lidar_to_imu_row_major"] == [
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ]
    assert contract["extrinsic"]["translation_lidar_to_imu_m"] == pytest.approx(
        [0.011, 0.02329, -0.04412]
    )


def test_contract_rejects_different_lidar_bag_than_frozen_run(tmp_path):
    _, run, config = _fixture(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="differs from benchmark manifest"):
        resolve_benchmark_ray_export_contract(
            run,
            fast_livo_config=config,
            lidar_bag=other,
        )


def test_export_cli_help_does_not_require_ros_runtime():
    script = Path(__file__).resolve().parents[1] / "tools" / "export_livox_observation_rays.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0
    assert "--benchmark-run" in completed.stdout
    assert "--export-point-stride" in completed.stdout
