#!/usr/bin/env python3
"""Export sampled map-frame Livox first-return rays from a benchmark full-bag run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Pair a raw Livox CustomMsg rosbag with the FAST-LIVO2 full-bag "
            "/aft_mapped_to_init trajectory and export schema-v1 observation rays. "
            "ROS 2 is imported only after arguments and provenance are validated."
        )
    )
    parser.add_argument("--benchmark-run", required=True)
    parser.add_argument("--fast-livo-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lidar-bag")
    parser.add_argument("--lidar-topic")
    parser.add_argument("--trajectory-bag")
    parser.add_argument("--trajectory-topic")
    parser.add_argument("--output-frame-id", default="map")
    parser.add_argument(
        "--allow-parent-frame-alias",
        action="store_true",
        help=(
            "Allow FAST-LIVO2's numeric parent frame (normally camera_init) to be "
            "stored under the canonical P1 frame name (normally map). No transform is applied."
        ),
    )
    parser.add_argument("--max-pose-gap-s", type=float, required=True)
    parser.add_argument("--export-point-stride", type=int, required=True)
    parser.add_argument("--scan-stride", type=int, default=1)
    parser.add_argument("--max-return-range-m", type=float)
    parser.add_argument("--max-rays", type=int)
    return parser


def main():
    args = build_parser().parse_args()

    from agt_map_reconstruction.io.rosbag_livox_ray_export import (
        export_livox_observation_rays,
        resolve_benchmark_ray_export_contract,
    )
    from agt_map_reconstruction.maps.observation_ray_bundle import (
        write_observation_ray_bundle,
    )

    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    contract = resolve_benchmark_ray_export_contract(
        args.benchmark_run,
        fast_livo_config=args.fast_livo_config,
        lidar_bag=args.lidar_bag,
        lidar_topic=args.lidar_topic,
        trajectory_bag=args.trajectory_bag,
        trajectory_topic=args.trajectory_topic,
    )
    bundle, summary = export_livox_observation_rays(
        contract,
        output_frame_id=args.output_frame_id,
        allow_parent_frame_alias=args.allow_parent_frame_alias,
        max_pose_gap_s=args.max_pose_gap_s,
        export_point_stride=args.export_point_stride,
        scan_stride=args.scan_stride,
        max_return_range_m=args.max_return_range_m,
        max_rays=args.max_rays,
    )

    rays_path = output / "observation_rays.npz"
    write_observation_ray_bundle(rays_path, bundle)
    manifest = {
        "schema_version": 1,
        "source_contract": contract,
        "timing": {
            "livox_point_timestamp_formula": "timebase_ns + offset_time_ns",
            "trajectory_time_source": "nav_msgs/msg/Odometry.header.stamp",
            "pose_interpolation": "linear_position_shortest_arc_quaternion_slerp",
            "pose_extrapolation": False,
            "max_pose_gap_s": float(args.max_pose_gap_s),
        },
        "sampling": {
            "scan_stride": int(args.scan_stride),
            "export_point_stride": int(args.export_point_stride),
            "max_return_range_m": (
                None if args.max_return_range_m is None else float(args.max_return_range_m)
            ),
            "max_rays": None if args.max_rays is None else int(args.max_rays),
        },
        "frames": {
            "trajectory_parent_frame": summary["trajectory_parent_frame"],
            "trajectory_child_frame": summary["trajectory_child_frame"],
            "output_frame_id": bundle.frame_id,
            "parent_frame_alias_without_transform": bool(
                bundle.frame_id != summary["trajectory_parent_frame"]
            ),
        },
        "preprocessing_scope": {
            "fast_livo_custommsg_quality_filter_reproduced": True,
            "platform_self_filter_reproduced": False,
            "source_stage": contract["point_source_stage"],
            "note": (
                "This exporter reads the recorded raw CustomMsg. Platform self-filter "
                "participation in the selected FAST-LIVO2 replay must be audited separately."
            ),
        },
        "summary": summary,
        "policy": {
            "first_return_endpoint_is_not_free": True,
            "automatic_semantic_promotion": False,
            "semantic_promotion": False,
        },
    }
    manifest_path = output / "observation_ray_export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("output:", output)
    print("run_id:", contract["run_id"])
    print("lidar_bag:", contract["lidar"]["bag"])
    print("lidar_topic:", contract["lidar"]["topic"])
    print("lidar_metadata_scans:", contract["lidar"]["message_count"])
    print("trajectory_bag:", contract["trajectory"]["bag"])
    print("trajectory_topic:", contract["trajectory"]["topic"])
    print("trajectory_metadata_poses:", contract["trajectory"]["message_count"])
    print("trajectory_parent_frame:", summary["trajectory_parent_frame"])
    print("trajectory_child_frame:", summary["trajectory_child_frame"])
    print("output_frame_id:", bundle.frame_id)
    print("pose_supported_rays:", summary["pose_supported_ray_count"])
    print("pose_rejected_before:", summary["pose_rejected_before_trajectory"])
    print("pose_rejected_after:", summary["pose_rejected_after_trajectory"])
    print("pose_rejected_gap:", summary["pose_rejected_large_gap"])
    print("trajectory_interval_p95_s:", f"{summary['trajectory_interval_p95_s']:.6f}")
    print("trajectory_interval_max_s:", f"{summary['trajectory_interval_max_s']:.6f}")
    print("max_offset_time_ns:", summary["max_offset_time_ns"])
    print("header_timebase_abs_delta_max_s:", summary["header_timebase_abs_delta_max_s"])
    print("platform_self_filter_reproduced: false")
    print("semantic_promotion: false")


if __name__ == "__main__":
    main()
