#!/usr/bin/env python3
"""Stream rosbag Livox rays directly into conservative ground-aware support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Stream raw Livox CustomMsg observations through the selected FAST-LIVO2 "
            "trajectory into global ground-aware ray- and scan-support grids without "
            "writing a full observation_rays.npz."
        )
    )
    parser.add_argument("--benchmark-run", required=True)
    parser.add_argument("--fast-livo-config", required=True)
    parser.add_argument("--ground-reference", required=True)
    parser.add_argument("--grid-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lidar-bag")
    parser.add_argument("--lidar-topic")
    parser.add_argument("--trajectory-bag")
    parser.add_argument("--trajectory-topic")
    parser.add_argument("--output-frame-id", default="map")
    parser.add_argument("--allow-parent-frame-alias", action="store_true")
    parser.add_argument("--max-pose-gap-s", type=float, required=True)
    parser.add_argument("--scan-stride", type=int, default=1)
    parser.add_argument("--export-point-stride", type=int, required=True)
    parser.add_argument("--batch-ray-limit", type=int, default=250000)
    parser.add_argument("--max-return-range-m", type=float)
    parser.add_argument("--max-rays", type=int)
    parser.add_argument("--min-ground-relative-height-m", type=float, required=True)
    parser.add_argument("--max-ground-relative-height-m", type=float, required=True)
    parser.add_argument("--min-support-rays", type=int, default=1)
    parser.add_argument("--min-ray-range-m", type=float, default=0.0)
    parser.add_argument("--max-ray-range-m", type=float)
    return parser


def _metadata_from_payload(payload):
    from agt_map_reconstruction.maps.grid_geometry import GridMetadata

    grid = payload.get("grid")
    if not isinstance(grid, dict):
        raise ValueError("grid manifest must contain a grid mapping")
    origin = grid.get("origin")
    if not isinstance(origin, list) or len(origin) != 3:
        raise ValueError("grid.origin must contain [x, y, yaw]")
    return GridMetadata(
        resolution=float(grid["resolution"]),
        origin_x=float(origin[0]),
        origin_y=float(origin[1]),
        origin_yaw=float(origin[2]),
        width=int(grid["width"]),
        height=int(grid["height"]),
        frame_id=str(grid.get("frame_id", "map")),
    )


def main():
    args = build_parser().parse_args()

    from agt_map_reconstruction.io.rosbag_livox_ground_evidence import (
        stream_livox_ground_aware_evidence,
    )
    from agt_map_reconstruction.io.rosbag_livox_ray_contract import (
        resolve_benchmark_ray_export_contract,
    )
    from agt_map_reconstruction.maps.ground_aware_ray_evidence import (
        GroundAwareRayConfig,
    )

    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    ground_path = Path(args.ground_reference).expanduser().resolve()
    manifest_path = Path(args.grid_manifest).expanduser().resolve()
    ground = np.load(ground_path, allow_pickle=False)
    grid_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = _metadata_from_payload(grid_payload)

    contract = resolve_benchmark_ray_export_contract(
        args.benchmark_run,
        fast_livo_config=args.fast_livo_config,
        lidar_bag=args.lidar_bag,
        lidar_topic=args.lidar_topic,
        trajectory_bag=args.trajectory_bag,
        trajectory_topic=args.trajectory_topic,
    )
    config = GroundAwareRayConfig(
        min_ground_relative_height_m=args.min_ground_relative_height_m,
        max_ground_relative_height_m=args.max_ground_relative_height_m,
        min_support_rays=args.min_support_rays,
        min_ray_range_m=args.min_ray_range_m,
        max_ray_range_m=args.max_ray_range_m,
    )
    result = stream_livox_ground_aware_evidence(
        contract,
        ground,
        metadata,
        config,
        output_frame_id=args.output_frame_id,
        allow_parent_frame_alias=args.allow_parent_frame_alias,
        max_pose_gap_s=args.max_pose_gap_s,
        export_point_stride=args.export_point_stride,
        scan_stride=args.scan_stride,
        max_return_range_m=args.max_return_range_m,
        max_rays=args.max_rays,
        batch_ray_limit=args.batch_ray_limit,
    )

    np.save(output / "ray_free_support_count.npy", result["support_count"])
    np.save(
        output / "ray_free_support_mask.npy",
        result["support_mask"].astype(np.uint8),
    )
    if "scan_support_count" not in result:
        raise RuntimeError("streaming evidence did not preserve scan identity")
    np.save(output / "scan_free_support_count.npy", result["scan_support_count"])

    payload = {
        "schema_version": 2,
        "source_contract": contract,
        "source_ground_reference": str(ground_path),
        "source_grid_manifest": str(manifest_path),
        "grid": metadata.to_dict(),
        "ray_policy": {
            "min_ground_relative_height_m": float(config.min_ground_relative_height_m),
            "max_ground_relative_height_m": float(config.max_ground_relative_height_m),
            "min_support_rays": int(config.min_support_rays),
            "min_ray_range_m": float(config.min_ray_range_m),
            "max_ray_range_m": (
                None if config.max_ray_range_m is None else float(config.max_ray_range_m)
            ),
            "hit_cell_is_free": False,
            "requires_finite_ground_reference": True,
            "support_threshold_applied_after_global_accumulation": True,
        },
        "scan_support_policy": {
            "definition": "at most one supporting vote per physical Livox scan per grid cell",
            "threshold_applied": False,
            "purpose": "density-independent temporal support diagnostic",
        },
        "streaming": {
            "scan_stride": int(args.scan_stride),
            "export_point_stride": int(args.export_point_stride),
            "batch_ray_limit": int(args.batch_ray_limit),
            "max_rays": None if args.max_rays is None else int(args.max_rays),
            "full_ray_npz_written": False,
        },
        "summary": result["summary"],
        "policy": {
            "platform_self_filter_reproduced": False,
            "navigation_map_modified": False,
            "automatic_semantic_promotion": False,
            "semantic_promotion": False,
        },
    }
    (output / "streaming_observation_evidence_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    s = result["summary"]
    print("output:", output)
    print("run_id:", contract["run_id"])
    print("scan_stride:", s["scan_stride"])
    print("export_point_stride:", s["export_point_stride"])
    print("selected_scans:", s["selected_scan_count"])
    print("sampled_points:", s["sampled_point_count"])
    print("pose_supported_rays:", s["pose_supported_ray_count"])
    print("pose_rejected_before:", s["pose_rejected_before_trajectory"])
    print("pose_rejected_after:", s["pose_rejected_after_trajectory"])
    print("pose_rejected_gap:", s["pose_rejected_large_gap"])
    print("accepted_ground_aware_rays:", s["accepted_ray_count"])
    print("ray_supported_cells:", s["supported_cell_count"])
    print("scan_supported_cells:", s["scan_supported_cell_count"])
    print("max_scan_support_count:", s["max_scan_support_count"])
    print("batch_count:", s["batch_count"])
    print("full_ray_npz_written: false")
    print("navigation_map_modified: false")
    print("semantic_promotion: false")


if __name__ == "__main__":
    main()
