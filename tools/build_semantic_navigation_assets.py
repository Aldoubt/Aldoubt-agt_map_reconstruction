#!/usr/bin/env python3
"""Build conservative semantic geometry and a Nav2 static map from a LIO PCD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Convert a LIO PCD into robust ground evidence, recovered agricultural "
            "aisles, current semantic labels, and a Nav2 static-map bundle."
        )
    )
    parser.add_argument("--pcd", required=True)
    parser.add_argument("--output", default="results/semantic-navigation-assets")
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--low-quantile", type=float, default=0.10)
    parser.add_argument("--histogram-bins", type=int, default=64)

    parser.add_argument("--min-points-per-cell", type=int, default=3)
    parser.add_argument("--min-ground-support-cells", type=int, default=2)
    parser.add_argument("--ground-window-m", type=float, default=0.50)
    parser.add_argument("--ground-percentile", type=float, default=20.0)
    parser.add_argument("--ground-seed-percentile", type=float, default=10.0)
    parser.add_argument("--max-ground-step-m", type=float, default=0.20)
    parser.add_argument("--max-interpolation-gap-m", type=float, default=0.25)
    parser.add_argument("--obstacle-height-m", type=float, default=0.15)

    parser.add_argument(
        "--row-direction",
        type=float,
        nargs=2,
        metavar=("DX", "DY"),
        help="Optional explicit map-grid row direction. If omitted, infer from evidence.",
    )
    parser.add_argument("--min-longitudinal-support-ratio", type=float, default=0.50)
    parser.add_argument("--min-aisle-width-m", type=float, default=0.30)
    parser.add_argument("--min-aisle-length-m", type=float, default=2.0)
    parser.add_argument(
        "--confirmed-free-only",
        action="store_true",
        help="Do not use bounded interpolated ground as aisle-geometry support.",
    )
    return parser


def main():
    args = build_parser().parse_args()

    # Keep heavy PCD/Open3D imports out of module import so --help remains usable
    # in lightweight CI environments.
    from agt_map_reconstruction.io.pcd_loader import load_pcd
    from agt_map_reconstruction.maps.ground_evidence import GroundEvidenceConfig
    from agt_map_reconstruction.maps.semantic_pipeline import (
        build_semantic_assets_from_points,
    )

    pcd_path = Path(args.pcd).expanduser().resolve()
    points = load_pcd(str(pcd_path))
    config = GroundEvidenceConfig(
        resolution=args.resolution,
        min_points_per_cell=args.min_points_per_cell,
        min_ground_support_cells=args.min_ground_support_cells,
        ground_window_m=args.ground_window_m,
        ground_percentile=args.ground_percentile,
        ground_seed_percentile=args.ground_seed_percentile,
        max_ground_step_m=args.max_ground_step_m,
        max_interpolation_gap_m=args.max_interpolation_gap_m,
        obstacle_height_m=args.obstacle_height_m,
    )

    result = build_semantic_assets_from_points(
        points=points,
        output_dir=args.output,
        resolution=args.resolution,
        chunk_size=args.chunk_size,
        low_quantile=args.low_quantile,
        histogram_bins=args.histogram_bins,
        ground_config=config,
        row_direction=args.row_direction,
        min_longitudinal_support_ratio=args.min_longitudinal_support_ratio,
        min_width_m=args.min_aisle_width_m,
        min_length_m=args.min_aisle_length_m,
        include_interpolated=not args.confirmed_free_only,
    )

    output = Path(args.output)
    source = {
        "pcd": str(pcd_path),
        "size_bytes": int(pcd_path.stat().st_size),
        "point_count": int(len(points)),
    }
    (output / "source.json").write_text(
        json.dumps(source, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    manifest = result["bundle"]["manifest"]
    print("output:", output)
    print("aisles:", manifest["aisle_count"])
    print("row_direction:", manifest["row_direction"])
    print("evidence_counts:", manifest["evidence_counts"])
    print("nav2_map:", output / "navigation" / "navigation_base_map.yaml")


if __name__ == "__main__":
    main()
