#!/usr/bin/env python3
"""Rebuild semantic/aisle/Nav2 assets from an existing evidence grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild semantic geometry and Nav2 assets from an existing evidence.npy "
            "without re-reading or re-rasterizing the source PCD."
        )
    )
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--row-direction",
        type=float,
        nargs=2,
        metavar=("DX", "DY"),
        help="Optional explicit row direction. If omitted, infer it from evidence.",
    )
    parser.add_argument(
        "--occupied-aisle-conflicts",
        choices=("hard", "candidate"),
        default="hard",
        help=(
            "How to interpret OCCUPIED_CONFIRMED cells inside recovered aisles. "
            "Default hard preserves the conservative baseline; candidate runs an "
            "explicit aisle-conditioned advisory-layer diagnostic."
        ),
    )
    return parser


def _metadata_from_manifest(payload):
    from agt_map_reconstruction.maps.grid_geometry import GridMetadata

    grid = payload.get("grid")
    if not isinstance(grid, dict):
        raise ValueError("manifest must contain a grid mapping")
    origin = grid.get("origin")
    if not isinstance(origin, list) or len(origin) != 3:
        raise ValueError("manifest grid.origin must contain [x, y, yaw]")
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

    from agt_map_reconstruction.maps.semantic_assets import (
        write_semantic_navigation_assets,
    )
    from agt_map_reconstruction.maps.semantic_pipeline import (
        infer_row_direction_from_evidence,
    )

    evidence_path = Path(args.evidence).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")

    evidence = np.load(evidence_path, allow_pickle=False)
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = _metadata_from_manifest(source_manifest)
    expected_shape = (metadata.height, metadata.width)
    if evidence.shape != expected_shape:
        raise ValueError(
            f"evidence shape {evidence.shape} does not match manifest grid {expected_shape}"
        )

    policy = source_manifest.get("geometry_policy", {})
    include_interpolated = bool(policy.get("include_interpolated", True))
    support_ratio = float(policy.get("min_longitudinal_support_ratio", 0.50))
    min_width_m = float(policy.get("min_width_m", 0.30))
    min_length_m = float(policy.get("min_length_m", 2.0))
    wide_band_iqr_factor = float(policy.get("wide_band_iqr_factor", 1.50))

    if args.row_direction is None:
        direction = infer_row_direction_from_evidence(
            evidence,
            include_interpolated=include_interpolated,
        )
        direction_source = "evidence_pca_occupied_banding"
    else:
        direction = np.asarray(args.row_direction, dtype=float)
        direction_source = "explicit"

    result = write_semantic_navigation_assets(
        evidence=evidence,
        metadata=metadata,
        row_direction=direction,
        output_dir=output,
        min_longitudinal_support_ratio=support_ratio,
        min_width_m=min_width_m,
        min_length_m=min_length_m,
        include_interpolated=include_interpolated,
        occupied_aisle_conflict_policy=args.occupied_aisle_conflicts,
        wide_band_iqr_factor=wide_band_iqr_factor,
    )

    rebuild_manifest = {
        "schema_version": 1,
        "source_evidence": str(evidence_path),
        "source_manifest": str(manifest_path),
        "row_direction_source": direction_source,
        "row_direction": result["manifest"]["row_direction"],
        "raw_row_band_count": result["manifest"]["raw_row_band_count"],
        "aisle_count": result["manifest"]["aisle_count"],
        "open_area_candidate_count": result["manifest"][
            "open_area_candidate_count"
        ],
        "geometry_policy": result["manifest"]["geometry_policy"],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "rebuild_manifest.json").write_text(
        json.dumps(rebuild_manifest, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    print("output:", output)
    print("raw_row_bands:", result["manifest"]["raw_row_band_count"])
    print("aisles:", result["manifest"]["aisle_count"])
    print("open_area_candidates:", result["manifest"]["open_area_candidate_count"])
    print("row_direction:", result["manifest"]["row_direction"])
    print("evidence_counts:", result["manifest"]["evidence_counts"])
    print(
        "aisle_conflict_candidates:",
        result["manifest"]["aisle_conflict_candidate_count"],
    )
    print("nav2_map:", output / "navigation" / "navigation_base_map.yaml")


if __name__ == "__main__":
    main()
