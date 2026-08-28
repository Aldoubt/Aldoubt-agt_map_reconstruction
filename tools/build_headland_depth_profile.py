#!/usr/bin/env python3
"""Build finite endpoint-relative headland depth masks from frozen D3.1 geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Build finite entry/exit headland depth bands outward from the outer "
            "edge of fused structural endpoint uncertainty. No physical site "
            "boundary or HARD-boundary flood fill is used."
        )
    )
    parser.add_argument("--fused-structural-bundle", required=True)
    parser.add_argument("--fused-uncertainty", required=True)
    parser.add_argument(
        "--depth-edges-m",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0, 2.0, 4.0],
    )
    parser.add_argument(
        "--uncertainty-quantile",
        choices=("p50", "p90", "p95", "max"),
        default="p95",
    )
    parser.add_argument("--output", required=True)
    return parser


def _read_pgm(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"failed to read PGM: {path}")
    return np.flipud(image).astype(np.uint8, copy=False)


def _blend_mask(image, mask, color, alpha):
    overlay = image.copy()
    overlay[np.asarray(mask, dtype=bool)] = np.asarray(color, dtype=np.uint8)
    cv2.addWeighted(overlay, float(alpha), image, 1.0 - float(alpha), 0.0, dst=image)


def _depth_band_colors(side, count):
    if side == "entry":
        palette = [
            (40, 210, 40),
            (70, 190, 70),
            (100, 170, 100),
            (130, 150, 130),
            (155, 135, 155),
            (175, 120, 175),
        ]
    else:
        palette = [
            (220, 120, 30),
            (205, 135, 55),
            (190, 150, 80),
            (175, 165, 105),
            (160, 180, 130),
            (145, 195, 155),
        ]
    if count <= len(palette):
        return palette[:count]
    return [palette[min(i, len(palette) - 1)] for i in range(count)]


def main():
    args = build_parser().parse_args()

    from agt_map_reconstruction.maps.headland_depth_profile import (
        build_headland_depth_profile,
    )

    fused_path = Path(args.fused_structural_bundle).expanduser().resolve()
    uncertainty_path = Path(args.fused_uncertainty).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    fused = json.loads(fused_path.read_text(encoding="utf-8"))
    uncertainty = json.loads(uncertainty_path.read_text(encoding="utf-8"))
    fused_sources = dict(fused.get("sources") or {})
    map_text = fused_sources.get("map")
    if not map_text:
        raise ValueError("fused structural bundle must preserve sources.map")
    map_path = Path(map_text).expanduser().resolve()
    base = _read_pgm(map_path)

    result, masks = build_headland_depth_profile(
        fused,
        uncertainty,
        grid_shape_yx=base.shape,
        depth_edges_m=args.depth_edges_m,
        uncertainty_quantile=args.uncertainty_quantile,
    )
    result["sources"] = {
        "fused_structural_bundle": str(fused_path),
        "fused_uncertainty": str(uncertainty_path),
        "map": str(map_path),
        "row_lattice_completion": fused_sources.get("row_lattice_completion"),
        "source_structural_bundle": fused_sources.get("source_structural_bundle"),
        "targeted_3d_audit": fused_sources.get("targeted_3d_audit"),
    }

    mask_files = {}
    for key, mask in masks.items():
        filename = f"{key}_mask.npy"
        np.save(output / filename, np.asarray(mask, dtype=bool))
        mask_files[key] = filename
    result["mask_files"] = mask_files
    (output / "headland_depth_profile.json").write_text(
        json.dumps(result, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    image = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    # Depth bands are evaluation-only. Lower outward depth is drawn more strongly;
    # the structural uncertainty bands remain visually distinct.
    for side in ("entry", "exit"):
        bands = result[side]["bands"]
        colors = _depth_band_colors(side, len(bands))
        for index, item in enumerate(bands):
            alpha = max(0.08, 0.22 - 0.03 * index)
            _blend_mask(image, masks[item["mask_key"]], colors[index], alpha)

    _blend_mask(image, masks["entry_boundary_uncertainty"], (0, 165, 255), 0.34)
    _blend_mask(image, masks["exit_boundary_uncertainty"], (255, 100, 0), 0.34)
    _blend_mask(image, masks["structurally_unresolved_cross"], (0, 0, 255), 0.42)

    display = np.flipud(image).copy()
    cv2.rectangle(display, (8, 8), (790, 164), (30, 30, 30), -1)
    legend = [
        ("green shades: finite entry outward depth bands", (0, 220, 0)),
        ("blue shades: finite exit outward depth bands", (220, 150, 50)),
        ("orange/blue: fused structural boundary uncertainty", (0, 165, 255)),
        ("red: structurally unresolved ridge cross-span; excluded", (0, 0, 255)),
        ("depth=0 is outer edge of fused structural uncertainty", (255, 255, 255)),
        ("no physical wall/site boundary used; evaluation only", (255, 220, 0)),
    ]
    for index, (text, color) in enumerate(legend):
        cv2.putText(
            display,
            text,
            (18, 30 + 24 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(output / "headland_depth_profile.png"), display)

    print("output:", output)
    print("method:", result["method"])
    print("uncertainty_quantile:", result["uncertainty_quantile"])
    print("depth_edges_m:", result["depth_edges_m"])
    print(f"max_outward_depth_m: {result['max_outward_depth_m']:.6f}")
    print("unresolved_ridges:", result["unresolved_ridge_ids"])
    for side in ("entry", "exit"):
        print(
            f"{side}: boundary_uncertainty_cells="
            f"{result[side]['boundary_uncertainty_cell_count']}"
        )
        for item in result[side]["bands"]:
            print(
                f"  {side} {item['depth_min_m']:.3f}-{item['depth_max_m']:.3f}m: "
                f"cells={item['cell_count']} mask={item['mask_key']}"
            )
    print("physical_site_boundary_required: false")
    print("hard_boundary_flood_fill_used: false")
    print("automatic_depth_band_selection: false")
    print("automatic_acceptance: false")
    print("navigation_map_modified: false")
    print("semantic_promotion: false")


if __name__ == "__main__":
    main()
