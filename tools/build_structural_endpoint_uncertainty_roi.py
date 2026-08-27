#!/usr/bin/env python3
"""Build conservative endpoint ROIs from fused D3.1 structural uncertainty."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Build conservative headland ROIs outside fused structural trend +/- "
            "residual uncertainty while excluding unresolved ridge cross strips."
        )
    )
    parser.add_argument("--fused-structural-bundle", required=True)
    parser.add_argument("--fused-uncertainty", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--uncertainty-quantile",
        choices=("p50", "p90", "p95", "max"),
        default="p95",
        help="diagnostic structural residual quantile; default p95",
    )
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


def main():
    args = build_parser().parse_args()

    from agt_map_reconstruction.maps.structural_endpoint_uncertainty_roi import (
        build_structural_endpoint_uncertainty_roi,
    )

    fused_path = Path(args.fused_structural_bundle).expanduser().resolve()
    uncertainty_path = Path(args.fused_uncertainty).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    fused = json.loads(fused_path.read_text(encoding="utf-8"))
    uncertainty = json.loads(uncertainty_path.read_text(encoding="utf-8"))
    map_text = (fused.get("sources") or {}).get("map")
    if not map_text:
        raise ValueError("fused structural bundle must preserve sources.map")
    map_path = Path(map_text).expanduser().resolve()
    base = _read_pgm(map_path)

    result, masks = build_structural_endpoint_uncertainty_roi(
        fused,
        uncertainty,
        grid_shape_yx=base.shape,
        uncertainty_quantile=args.uncertainty_quantile,
    )
    result["sources"] = {
        "fused_structural_bundle": str(fused_path),
        "fused_uncertainty": str(uncertainty_path),
        "map": str(map_path),
    }

    mask_files = {
        "entry_conservative_outward": "entry_conservative_outward_mask.npy",
        "entry_boundary_uncertainty": "entry_boundary_uncertainty_mask.npy",
        "exit_conservative_outward": "exit_conservative_outward_mask.npy",
        "exit_boundary_uncertainty": "exit_boundary_uncertainty_mask.npy",
        "structurally_unresolved_cross": "structurally_unresolved_cross_mask.npy",
    }
    for key, filename in mask_files.items():
        np.save(output / filename, masks[key].astype(bool, copy=False))
    result["mask_files"] = mask_files
    (output / "structural_endpoint_uncertainty_roi.json").write_text(
        json.dumps(result, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    image = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    # Conservative ROIs are deliberately subtle; uncertainty and unresolved
    # regions remain visually dominant so the figure cannot imply semantic free.
    _blend_mask(image, masks["entry_conservative_outward"], (0, 180, 0), 0.12)
    _blend_mask(image, masks["exit_conservative_outward"], (180, 120, 0), 0.12)
    _blend_mask(image, masks["entry_boundary_uncertainty"], (0, 165, 255), 0.28)
    _blend_mask(image, masks["exit_boundary_uncertainty"], (255, 100, 0), 0.28)
    _blend_mask(image, masks["structurally_unresolved_cross"], (0, 0, 255), 0.38)

    display = np.flipud(image).copy()
    cv2.rectangle(display, (8, 8), (730, 140), (30, 30, 30), -1)
    legend = [
        ("green tint: conservative entry-side outward ROI (evaluation only)", (0, 180, 0)),
        ("blue tint: conservative exit-side outward ROI (evaluation only)", (180, 120, 0)),
        ("orange/blue bands: fused endpoint boundary uncertainty", (0, 165, 255)),
        ("red strip: structurally unresolved ridge cross-span; excluded from ROI", (0, 0, 255)),
        ("no ROI mask changes navigation or promotes semantic free", (255, 220, 0)),
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
    cv2.imwrite(str(output / "structural_endpoint_uncertainty_roi.png"), display)

    print("output:", output)
    print("method:", result["method"])
    print("uncertainty_quantile:", result["uncertainty_quantile"])
    print("unresolved_ridges:", result["unresolved_ridge_ids"])
    print("structurally_unresolved_cross_cells:", result["structurally_unresolved_cross_cell_count"])
    for side in ("entry", "exit"):
        item = result[side]
        print(
            f"{side}: uncertainty_half_width_m={item['uncertainty_half_width_m']:.6f} "
            f"conservative_outward_cells={item['conservative_outward_cell_count']} "
            f"boundary_uncertainty_cells={item['boundary_uncertainty_cell_count']}"
        )
    print("center_trend_promoted_to_semantic_boundary: false")
    print("unresolved_cross_strip_excluded: true")
    print("automatic_acceptance: false")
    print("navigation_map_modified: false")
    print("semantic_promotion: false")


if __name__ == "__main__":
    main()
