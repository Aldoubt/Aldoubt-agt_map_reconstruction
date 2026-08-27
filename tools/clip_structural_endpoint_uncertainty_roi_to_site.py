#!/usr/bin/env python3
"""Clip frozen D3.1 uncertainty ROI masks to an anchor-validated site interior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Intersect the existing unbounded D3.1 uncertainty ROI partition with "
            "an anchor-validated flood-filled non-HARD site interior. The original "
            "ROI remains unchanged and is preserved as provenance."
        )
    )
    parser.add_argument("--roi", required=True)
    parser.add_argument("--site-interior", required=True)
    parser.add_argument("--output", required=True)
    return parser


def _load_masks(payload, root):
    files = dict(payload.get("mask_files") or {})
    if not files:
        raise ValueError("payload must contain mask_files")
    return {
        name: np.load(root / filename, allow_pickle=False).astype(bool, copy=False)
        for name, filename in files.items()
    }


def _read_map(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"failed to read PGM: {path}")
    return np.flipud(image).astype(np.uint8, copy=False)


def _blend(image, mask, color, alpha):
    overlay = image.copy()
    overlay[np.asarray(mask, dtype=bool)] = np.asarray(color, dtype=np.uint8)
    cv2.addWeighted(overlay, float(alpha), image, 1.0 - float(alpha), 0.0, dst=image)


def _require_anchor_validated_site(site_payload):
    if site_payload.get("status") != "ok":
        raise ValueError("site interior flood fill is not valid for clipping")
    if site_payload.get("interior_anchor_validation_requested") is not True:
        raise ValueError(
            "site interior must be anchor-validated before clipping; topology-only "
            "site masks are diagnostic and cannot define the physical ROI"
        )
    if site_payload.get("interior_anchor_validation_passed") is not True:
        raise ValueError("site interior anchor validation did not pass")
    if int(site_payload.get("interior_anchor_exterior_reachable_cell_count", 0)) != 0:
        raise ValueError("site interior contains exterior-reachable trusted anchors")


def main():
    args = build_parser().parse_args()

    from agt_map_reconstruction.maps.structural_endpoint_uncertainty_roi_site_clip import (
        clip_uncertainty_roi_to_site_interior,
    )

    roi_path = Path(args.roi).expanduser().resolve()
    site_path = Path(args.site_interior).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    roi_payload = json.loads(roi_path.read_text(encoding="utf-8"))
    site_payload = json.loads(site_path.read_text(encoding="utf-8"))
    _require_anchor_validated_site(site_payload)

    roi_masks = _load_masks(roi_payload, roi_path.parent)
    site_masks = _load_masks(site_payload, site_path.parent)
    if "site_interior_nonhard" not in site_masks:
        raise ValueError("site interior payload must contain site_interior_nonhard mask")

    result, clipped = clip_uncertainty_roi_to_site_interior(
        roi_masks,
        site_masks["site_interior_nonhard"],
    )
    result["uncertainty_quantile"] = roi_payload.get("uncertainty_quantile")
    result["unresolved_ridge_ids"] = roi_payload.get("unresolved_ridge_ids", [])
    result["site_interior_anchor_validated"] = True
    result["site_interior_status_basis"] = site_payload.get("status_basis")
    result["sources"] = {
        "unbounded_roi": str(roi_path),
        "site_interior": str(site_path),
    }
    map_text = (roi_payload.get("sources") or {}).get("map") or (
        site_payload.get("sources") or {}
    ).get("map")
    if not map_text:
        raise ValueError("cannot resolve source map from ROI/site payloads")
    map_path = Path(map_text).expanduser().resolve()
    result["sources"]["map"] = str(map_path)

    mask_files = {
        "entry_conservative_outward": "entry_conservative_outward_mask.npy",
        "entry_boundary_uncertainty": "entry_boundary_uncertainty_mask.npy",
        "exit_conservative_outward": "exit_conservative_outward_mask.npy",
        "exit_boundary_uncertainty": "exit_boundary_uncertainty_mask.npy",
        "structurally_unresolved_cross": "structurally_unresolved_cross_mask.npy",
    }
    for name, filename in mask_files.items():
        np.save(output / filename, clipped[name].astype(bool, copy=False))
    result["mask_files"] = mask_files
    (output / "structural_endpoint_uncertainty_roi.json").write_text(
        json.dumps(result, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    base = _read_map(map_path)
    image = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    exterior = site_masks.get("exterior_reachable_nonhard")
    if exterior is not None:
        _blend(image, exterior, (180, 0, 180), 0.12)
    _blend(image, clipped["entry_conservative_outward"], (0, 180, 0), 0.18)
    _blend(image, clipped["exit_conservative_outward"], (180, 120, 0), 0.18)
    _blend(image, clipped["entry_boundary_uncertainty"], (0, 165, 255), 0.30)
    _blend(image, clipped["exit_boundary_uncertainty"], (255, 100, 0), 0.30)
    _blend(image, clipped["structurally_unresolved_cross"], (0, 0, 255), 0.40)

    display = np.flipud(image).copy()
    cv2.rectangle(display, (8, 8), (760, 140), (30, 30, 30), -1)
    legend = [
        ("green/blue: anchor-validated site-interior clipped entry/exit ROI", (0, 220, 0)),
        ("orange/blue bands: clipped fused endpoint uncertainty", (0, 165, 255)),
        ("red: clipped structurally unresolved cross strip", (0, 0, 255)),
        ("magenta tint: exterior-reachable non-HARD removed from evaluation", (255, 0, 255)),
        ("unbounded ROI preserved; no navigation/semantic promotion", (255, 220, 0)),
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
    for name, item in result["regions"].items():
        print(
            f"{name}: original={item['original_cell_count']} "
            f"clipped={item['clipped_cell_count']} "
            f"removed_exterior={item['removed_exterior_cell_count']} "
            f"retained_fraction={item['retained_fraction']:.6f}"
        )
    print("site_interior_anchor_validated: true")
    print("unbounded_roi_preserved: true")
    print("site_interior_mask_is_semantic_free: false")
    print("structural_geometry_modified: false")
    print("navigation_map_modified: false")
    print("semantic_promotion: false")


if __name__ == "__main__":
    main()
