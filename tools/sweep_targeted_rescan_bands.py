#!/usr/bin/env python3
"""Sweep cumulative endpoint reacquisition bands from frozen P1-D3 geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Intersect the frozen P1-D3 entry/exit ROIs with explicit outward-depth bands "
            "and summarize targeted-rescan requirements. No band is selected automatically."
        )
    )
    parser.add_argument("--requirement-labels", required=True)
    parser.add_argument("--baseline-envelope", required=True)
    parser.add_argument("--resolution-m", type=float, required=True)
    parser.add_argument("--max-outward-depth-m", nargs="+", type=float, required=True)
    parser.add_argument("--output", required=True)
    return parser


def _render(labels, mask):
    from agt_map_reconstruction.maps.targeted_rescan_requirement import (
        RESCAN_KNOWN_FREE,
        RESCAN_NO_GROUND_REFERENCE,
        RESCAN_NO_OBSERVATION,
        RESCAN_OCCUPIED,
        RESCAN_REPEATED_SCAN_ANCHOR,
        RESCAN_SINGLE_SCAN_REVISIT,
    )

    image = np.full((*labels.shape, 3), 25, dtype=np.uint8)
    palette = {
        int(RESCAN_OCCUPIED): (0, 0, 0),
        int(RESCAN_KNOWN_FREE): (245, 245, 245),
        int(RESCAN_REPEATED_SCAN_ANCHOR): (70, 210, 70),
        int(RESCAN_SINGLE_SCAN_REVISIT): (0, 190, 255),
        int(RESCAN_NO_OBSERVATION): (40, 90, 230),
        int(RESCAN_NO_GROUND_REFERENCE): (180, 60, 180),
    }
    for value, color in palette.items():
        image[mask & (labels == value)] = color
    return image


def _slug(value):
    return f"{float(value):.2f}".replace(".", "p")


def _write_crop(path, image_grid, roi):
    ys, xs = np.nonzero(roi)
    if not ys.size:
        return None
    pad = 5
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(roi.shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(roi.shape[1], int(xs.max()) + pad + 1)
    crop = image_grid[y0:y1, x0:x1].copy()
    local_roi = roi[y0:y1, x0:x1]
    crop[~local_roi] = (25, 25, 25)
    cv2.imwrite(str(path), np.flipud(crop))
    return {"x0": x0, "x1": x1, "y0": y0, "y1": y1}


def main():
    args = build_parser().parse_args()

    from agt_map_reconstruction.maps.ray_endpoint_support_diagnostics import _side_roi
    from agt_map_reconstruction.maps.targeted_rescan_band_sweep import (
        summarize_targeted_rescan_depth_bands,
    )

    labels_path = Path(args.requirement_labels).expanduser().resolve()
    baseline_path = Path(args.baseline_envelope).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    labels = np.load(labels_path, allow_pickle=False)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    depths = [float(value) for value in args.max_outward_depth_m]

    payload = {
        "schema_version": 1,
        "resolution_m": float(args.resolution_m),
        "max_outward_depth_m_values": depths,
        "endpoint_rois": {},
        "sources": {
            "requirement_labels": str(labels_path),
            "baseline_envelope": str(baseline_path),
        },
        "policy": {
            "diagnostic_acquisition_band_sweep_only": True,
            "automatic_band_selection": False,
            "navigation_map_modified": False,
            "semantic_promotion": False,
        },
    }

    for side_name in ("entry", "exit"):
        roi, _, outward_depth_cells = _side_roi(labels.shape, baseline, side_name)
        side = summarize_targeted_rescan_depth_bands(
            labels,
            roi,
            outward_depth_cells,
            resolution_m=args.resolution_m,
            max_outward_depth_m_values=depths,
        )
        payload["endpoint_rois"][side_name] = side

        for band in side["bands"]:
            max_depth = float(band["max_outward_depth_m"])
            band_mask = roi & (
                outward_depth_cells * float(args.resolution_m) <= max_depth + 1e-12
            )
            image = _render(labels, band_mask)
            crop = _write_crop(
                output / f"{side_name}_rescan_band_{_slug(max_depth)}m.png",
                image,
                band_mask,
            )
            band["crop_grid_bounds"] = crop

    (output / "targeted_rescan_band_sweep.json").write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )

    print("output:", output)
    for side_name in ("entry", "exit"):
        print(f"{side_name}:")
        for band in payload["endpoint_rois"][side_name]["bands"]:
            classes = band["classes"]
            print(
                f"  depth<={band['max_outward_depth_m']:.2f}m "
                f"band_cells={band['roi_cell_count']} "
                f"rescan_required={band['rescan_required_cell_count']} "
                f"fraction_of_band={band['rescan_required_fraction_of_roi']:.6f} "
                f"fraction_of_endpoint={band['rescan_required_fraction_of_endpoint_roi']:.6f} "
                f"anchors={band['repeated_scan_anchor_cell_count']} "
                f"no_ground={classes['rescan_no_ground_reference']['count']} "
                f"no_observation={classes['rescan_ground_known_no_observation']['count']} "
                f"single_scan={classes['rescan_single_scan_revisit']['count']}"
            )
    print("automatic_band_selection: false")
    print("navigation_map_modified: false")
    print("semantic_promotion: false")


if __name__ == "__main__":
    main()
