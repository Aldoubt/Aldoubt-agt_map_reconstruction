#!/usr/bin/env python3
"""Build endpoint targeted-rescan requirement assets from sufficiency labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Convert observation-sufficiency labels into endpoint acquisition "
            "requirements using the exact frozen P1-D3 entry/exit ROIs."
        )
    )
    parser.add_argument("--sufficiency-labels", required=True)
    parser.add_argument("--baseline-envelope", required=True)
    parser.add_argument("--output", required=True)
    return parser


def _render(labels):
    from agt_map_reconstruction.maps.targeted_rescan_requirement import (
        RESCAN_CLASS_NAMES,
        RESCAN_KNOWN_FREE,
        RESCAN_NO_GROUND_REFERENCE,
        RESCAN_NO_OBSERVATION,
        RESCAN_OCCUPIED,
        RESCAN_OUTSIDE_ENDPOINT,
        RESCAN_REPEATED_SCAN_ANCHOR,
        RESCAN_SINGLE_SCAN_REVISIT,
    )

    # BGR; diagnostic palette only.
    palette = {
        int(RESCAN_OUTSIDE_ENDPOINT): (25, 25, 25),
        int(RESCAN_OCCUPIED): (0, 0, 0),
        int(RESCAN_KNOWN_FREE): (245, 245, 245),
        int(RESCAN_REPEATED_SCAN_ANCHOR): (70, 210, 70),
        int(RESCAN_SINGLE_SCAN_REVISIT): (0, 190, 255),
        int(RESCAN_NO_OBSERVATION): (40, 90, 230),
        int(RESCAN_NO_GROUND_REFERENCE): (180, 60, 180),
    }
    image = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for value, color in palette.items():
        image[labels == value] = color
    return image, palette, RESCAN_CLASS_NAMES


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
    from agt_map_reconstruction.maps.targeted_rescan_requirement import (
        build_targeted_rescan_requirement,
        summarize_targeted_rescan_requirement,
    )

    sufficiency_path = Path(args.sufficiency_labels).expanduser().resolve()
    baseline_path = Path(args.baseline_envelope).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    sufficiency = np.load(sufficiency_path, allow_pickle=False)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    rois = {}
    union_roi = np.zeros(sufficiency.shape, dtype=bool)
    for side_name in ("entry", "exit"):
        roi, _, _ = _side_roi(sufficiency.shape, baseline, side_name)
        rois[side_name] = roi
        union_roi |= roi

    labels = build_targeted_rescan_requirement(sufficiency, union_roi)
    image, palette, names = _render(labels)

    np.save(output / "targeted_rescan_requirement_labels.npy", labels)
    cv2.imwrite(str(output / "targeted_rescan_requirement.png"), np.flipud(image))

    payload = {
        "schema_version": 1,
        "label_semantics": {str(k): v for k, v in names.items()},
        "palette_bgr": {str(k): list(v) for k, v in palette.items()},
        "endpoint_union": summarize_targeted_rescan_requirement(labels, roi_mask=union_roi),
        "endpoint_rois": {},
        "sources": {
            "observation_sufficiency_labels": str(sufficiency_path),
            "baseline_envelope": str(baseline_path),
        },
        "policy": {
            "diagnostic_acquisition_requirement_only": True,
            "automatic_target_selection": False,
            "navigation_map_modified": False,
            "semantic_promotion": False,
        },
    }

    for side_name, roi in rois.items():
        payload["endpoint_rois"][side_name] = summarize_targeted_rescan_requirement(
            labels, roi_mask=roi
        )
        payload["endpoint_rois"][side_name]["crop_grid_bounds"] = _write_crop(
            output / f"{side_name}_targeted_rescan_requirement.png", image, roi
        )

    (output / "targeted_rescan_requirement.json").write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )

    print("output:", output)
    for side_name in ("entry", "exit"):
        side = payload["endpoint_rois"][side_name]
        print(
            f"{side_name}: roi_cells={side['roi_cell_count']} "
            f"rescan_required={side['rescan_required_cell_count']} "
            f"fraction={side['rescan_required_fraction_of_roi']:.6f} "
            f"repeated_anchor={side['repeated_scan_anchor_cell_count']}"
        )
        for class_name, metrics in side["classes"].items():
            if class_name.startswith("rescan_"):
                print(f"  {class_name}: {metrics['count']}")
        comp = side["rescan_required_components"]
        print(
            f"  components={comp['component_count']} "
            f"largest_component={comp['largest_component_cell_count']}"
        )
    print("automatic_target_selection: false")
    print("navigation_map_modified: false")
    print("semantic_promotion: false")


if __name__ == "__main__":
    main()
