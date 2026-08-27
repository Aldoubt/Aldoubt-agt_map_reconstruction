#!/usr/bin/env python3
"""Build a conservative observation-sufficiency diagnostic layer for P1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Partition frozen navigation-map UNKNOWN cells into ground-reference and "
            "unique-scan observation-sufficiency classes. The canonical PGM is never modified."
        )
    )
    parser.add_argument("--navigation-map", required=True)
    parser.add_argument("--ground-reference", required=True)
    parser.add_argument("--scan-support-count", required=True)
    parser.add_argument("--baseline-envelope", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-repeated-scans", type=int, default=2)
    return parser


def _read_grid_pgm(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"failed to read PGM: {path}")
    return np.flipud(image).astype(np.uint8, copy=False)


def _render_labels(labels):
    from agt_map_reconstruction.maps.observation_sufficiency import (
        LABEL_KNOWN_FREE,
        LABEL_OCCUPIED,
        LABEL_UNKNOWN_NO_GROUND_REFERENCE,
        LABEL_UNKNOWN_NO_OBSERVATION,
        LABEL_UNKNOWN_REPEATED_SCAN,
        LABEL_UNKNOWN_SINGLE_SCAN,
    )

    image = np.zeros((*labels.shape, 3), dtype=np.uint8)
    # BGR palette. Colors are diagnostic only and have no navigation semantics.
    palette = {
        int(LABEL_OCCUPIED): (0, 0, 0),
        int(LABEL_KNOWN_FREE): (245, 245, 245),
        int(LABEL_UNKNOWN_NO_GROUND_REFERENCE): (180, 60, 180),
        int(LABEL_UNKNOWN_NO_OBSERVATION): (105, 105, 105),
        int(LABEL_UNKNOWN_SINGLE_SCAN): (0, 190, 255),
        int(LABEL_UNKNOWN_REPEATED_SCAN): (70, 210, 70),
    }
    for value, color in palette.items():
        image[labels == value] = color
    return image, palette


def _write_roi_view(path, image_grid, roi):
    ys, xs = np.nonzero(roi)
    if not ys.size:
        return None
    pad = 5
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(roi.shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(roi.shape[1], int(xs.max()) + pad + 1)
    crop = image_grid[y0:y1, x0:x1].copy()
    crop_roi = roi[y0:y1, x0:x1]
    crop[~crop_roi] = (25, 25, 25)
    # Convert repository lower-left grid orientation to image top-left orientation.
    cv2.imwrite(str(path), np.flipud(crop))
    return {"x0": x0, "x1": x1, "y0": y0, "y1": y1}


def main():
    args = build_parser().parse_args()

    from agt_map_reconstruction.maps.observation_sufficiency import (
        LABEL_NAMES,
        build_observation_sufficiency_labels,
        summarize_observation_sufficiency,
    )
    from agt_map_reconstruction.maps.ray_endpoint_support_diagnostics import _side_roi

    map_path = Path(args.navigation_map).expanduser().resolve()
    ground_path = Path(args.ground_reference).expanduser().resolve()
    support_path = Path(args.scan_support_count).expanduser().resolve()
    baseline_path = Path(args.baseline_envelope).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    base_map = _read_grid_pgm(map_path)
    ground = np.load(ground_path, allow_pickle=False)
    support = np.load(support_path, allow_pickle=False)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    labels = build_observation_sufficiency_labels(
        base_map,
        ground,
        support,
        min_repeated_scans=args.min_repeated_scans,
    )
    image, palette = _render_labels(labels)

    np.save(output / "observation_sufficiency_labels.npy", labels)
    cv2.imwrite(str(output / "observation_sufficiency.png"), np.flipud(image))

    summary = {
        "schema_version": 1,
        "min_repeated_scans": int(args.min_repeated_scans),
        "label_semantics": {str(key): value for key, value in LABEL_NAMES.items()},
        "palette_bgr": {str(key): list(value) for key, value in palette.items()},
        "full_map": summarize_observation_sufficiency(labels),
        "endpoint_rois": {},
        "sources": {
            "navigation_map": str(map_path),
            "ground_reference": str(ground_path),
            "scan_support_count": str(support_path),
            "baseline_envelope": str(baseline_path),
        },
        "policy": {
            "diagnostic_layer_only": True,
            "navigation_map_modified": False,
            "automatic_semantic_promotion": False,
            "semantic_promotion": False,
        },
    }

    for side_name in ("entry", "exit"):
        roi, _, _ = _side_roi(labels.shape, baseline, side_name)
        summary["endpoint_rois"][side_name] = summarize_observation_sufficiency(
            labels, roi_mask=roi
        )
        crop = _write_roi_view(output / f"{side_name}_observation_sufficiency.png", image, roi)
        summary["endpoint_rois"][side_name]["crop_grid_bounds"] = crop

    (output / "observation_sufficiency.json").write_text(
        json.dumps(summary, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )

    print("output:", output)
    print("min_repeated_scans:", args.min_repeated_scans)
    for scope_name, scope in [("full_map", summary["full_map"])] + list(
        summary["endpoint_rois"].items()
    ):
        print(f"{scope_name}: roi_cells={scope['roi_cell_count']} unknown={scope['unknown_cell_count']}")
        for class_name, metrics in scope["classes"].items():
            if class_name.startswith("unknown_"):
                print(
                    f"  {class_name}: count={metrics['count']} "
                    f"fraction_of_unknown={metrics.get('fraction_of_unknown', 0.0):.6f}"
                )
    print("navigation_map_modified: false")
    print("semantic_promotion: false")


if __name__ == "__main__":
    main()
