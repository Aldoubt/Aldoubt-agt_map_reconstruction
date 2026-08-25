#!/usr/bin/env python3
"""Export segmentation masks as a semantic PGM and editable aisle rectangles."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from agt_map_reconstruction.maps.navigation_export import export_navigation_assets


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("segmentation_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--origin-x", type=float, default=0.0)
    parser.add_argument("--origin-y", type=float, default=0.0)
    args = parser.parse_args(argv)
    labels = np.load(args.segmentation_dir / "labels.npy", allow_pickle=False)
    scene = np.load(args.segmentation_dir / "scene_mask.npy", allow_pickle=False)
    metrics = json.loads((args.segmentation_dir / "metrics.json").read_text(encoding="utf-8"))
    result = {
        "labels": labels,
        "scene_mask": scene,
        "row_angle_rad": metrics["row_angle_rad"],
        "rows": metrics["rows"],
        "aisles": metrics["aisles"],
        "config": metrics["config"],
    }
    payload = export_navigation_assets(
        result, args.output,
        origin_xy=(args.origin_x, args.origin_y),
    )
    rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
    rgb[labels == 0] = (40, 40, 40)
    rgb[labels == 1] = (70, 160, 255)
    rgb[labels == 2] = (235, 185, 45)
    rgb[labels == 3] = (225, 55, 55)
    rgb[labels == 4] = (255, 80, 80)
    rgb[~scene] = (0, 0, 0)
    figure, axis = plt.subplots(figsize=(12, 10))
    axis.imshow(rgb, origin="lower", interpolation="nearest")
    axis.set_title(f"semantic navigation map — aisles: {payload['aisle_count']}")
    for rectangle in payload["rectangles"]:
        polygon = np.asarray(rectangle["polygon_xy"] + [rectangle["polygon_xy"][0]])
        axis.plot(polygon[:, 0], polygon[:, 1], color="white", linewidth=0.8)
        center = polygon[:-1].mean(axis=0)
        axis.text(center[0], center[1], str(rectangle["aisle_id"]), color="white", fontsize=7)
    for rectangle in payload.get("ridge_rectangles", []):
        polygon = np.asarray(rectangle["polygon_xy"] + [rectangle["polygon_xy"][0]])
        axis.plot(polygon[:, 0], polygon[:, 1], color="yellow", linewidth=0.8)
        center = polygon[:-1].mean(axis=0)
        axis.text(center[0], center[1], rectangle["label"], color="yellow", fontsize=7)
    for rectangle in payload.get("wall_rectangles", []):
        polygon = np.asarray(rectangle["polygon_xy"] + [rectangle["polygon_xy"][0]])
        axis.plot(polygon[:, 0], polygon[:, 1], color="red", linewidth=1.2)
    axis.contour(scene, levels=[0.5], colors="red", linewidths=1.5)
    figure.savefig(args.output / "navigation_semantic_overlay.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps({"output": str(args.output), "aisle_count": payload["aisle_count"]}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
