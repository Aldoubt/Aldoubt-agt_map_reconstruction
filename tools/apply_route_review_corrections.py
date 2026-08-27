#!/usr/bin/env python3
"""Apply manual PCD review corrections and export a navigation-map bundle."""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from agt_map_reconstruction.maps.navigation_export import write_navigation_bundle
from agt_map_reconstruction.maps.review_corrections import apply_review_corrections, load_review


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-dir", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=float)
    parser.add_argument("--origin", type=float, nargs=3)
    args = parser.parse_args(argv)
    payload = json.loads((args.map_dir / "aisle_rectangles.json").read_text(encoding="utf-8"))
    labels = np.load(args.map_dir / "semantic_labels.npy", allow_pickle=False)
    scene = np.load(args.map_dir / "scene_mask.npy", allow_pickle=False)
    corrected, corrected_payload = apply_review_corrections(
        labels, scene, payload, load_review(args.review)
    )
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "semantic_labels.npy", corrected, allow_pickle=False)
    np.save(args.output / "scene_mask.npy", scene, allow_pickle=False)
    resolution = float(args.resolution or payload["resolution_m"])
    origin = tuple(args.origin or [*payload["origin_xy"], 0.0])
    for rectangle in corrected_payload["rectangles"]:
        rectangle["metric_polygon_xy"] = [
            [origin[0] + point[0] * resolution,
             origin[1] + point[1] * resolution]
            for point in rectangle["polygon_xy"]
        ]
    bundle = write_navigation_bundle(
        corrected, corrected_payload["rectangles"], args.output,
        resolution=resolution, origin=origin,
    )
    merged = dict(corrected_payload)
    merged.update({
        "resolution_m": resolution,
        "origin_xy": list(origin[:2]),
        "semantic_labels": payload.get("semantic_labels", {}),
        "ridge_rectangles": corrected_payload.get("ridge_rectangles", []),
        "wall_rectangles": payload.get("wall_rectangles", []),
        "boundary_polygon_metric_xy": payload.get("boundary_polygon_metric_xy", []),
        "review_file": str(args.review),
    })
    (args.output / "aisle_rectangles.json").write_text(
        json.dumps(merged, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "review_corrections.json").write_text(
        json.dumps({"changes": merged.get("review_corrections", [])}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "aisle_count": len(merged["rectangles"]),
        "changes": len(merged.get("review_corrections", [])),
        "pillar_as_free": bundle["validation"]["pillar_as_free_cell_count"],
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
