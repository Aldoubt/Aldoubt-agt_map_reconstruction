#!/usr/bin/env python3
"""Build enclosed non-HARD site interior from canonical PGM by border flood fill."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Treat canonical HARD occupied cells as impermeable barriers, flood-fill "
            "non-HARD cells from the map border with 4-connectivity, and retain only "
            "enclosed non-HARD cells as a site-interior candidate. No morphology or "
            "wall-gap closure is performed."
        )
    )
    parser.add_argument("--map", required=True)
    parser.add_argument("--output", required=True)
    return parser


def _read_pgm(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"failed to read PGM: {path}")
    return np.flipud(image).astype(np.uint8, copy=False)


def _blend(image, mask, color, alpha):
    overlay = image.copy()
    overlay[np.asarray(mask, dtype=bool)] = np.asarray(color, dtype=np.uint8)
    cv2.addWeighted(overlay, float(alpha), image, 1.0 - float(alpha), 0.0, dst=image)


def main():
    args = build_parser().parse_args()

    from agt_map_reconstruction.maps.site_interior_flood_fill import (
        build_site_interior_flood_fill,
    )

    map_path = Path(args.map).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    base = _read_pgm(map_path)
    result, masks = build_site_interior_flood_fill(base)
    result["sources"] = {"map": str(map_path)}
    mask_files = {
        "site_interior_nonhard": "site_interior_nonhard_mask.npy",
        "exterior_reachable_nonhard": "exterior_reachable_nonhard_mask.npy",
        "hard_barrier": "hard_barrier_mask.npy",
    }
    for name, filename in mask_files.items():
        np.save(output / filename, masks[name].astype(bool, copy=False))
    result["mask_files"] = mask_files

    json_path = output / "site_interior_flood_fill.json"
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    image = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    _blend(image, masks["exterior_reachable_nonhard"], (180, 0, 180), 0.24)
    _blend(image, masks["site_interior_nonhard"], (0, 180, 0), 0.22)
    display = np.flipud(image).copy()
    cv2.rectangle(display, (8, 8), (720, 116), (30, 30, 30), -1)
    legend = [
        ("green: enclosed non-HARD site interior candidate", (0, 220, 0)),
        ("magenta: non-HARD reachable from map border = physical exterior", (255, 0, 255)),
        ("black: HARD barrier; flood fill never crosses it", (220, 220, 220)),
        ("no morphology / no wall-gap closure / no semantic free promotion", (0, 200, 255)),
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
    cv2.imwrite(str(output / "site_interior_flood_fill.png"), display)

    print("output:", output)
    print("method:", result["method"])
    print("status:", result["status"])
    print("connectivity:", result["connectivity"])
    print("hard_cells:", result["hard_cell_count"])
    print("exterior_reachable_nonhard_cells:", result["exterior_reachable_nonhard_cell_count"])
    print("interior_nonhard_cells:", result["interior_nonhard_cell_count"])
    print("interior_components:", result["interior_component_count"])
    print("interior_component_sizes:", result["interior_component_sizes"])
    print("morphology_applied: false")
    print("automatic_wall_gap_closure: false")
    print("automatic_component_selection: false")
    print("navigation_map_modified: false")
    print("semantic_promotion: false")

    if result["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
