#!/usr/bin/env python3
"""Build anchor-validated enclosed non-HARD site interior by border flood fill."""

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
            "non-HARD cells from the map border with 4-connectivity, and validate the "
            "result against observed row-lattice centerline-midpoint anchors. No "
            "morphology or wall-gap closure is performed."
        )
    )
    parser.add_argument("--map", required=True)
    parser.add_argument(
        "--row-lattice-completion",
        required=True,
        help=(
            "frozen row_lattice_completion.json; observed slot midpoints are used "
            "only as trusted site-interior validation anchors"
        ),
    )
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


def _observed_slot_anchor_mask(lattice_payload, shape):
    anchors = np.zeros(shape, dtype=bool)
    selected = []
    slots = list(lattice_payload.get("slots") or [])
    for slot in slots:
        source = str(slot.get("source", ""))
        if source not in {"observed_row_aisle", "observed_split_group"}:
            continue
        line = np.asarray(slot.get("centerline_xy"), dtype=np.float64)
        if line.shape != (2, 2) or not np.isfinite(line).all():
            raise ValueError(
                f"observed lattice slot {slot.get('slot_id')} has invalid centerline_xy"
            )
        midpoint = np.mean(line, axis=0)
        x = int(np.rint(midpoint[0]))
        y = int(np.rint(midpoint[1]))
        if not (0 <= y < shape[0] and 0 <= x < shape[1]):
            raise ValueError(
                f"observed lattice slot {slot.get('slot_id')} midpoint is outside map"
            )
        anchors[y, x] = True
        selected.append(
            {
                "slot_id": str(slot.get("slot_id", "")),
                "source": source,
                "grid_xy": [x, y],
            }
        )
    if not selected:
        raise ValueError("row lattice contains no observed slots for interior validation")
    return anchors, selected


def main():
    args = build_parser().parse_args()

    from agt_map_reconstruction.maps.site_interior_flood_fill import (
        build_site_interior_flood_fill,
    )

    map_path = Path(args.map).expanduser().resolve()
    lattice_path = Path(args.row_lattice_completion).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    base = _read_pgm(map_path)
    lattice = json.loads(lattice_path.read_text(encoding="utf-8"))
    anchor_mask, anchor_slots = _observed_slot_anchor_mask(lattice, base.shape)
    result, masks = build_site_interior_flood_fill(
        base,
        interior_anchor_mask=anchor_mask,
    )
    result["sources"] = {
        "map": str(map_path),
        "row_lattice_completion": str(lattice_path),
    }
    result["interior_anchor_slots"] = anchor_slots
    result["interior_anchor_source"] = "observed_row_lattice_centerline_midpoints"
    mask_files = {
        "site_interior_nonhard": "site_interior_nonhard_mask.npy",
        "exterior_reachable_nonhard": "exterior_reachable_nonhard_mask.npy",
        "hard_barrier": "hard_barrier_mask.npy",
        "interior_anchor": "interior_anchor_mask.npy",
        "leak_path": "leak_path_mask.npy",
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
    image[masks["interior_anchor"]] = np.asarray((255, 255, 0), dtype=np.uint8)
    image[masks["leak_path"]] = np.asarray((0, 0, 255), dtype=np.uint8)
    display = np.flipud(image).copy()
    cv2.rectangle(display, (8, 8), (770, 164), (30, 30, 30), -1)
    legend = [
        ("green: enclosed non-HARD site interior candidate", (0, 220, 0)),
        ("magenta: non-HARD reachable from map border", (255, 0, 255)),
        ("black: HARD barrier; flood fill never crosses it", (220, 220, 220)),
        ("cyan: observed row-lattice midpoint = trusted interior validation anchor", (255, 255, 0)),
        ("red: one leaked-anchor parent path to map border", (0, 0, 255)),
        ("anchors validate only; no morphology / wall-gap closure / semantic free", (0, 200, 255)),
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
    print("status_basis:", result["status_basis"])
    print("connectivity:", result["connectivity"])
    print("hard_cells:", result["hard_cell_count"])
    print("exterior_reachable_nonhard_cells:", result["exterior_reachable_nonhard_cell_count"])
    print("interior_nonhard_cells:", result["interior_nonhard_cell_count"])
    print("interior_components:", result["interior_component_count"])
    print("interior_component_sizes:", result["interior_component_sizes"])
    print("interior_anchor_cells:", result["interior_anchor_cell_count"])
    print("interior_anchor_hard_cells:", result["interior_anchor_hard_cell_count"])
    print(
        "interior_anchor_exterior_reachable_cells:",
        result["interior_anchor_exterior_reachable_cell_count"],
    )
    print("interior_anchor_enclosed_cells:", result["interior_anchor_enclosed_cell_count"])
    print(
        "interior_anchor_validation_passed:",
        str(bool(result["interior_anchor_validation_passed"])).lower(),
    )
    print("leak_anchor_xy:", result["leak_anchor_xy"])
    print("leak_path_cells:", len(result["leak_path_xy"]))
    print("morphology_applied: false")
    print("automatic_wall_gap_closure: false")
    print("automatic_component_selection: false")
    print("interior_anchor_used_to_construct_mask: false")
    print("navigation_map_modified: false")
    print("semantic_promotion: false")

    if result["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
