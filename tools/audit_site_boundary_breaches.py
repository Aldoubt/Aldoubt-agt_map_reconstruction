#!/usr/bin/env python3
"""Audit border-reachable HARD-boundary breaches from observed row-lattice anchors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Trace every observed row-lattice midpoint anchor that is reachable from "
            "the map border through non-HARD cells. Overlapping parent paths localize "
            "common breach bottlenecks. The audit never repairs walls or modifies the map."
        )
    )
    parser.add_argument("--map", required=True)
    parser.add_argument("--row-lattice-completion", required=True)
    parser.add_argument("--resolution-m", required=True, type=float)
    parser.add_argument("--output", required=True)
    return parser


def _read_pgm(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"failed to read PGM: {path}")
    return np.flipud(image).astype(np.uint8, copy=False)


def _observed_anchor_records(payload, shape):
    records = []
    height, width = shape
    for slot in list(payload.get("slots") or []):
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
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(
                f"observed lattice slot {slot.get('slot_id')} midpoint lies outside map"
            )
        records.append(
            {
                "slot_id": str(slot.get("slot_id", "")),
                "source": source,
                "grid_xy": [x, y],
            }
        )
    if not records:
        raise ValueError("row lattice contains no observed slots")
    return records


def _blend(image, mask, color, alpha):
    overlay = image.copy()
    overlay[np.asarray(mask, dtype=bool)] = np.asarray(color, dtype=np.uint8)
    cv2.addWeighted(overlay, float(alpha), image, 1.0 - float(alpha), 0.0, dst=image)


def main():
    args = build_parser().parse_args()

    from agt_map_reconstruction.maps.site_boundary_breach_audit import (
        audit_site_boundary_breaches,
    )

    map_path = Path(args.map).expanduser().resolve()
    lattice_path = Path(args.row_lattice_completion).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    base = _read_pgm(map_path)
    lattice = json.loads(lattice_path.read_text(encoding="utf-8"))
    anchors = _observed_anchor_records(lattice, base.shape)
    result, masks = audit_site_boundary_breaches(
        base,
        anchors,
        resolution_m=args.resolution_m,
    )
    result["sources"] = {
        "map": str(map_path),
        "row_lattice_completion": str(lattice_path),
    }

    mask_files = {
        "anchor": "anchor_mask.npy",
        "hard_anchor": "hard_anchor_mask.npy",
        "leaked_paths": "leaked_paths_mask.npy",
        "path_support_count": "path_support_count.npy",
        "max_path_support": "max_path_support_mask.npy",
        "exterior_reachable_nonhard": "exterior_reachable_nonhard_mask.npy",
        "hard_barrier": "hard_barrier_mask.npy",
    }
    for name, filename in mask_files.items():
        np.save(output / filename, masks[name])
    result["mask_files"] = mask_files
    (output / "site_boundary_breach_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    image = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    _blend(image, masks["exterior_reachable_nonhard"], (180, 0, 180), 0.10)
    image[masks["leaked_paths"]] = np.asarray((0, 0, 255), dtype=np.uint8)
    image[masks["anchor"]] = np.asarray((255, 255, 0), dtype=np.uint8)
    image[masks["hard_anchor"]] = np.asarray((0, 165, 255), dtype=np.uint8)
    image[masks["max_path_support"]] = np.asarray((255, 255, 255), dtype=np.uint8)

    display = np.flipud(image).copy()
    cv2.rectangle(display, (8, 8), (790, 164), (30, 30, 30), -1)
    legend = [
        ("cyan: observed row-lattice midpoint anchor", (255, 255, 0)),
        ("orange: anchor lies on HARD; reported separately and never shifted", (0, 165, 255)),
        ("red: border-reachable leak paths for all leaked anchors", (0, 0, 255)),
        ("white: cells with maximum overlap among leaked-anchor paths", (255, 255, 255)),
        ("magenta tint: border-reachable non-HARD", (255, 0, 255)),
        ("diagnostic only: no wall closure / doorway classification / semantic promotion", (0, 220, 255)),
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
    cv2.imwrite(str(output / "site_boundary_breach_audit.png"), display)

    print("output:", output)
    print("method:", result["method"])
    print("status:", result["status"])
    print("anchors:", result["anchor_count"])
    print("leaked_anchors:", result["leaked_anchor_count"])
    print("hard_anchors:", result["hard_anchor_count"])
    print("enclosed_anchors:", result["enclosed_anchor_count"])
    print("unique_border_exits:", result["unique_border_exit_count"])
    print("border_exit_side_counts:", result["border_exit_side_counts"])
    print("path_length_m:", result["path_length_m"])
    print("max_path_support_count:", result["max_path_support_count"])
    print(
        "max_path_support_fraction_of_leaked_anchors:",
        f"{result['max_path_support_fraction_of_leaked_anchors']:.6f}",
    )
    print("max_path_support_cells_xy:", result["max_path_support_cells_xy"][:20])
    for record in result["anchors"]:
        print(
            f"{record['slot_id']}: {record['classification']} "
            f"grid_xy={record['grid_xy']} path_cells={record['path_cell_count']} "
            f"path_length_m={record['path_length_m']:.3f} "
            f"border_exit={record['border_exit_xy']} side={record['border_exit_side']}"
        )
    print("automatic_wall_gap_closure: false")
    print("automatic_boundary_repair: false")
    print("automatic_doorway_classification: false")
    print("anchor_on_hard_auto_shifted: false")
    print("navigation_map_modified: false")
    print("semantic_promotion: false")


if __name__ == "__main__":
    main()
