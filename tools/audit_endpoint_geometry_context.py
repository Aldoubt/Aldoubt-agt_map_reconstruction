#!/usr/bin/env python3
"""Audit raw row endpoints against clearance-conditioned handoff geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compare P1-D3 raw row-centerline endpoints against C-stage "
            "clearance-conditioned handoff cells and render both on the full map."
        )
    )
    parser.add_argument("--map", required=True)
    parser.add_argument("--row-band-regions", required=True)
    parser.add_argument("--handoffs", required=True)
    parser.add_argument("--output", required=True)
    return parser


def _read_grid_pgm(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"failed to read PGM: {path}")
    return np.flipud(image).astype(np.uint8, copy=False)


def _fit_segment(fit, row_axis, cross_axis, v_min, v_max):
    v = np.array([float(v_min), float(v_max)], dtype=np.float64)
    u = float(fit["slope_du_dv"]) * v + float(fit["intercept_u"])
    points = u[:, None] * row_axis[None, :] + v[:, None] * cross_axis[None, :]
    return np.rint(points).astype(np.int32)


def _draw_point(image, point, color, radius=4):
    x, y = np.rint(np.asarray(point, dtype=float)).astype(int)
    cv2.circle(image, (x, y), radius, color, -1, lineType=cv2.LINE_AA)


def _draw_line(image, p0, p1, color, thickness=2):
    a = tuple(np.rint(np.asarray(p0, dtype=float)).astype(int))
    b = tuple(np.rint(np.asarray(p1, dtype=float)).astype(int))
    cv2.line(image, a, b, color, thickness, lineType=cv2.LINE_AA)


def _write_crop(path, image_grid, points, pad=40):
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return None
    x0 = max(0, int(np.floor(np.min(pts[:, 0]))) - pad)
    x1 = min(image_grid.shape[1], int(np.ceil(np.max(pts[:, 0]))) + pad + 1)
    y0 = max(0, int(np.floor(np.min(pts[:, 1]))) - pad)
    y1 = min(image_grid.shape[0], int(np.ceil(np.max(pts[:, 1]))) + pad + 1)
    crop = image_grid[y0:y1, x0:x1]
    cv2.imwrite(str(path), np.flipud(crop))
    return {"x0": x0, "x1": x1, "y0": y0, "y1": y1}


def main():
    args = build_parser().parse_args()

    from agt_map_reconstruction.maps.endpoint_geometry_audit import audit_endpoint_geometry

    map_path = Path(args.map).expanduser().resolve()
    regions_path = Path(args.row_band_regions).expanduser().resolve()
    handoffs_path = Path(args.handoffs).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    base = _read_grid_pgm(map_path)
    regions_payload = json.loads(regions_path.read_text(encoding="utf-8"))
    handoffs_payload = json.loads(handoffs_path.read_text(encoding="utf-8"))
    grid = regions_payload.get("grid", {})
    resolution = float(grid["resolution"])
    expected_shape = (int(grid["height"]), int(grid["width"]))
    if base.shape != expected_shape:
        raise ValueError(f"map shape {base.shape} != region grid {expected_shape}")

    rows = [
        item for item in regions_payload.get("regions", [])
        if item.get("region_class") == "row_aisle"
    ]
    handoffs = list(handoffs_payload.get("handoffs", []))
    result = audit_endpoint_geometry(rows, handoffs, resolution_m=resolution)
    result["sources"] = {
        "map": str(map_path),
        "row_band_regions": str(regions_path),
        "handoffs": str(handoffs_path),
    }

    # Full-map context: keep the original navigation map visible.
    gray = np.clip(base.astype(np.int16), 0, 255).astype(np.uint8)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    eligible = set(result["eligible_row_labels"])
    row_by_label = {str(item.get("label", "")): item for item in rows}
    for label in result["eligible_row_labels"]:
        row = row_by_label[label]
        polygon = np.rint(np.asarray(row["polygon_xy"], dtype=float)).astype(np.int32)
        cv2.polylines(image, [polygon], True, (150, 150, 150), 1, lineType=cv2.LINE_AA)
        line = np.asarray(row["centerline_xy"], dtype=float)
        _draw_line(image, line[0], line[1], (160, 160, 160), 1)

    raw_entry_points = []
    raw_exit_points = []
    handoff_entry_points = []
    handoff_exit_points = []
    for item in result["rows"]:
        re = item["raw_entry_grid_xy"]
        rx = item["raw_exit_grid_xy"]
        he = item["handoff_entry_grid_xy"]
        hx = item["handoff_exit_grid_xy"]
        raw_entry_points.append(re)
        raw_exit_points.append(rx)
        handoff_entry_points.append(he)
        handoff_exit_points.append(hx)

        # Raw -> clearance handoff displacement.
        _draw_line(image, re, he, (255, 180, 0), 1)
        _draw_line(image, rx, hx, (255, 180, 0), 1)
        _draw_point(image, re, (255, 0, 255), 3)
        _draw_point(image, rx, (255, 0, 255), 3)
        _draw_point(image, he, (0, 220, 0), 4)
        _draw_point(image, hx, (0, 220, 0), 4)

    row_axis = np.asarray(result["row_axis_direction"], dtype=float)
    cross_axis = np.asarray(result["cross_row_direction"], dtype=float)
    v_min, v_max = result["row_cross_span"]

    for side in ("entry", "exit"):
        raw_seg = _fit_segment(
            result["raw_endpoint_fit"][side], row_axis, cross_axis, v_min, v_max
        )
        handoff_seg = _fit_segment(
            result["clearance_handoff_fit"][side], row_axis, cross_axis, v_min, v_max
        )
        _draw_line(image, raw_seg[0], raw_seg[1], (255, 0, 255), 2)
        _draw_line(image, handoff_seg[0], handoff_seg[1], (0, 220, 0), 2)

    # Legend, kept in grid orientation; final image is flipped below.
    cv2.rectangle(image, (8, 8), (330, 76), (40, 40, 40), -1)
    cv2.putText(image, "magenta: raw row endpoint / fit", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(image, "green: clearance handoff / fit", (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1, cv2.LINE_AA)
    cv2.putText(image, "cyan line: raw -> handoff offset", (16, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 180, 0), 1, cv2.LINE_AA)

    cv2.imwrite(str(output / "endpoint_geometry_context.png"), np.flipud(image))
    result["entry_crop_grid_bounds"] = _write_crop(
        output / "entry_endpoint_geometry_context.png",
        image,
        raw_entry_points + handoff_entry_points,
    )
    result["exit_crop_grid_bounds"] = _write_crop(
        output / "exit_endpoint_geometry_context.png",
        image,
        raw_exit_points + handoff_exit_points,
    )

    (output / "endpoint_geometry_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )

    print("output:", output)
    print("eligible_rows:", result["eligible_row_count"])
    for side in ("entry", "exit"):
        summary = result["offset_summary"][f"{side}_inward"]
        print(
            f"{side}_inward_offset_m: "
            f"median={summary['median_m']:.3f} "
            f"p95={summary['p95_m']:.3f} "
            f"max={summary['max_m']:.3f}"
        )
        raw_rmse = result["raw_endpoint_fit"][side]["residual_rmse_cells"] * resolution
        handoff_rmse = result["clearance_handoff_fit"][side]["residual_rmse_cells"] * resolution
        print(
            f"{side}_fit_rmse_m: raw={raw_rmse:.3f} clearance_handoff={handoff_rmse:.3f}"
        )
    print("d3_geometry_modified: false")
    print("navigation_map_modified: false")
    print("semantic_promotion: false")


if __name__ == "__main__":
    main()
