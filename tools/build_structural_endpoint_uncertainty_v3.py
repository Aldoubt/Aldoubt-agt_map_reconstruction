#!/usr/bin/env python3
"""Build an uncertainty-preserving D3.1 structural endpoint envelope."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Preserve every supported ridge termination and summarize a robust "
            "center trend plus residual uncertainty envelope."
        )
    )
    parser.add_argument("--structural-bundle", required=True)
    parser.add_argument("--output", required=True)
    return parser


def _read_pgm(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"failed to read PGM: {path}")
    return np.flipud(image).astype(np.uint8, copy=False)


def _draw_line_from_trend(image, side_payload, axis, cross, cross_span, resolution, color):
    trend = side_payload.get("trend")
    if not trend:
        return
    v = np.asarray(cross_span, dtype=np.float64)
    u = float(trend["slope_du_dv"]) * v + float(trend["intercept_u"])
    center = u[:, None] * axis[None, :] + v[:, None] * cross[None, :]

    p95 = side_payload.get("abs_residual_m", {}).get("p95")
    if p95 is not None and float(p95) > 0.0:
        offset_cells = float(p95) / float(resolution)
        upper = center + offset_cells * axis[None, :]
        lower = center - offset_cells * axis[None, :]
        polygon = np.vstack([upper, lower[::-1]])
        overlay = image.copy()
        cv2.fillPoly(overlay, [np.rint(polygon).astype(np.int32)], color)
        cv2.addWeighted(overlay, 0.16, image, 0.84, 0.0, dst=image)

    cv2.line(
        image,
        tuple(np.rint(center[0]).astype(int)),
        tuple(np.rint(center[1]).astype(int)),
        color,
        2,
        lineType=cv2.LINE_AA,
    )


def main():
    args = build_parser().parse_args()

    from agt_map_reconstruction.maps.structural_endpoint_uncertainty import (
        build_structural_endpoint_uncertainty_envelope,
    )

    source_path = Path(args.structural_bundle).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    bundle = json.loads(source_path.read_text(encoding="utf-8"))
    result = build_structural_endpoint_uncertainty_envelope(bundle)
    result["sources"] = {"structural_bundle": str(source_path)}

    json_path = output / "structural_endpoint_uncertainty_v3.json"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    map_path = (bundle.get("sources") or {}).get("map")
    if map_path:
        base = _read_pgm(Path(map_path))
        image = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        axis = np.asarray(result["row_axis_direction"], dtype=np.float64)
        cross = np.asarray(result["cross_row_direction"], dtype=np.float64)

        rows = bundle.get("lattice_rows") or []
        cross_values = []
        for row in rows:
            line = np.asarray(row.get("centerline_xy"), dtype=np.float64)
            if line.shape == (2, 2):
                cross_values.extend((line @ cross).tolist())
                source = str(row.get("geometry_source", ""))
                color = (255, 220, 0) if source == "lattice_inferred_wide_band" else (0, 180, 0)
                cv2.line(
                    image,
                    tuple(np.rint(line[0]).astype(int)),
                    tuple(np.rint(line[1]).astype(int)),
                    color,
                    1,
                    lineType=cv2.LINE_AA,
                )
        if len(cross_values) >= 2:
            cross_span = [float(min(cross_values)), float(max(cross_values))]
            _draw_line_from_trend(
                image,
                result["entry"],
                axis,
                cross,
                cross_span,
                result["resolution_m"],
                (0, 140, 255),
            )
            _draw_line_from_trend(
                image,
                result["exit"],
                axis,
                cross,
                cross_span,
                result["resolution_m"],
                (255, 100, 0),
            )

        for point in result["entry"]["ridge_points"]:
            x, y = np.rint(point["grid_xy"]).astype(int)
            cv2.circle(image, (x, y), 4, (0, 165, 255), -1, lineType=cv2.LINE_AA)
        for point in result["exit"]["ridge_points"]:
            x, y = np.rint(point["grid_xy"]).astype(int)
            cv2.circle(image, (x, y), 4, (255, 100, 0), -1, lineType=cv2.LINE_AA)

        display = np.flipud(image).copy()
        cv2.rectangle(display, (8, 8), (570, 92), (30, 30, 30), -1)
        legend = [
            ("orange: all supported entry ridge terminations + p95 band", (0, 165, 255)),
            ("blue: all supported exit ridge terminations + p95 band", (255, 100, 0)),
            ("cyan lattice remains geometry-only; center trend is not semantic free", (255, 220, 0)),
        ]
        for index, (text, color) in enumerate(legend):
            cv2.putText(
                display,
                text,
                (18, 30 + 24 * index),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )
        cv2.imwrite(str(output / "structural_endpoint_uncertainty_v3.png"), display)

    print("output:", output)
    print("method: ridge_termination_uncertainty_envelope")
    print("ridge_count:", result["ridge_count"])
    print("supported_ridges:", result["supported_ridge_count"])
    print("unsupported_ridges:", result["unsupported_ridge_count"])
    for side in ("entry", "exit"):
        item = result[side]
        print(
            f"{side}: supported={item['supported_count']} "
            f"fraction={item['supported_fraction']:.6f} "
            f"cross_span_fraction={item['cross_row_span_fraction']:.6f}"
        )
        q = item["abs_residual_m"]
        print(
            f"  abs_residual_m: p50={q['p50']} p90={q['p90']} "
            f"p95={q['p95']} max={q['max']}"
        )
    print("ridge_outliers_deleted: false")
    print("bilateral_agreement_required_for_envelope: false")
    print("automatic_acceptance: false")
    print("navigation_map_modified: false")
    print("semantic_promotion: false")


if __name__ == "__main__":
    main()
