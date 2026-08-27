#!/usr/bin/env python3
"""Fuse PGM and targeted 3D ridge endpoints, then rebuild D3.1 uncertainty."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Fuse frozen PGM ridge terminations with endpoint-eligible targeted 3D "
            "ridge evidence and rebuild the uncertainty-preserving D3.1 envelope."
        )
    )
    parser.add_argument("--structural-bundle", required=True)
    parser.add_argument("--three-d-audit", required=True)
    parser.add_argument("--output", required=True)
    return parser


def _read_pgm(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"failed to read PGM: {path}")
    return np.flipud(image).astype(np.uint8, copy=False)


def _draw_trend_band(image, side_payload, axis, cross, cross_span, resolution, color):
    trend = side_payload.get("trend")
    if not trend:
        return
    v = np.asarray(cross_span, dtype=np.float64)
    u = float(trend["slope_du_dv"]) * v + float(trend["intercept_u"])
    center = u[:, None] * axis[None, :] + v[:, None] * cross[None, :]
    p95 = side_payload.get("abs_residual_m", {}).get("p95")
    if p95 is not None and float(p95) > 0.0:
        offset = float(p95) / float(resolution)
        polygon = np.vstack([center + offset * axis, (center - offset * axis)[::-1]])
        overlay = image.copy()
        cv2.fillPoly(overlay, [np.rint(polygon).astype(np.int32)], color)
        cv2.addWeighted(overlay, 0.12, image, 0.88, 0.0, dst=image)
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

    from agt_map_reconstruction.maps.structural_endpoint_fusion import (
        fuse_structural_endpoint_evidence,
    )
    from agt_map_reconstruction.maps.structural_endpoint_uncertainty import (
        build_structural_endpoint_uncertainty_envelope,
    )

    source_path = Path(args.structural_bundle).expanduser().resolve()
    audit_path = Path(args.three_d_audit).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    source = json.loads(source_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    fused = fuse_structural_endpoint_evidence(source, audit)
    fused["sources"] = {
        **dict(source.get("sources") or {}),
        "source_structural_bundle": str(source_path),
        "targeted_3d_audit": str(audit_path),
    }
    uncertainty = build_structural_endpoint_uncertainty_envelope(fused)
    uncertainty["sources"] = {
        "fused_structural_bundle": str(output / "structural_endpoint_fused.json")
    }

    (output / "structural_endpoint_fused.json").write_text(
        json.dumps(fused, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    (output / "structural_endpoint_uncertainty_fused.json").write_text(
        json.dumps(uncertainty, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    map_path = (source.get("sources") or {}).get("map")
    if map_path:
        base = _read_pgm(Path(map_path))
        image = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        profiles = {str(item["ridge_id"]): item for item in fused.get("ridge_profiles") or []}
        terminations = {str(item["ridge_id"]): item for item in fused.get("ridge_terminations") or []}

        color_by_source = {
            "pgm_hard": (0, 170, 0),
            "height_3d": (255, 180, 0),
            "unresolved_local": (0, 165, 255),
            "unresolved": (0, 0, 255),
        }
        for ridge_id, profile in profiles.items():
            termination = terminations.get(ridge_id, {})
            source_kind = str(termination.get("evidence_source", "unresolved"))
            if source_kind == "unresolved" and termination.get("local_3d_structure_observed"):
                source_kind = "unresolved_local"
            color = color_by_source[source_kind]
            centers = np.asarray(profile.get("bin_center_grid_xy"), dtype=np.float64)
            if centers.ndim == 2 and centers.shape[0] >= 2:
                cv2.polylines(
                    image,
                    [np.rint(centers).astype(np.int32)],
                    False,
                    color,
                    2,
                    lineType=cv2.LINE_AA,
                )
            if termination.get("status") == "ok":
                for side in ("entry", "exit"):
                    point = termination.get(f"{side}_grid_xy")
                    if point is not None:
                        x, y = np.rint(np.asarray(point, dtype=np.float64)).astype(int)
                        cv2.circle(image, (x, y), 4, color, -1, lineType=cv2.LINE_AA)

        axis = np.asarray(uncertainty["row_axis_direction"], dtype=np.float64)
        cross = np.asarray(uncertainty["cross_row_direction"], dtype=np.float64)
        cross_values = []
        for row in fused.get("lattice_rows") or []:
            line = np.asarray(row.get("centerline_xy"), dtype=np.float64)
            if line.shape == (2, 2):
                cross_values.extend((line @ cross).tolist())
        if len(cross_values) >= 2:
            span = [float(min(cross_values)), float(max(cross_values))]
            _draw_trend_band(
                image,
                uncertainty["entry"],
                axis,
                cross,
                span,
                uncertainty["resolution_m"],
                (0, 140, 255),
            )
            _draw_trend_band(
                image,
                uncertainty["exit"],
                axis,
                cross,
                span,
                uncertainty["resolution_m"],
                (255, 100, 0),
            )

        display = np.flipud(image).copy()
        cv2.rectangle(display, (8, 8), (720, 140), (30, 30, 30), -1)
        legend = [
            ("green: ridge endpoint from PGM HARD evidence", (0, 170, 0)),
            ("cyan: PGM-missing ridge endpoint recovered by endpoint-eligible 3D evidence", (255, 180, 0)),
            ("orange: local 3D structure observed, but not eligible for full-ridge endpoints", (0, 165, 255)),
            ("red: ridge endpoint remains structurally unresolved", (0, 0, 255)),
            ("entry/exit center trends + p95 bands remain uncertainty summaries only", (255, 220, 0)),
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
        cv2.imwrite(str(output / "structural_endpoint_fusion_context.png"), display)

    summary = fused["fusion_summary"]
    print("output:", output)
    print("method: pgm_plus_3d_structural_ridge_fusion")
    print("pgm_supported_ridges:", summary["pgm_supported_ridge_count"])
    print("three_d_supported_ridges:", summary["three_d_supported_ridge_count"])
    print("local_3d_only_ridges:", summary["local_3d_only_ridge_count"])
    print("unresolved_ridges:", summary["unresolved_ridge_count"])
    for side in ("entry", "exit"):
        item = uncertainty[side]
        q = item["abs_residual_m"]
        print(
            f"{side}: supported={item['supported_count']} "
            f"fraction={item['supported_fraction']:.6f} "
            f"cross_span_fraction={item['cross_row_span_fraction']:.6f} "
            f"p50={q['p50']} p95={q['p95']} max={q['max']}"
        )
        for source_name, source_q in item.get("abs_residual_m_by_evidence_source", {}).items():
            print(
                f"  {source_name}: count={source_q['count']} "
                f"p50={source_q['p50']} p95={source_q['p95']} max={source_q['max']}"
            )
    print("geometry_only_lattice_supplies_structural_evidence: false")
    print("local_3d_structure_promoted_to_endpoint_support: false")
    print("automatic_acceptance: false")
    print("navigation_map_modified: false")
    print("semantic_promotion: false")


if __name__ == "__main__":
    main()
