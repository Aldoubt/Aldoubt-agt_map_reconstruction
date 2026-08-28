import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

from agt_map_reconstruction.maps.navigation_export import UNKNOWN_VALUE


def _payloads(map_path):
    fused = {
        "row_axis_direction": [1.0, 0.0],
        "cross_row_direction": [0.0, 1.0],
        "resolution_m": 0.5,
        "lattice_rows": [
            {"label": "L1", "polygon_xy": [[0, 2], [29, 2], [29, 3], [0, 3]]},
            {"label": "L2", "polygon_xy": [[0, 6], [29, 6], [29, 7], [0, 7]]},
            {"label": "L3", "polygon_xy": [[0, 10], [29, 10], [29, 11], [0, 11]]},
        ],
        "ridge_profiles": [
            {"ridge_id": "R12", "ridge_cross_span_cells": [3.0, 6.0]},
            {"ridge_id": "R23", "ridge_cross_span_cells": [7.0, 10.0]},
        ],
        "ridge_terminations": [
            {"ridge_id": "R12", "status": "ok", "evidence_source": "pgm_hard"},
            {"ridge_id": "R23", "status": "insufficient_structural_support"},
        ],
        "sources": {"map": str(map_path)},
    }

    def side(intercept):
        return {
            "trend_status": "ok",
            "trend": {"slope_du_dv": 0.0, "intercept_u": float(intercept)},
            "abs_residual_m": {"p50": 0.5, "p90": 0.75, "p95": 1.0, "max": 1.5},
        }

    uncertainty = {
        "row_axis_direction": [1.0, 0.0],
        "cross_row_direction": [0.0, 1.0],
        "resolution_m": 0.5,
        "entry": side(10.0),
        "exit": side(20.0),
    }
    return fused, uncertainty


def test_headland_depth_profile_cli_writes_finite_geometry_bundle(tmp_path):
    map_path = tmp_path / "map.pgm"
    grid = np.full((14, 30), UNKNOWN_VALUE, dtype=np.uint8)
    cv2.imwrite(str(map_path), np.flipud(grid))

    fused, uncertainty = _payloads(map_path.resolve())
    fused_path = tmp_path / "structural_endpoint_fused.json"
    uncertainty_path = tmp_path / "structural_endpoint_uncertainty_fused.json"
    fused_path.write_text(json.dumps(fused), encoding="utf-8")
    uncertainty_path.write_text(json.dumps(uncertainty), encoding="utf-8")

    output = tmp_path / "out"
    script = Path(__file__).resolve().parents[1] / "tools" / "build_headland_depth_profile.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--fused-structural-bundle",
            str(fused_path),
            "--fused-uncertainty",
            str(uncertainty_path),
            "--depth-edges-m",
            "0",
            "0.5",
            "1",
            "2",
            "4",
            "--uncertainty-quantile",
            "p95",
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads((output / "headland_depth_profile.json").read_text(encoding="utf-8"))
    assert payload["method"] == "finite_structural_headland_depth_profile"
    assert payload["depth_edges_m"] == [0.0, 0.5, 1.0, 2.0, 4.0]
    assert payload["policy"]["physical_site_boundary_required"] is False
    assert payload["policy"]["hard_boundary_flood_fill_used"] is False
    assert payload["policy"]["navigation_map_modified"] is False
    assert payload["policy"]["semantic_promotion"] is False
    assert payload["sources"]["fused_structural_bundle"] == str(fused_path.resolve())
    assert payload["sources"]["fused_uncertainty"] == str(uncertainty_path.resolve())
    assert payload["sources"]["map"] == str(map_path.resolve())

    for mask_key, filename in payload["mask_files"].items():
        mask = np.load(output / filename)
        assert mask.shape == grid.shape, mask_key
        assert mask.dtype == np.bool_

    assert (output / "headland_depth_profile.png").exists()
    assert "max_outward_depth_m: 4.000000" in completed.stdout
    assert "physical_site_boundary_required: false" in completed.stdout
    assert "navigation_map_modified: false" in completed.stdout
