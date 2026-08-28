import json
from pathlib import Path
import os
import subprocess
import sys

import cv2
import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _api():
    try:
        from agt_map_reconstruction.maps.grid_geometry import GridMetadata
        from agt_map_reconstruction.maps.planner_request_footprint_preflight import (
            audit_planner_request_footprints,
            write_planner_request_footprint_preflight_bundle,
        )
    except ImportError as exc:
        pytest.fail(f"P1-G1.1 planner request footprint preflight API is missing: {exc}")
    return GridMetadata, audit_planner_request_footprints, write_planner_request_footprint_preflight_bundle


def _pose(x, y, yaw=0.0):
    return {"x": float(x), "y": float(y), "yaw": float(yaw)}


def _planner_test(case_id, *, enabled, forward, reverse):
    pair_id, side = case_id.rsplit("-", 1)
    return {
        "id": case_id,
        "pair_id": pair_id,
        "side": side,
        "radius_m": 0.2,
        "enabled": bool(enabled),
        "baseline_connected": bool(enabled),
        "conservative_connected": bool(enabled),
        "gained_by_trusted_overlay": False,
        "forward": forward,
        "reverse": reverse,
    }


def _payloads():
    positive = _planner_test(
        "A01-A02-entry",
        enabled=True,
        forward={"start": _pose(1.05, 1.05), "goal": _pose(2.05, 1.05)},
        reverse={"start": _pose(2.05, 1.05), "goal": _pose(1.05, 1.05)},
    )
    negative = _planner_test(
        "A02-A03-exit",
        enabled=False,
        forward={"start": _pose(1.05, 2.05), "goal": _pose(2.05, 2.05)},
        reverse={"start": _pose(2.05, 2.05), "goal": _pose(1.05, 2.05)},
    )
    ignored = _planner_test(
        "A03-A04-entry",
        enabled=False,
        forward={"start": _pose(3.05, 1.05), "goal": _pose(3.05, 2.05)},
        reverse={"start": _pose(3.05, 2.05), "goal": _pose(3.05, 1.05)},
    )
    planner_pairs = {
        "schema_version": 1,
        "method": "nav2_headland_adjacent_pair_smoke_tests",
        "radius_m": 0.2,
        "bidirectional": True,
        "tests": [positive, negative, ignored],
    }
    diagnostics = {
        "schema_version": 2,
        "method": "scoped_headland_gap_diagnostics",
        "radius_m": 0.2,
        "records": [
            {
                "pair_id": "A02-A03",
                "side": "exit",
                "evaluation_status": "evaluated",
                "strict_connected": False,
                "bridge_type": "mixed_bridge",
            },
            {
                "pair_id": "A03-A04",
                "side": "entry",
                "evaluation_status": "evaluated",
                "strict_connected": False,
                "bridge_type": "not_available",
            },
        ],
    }
    return planner_pairs, diagnostics


def _footprint():
    return np.asarray(
        [[0.15, 0.15], [0.15, -0.15], [-0.15, -0.15], [-0.15, 0.15]],
        dtype=float,
    )


def _grid():
    # Internal repository convention: y-up array coordinates.
    grid = np.full((40, 40), 254, dtype=np.uint8)
    # Positive pair target at world (2.05, 1.05) overlaps UNKNOWN only.
    grid[9:13, 19:23] = 205
    # Negative pair target at world (2.05, 2.05) overlaps OCCUPIED only.
    grid[19:23, 19:23] = 0
    return grid


def test_preflight_expands_same_positive_and_diagnostic_negative_requests_as_runtime():
    GridMetadata, audit, _ = _api()
    planner_pairs, diagnostics = _payloads()
    metadata = GridMetadata(0.1, 0.0, 0.0, 40, 40)

    result = audit(
        _grid(),
        planner_pairs,
        diagnostics,
        _footprint(),
        metadata,
        footprint_name="test_robot",
    )

    assert result["summary"]["request_count"] == 4
    assert result["summary"]["pose_count"] == 8
    assert {item["request_id"] for item in result["requests"]} == {
        "A01-A02-entry-forward",
        "A01-A02-entry-reverse",
        "A02-A03-exit-forward",
        "A02-A03-exit-reverse",
    }
    assert all("A03-A04" not in item["request_id"] for item in result["requests"])


def test_preflight_reports_start_and_goal_polygon_blocking_sources_separately():
    GridMetadata, audit, _ = _api()
    planner_pairs, diagnostics = _payloads()
    metadata = GridMetadata(0.1, 0.0, 0.0, 40, 40)

    result = audit(
        _grid(),
        planner_pairs,
        diagnostics,
        _footprint(),
        metadata,
        footprint_name="test_robot",
    )
    by_id = {item["request_id"]: item for item in result["requests"]}

    forward_positive = by_id["A01-A02-entry-forward"]
    assert forward_positive["start_preflight"]["pose_class"] == "valid"
    assert forward_positive["goal_preflight"]["pose_class"] == "unknown_overlap"
    assert forward_positive["goal_preflight"]["unknown_overlap_cell_count"] > 0
    assert forward_positive["goal_preflight"]["occupied_overlap_cell_count"] == 0
    assert forward_positive["request_valid"] is False

    reverse_positive = by_id["A01-A02-entry-reverse"]
    assert reverse_positive["start_preflight"]["pose_class"] == "unknown_overlap"
    assert reverse_positive["goal_preflight"]["pose_class"] == "valid"

    forward_negative = by_id["A02-A03-exit-forward"]
    assert forward_negative["goal_preflight"]["pose_class"] == "occupied_overlap"
    assert forward_negative["goal_preflight"]["occupied_overlap_cell_count"] > 0
    assert forward_negative["goal_preflight"]["unknown_overlap_cell_count"] == 0

    summary = result["summary"]
    assert summary["request_valid_count"] == 0
    assert summary["start_invalid_request_count"] == 2
    assert summary["goal_invalid_request_count"] == 2
    assert summary["pose_class_counts"] == {
        "valid": 4,
        "unknown_overlap": 2,
        "occupied_overlap": 2,
        "mixed_blocking_overlap": 0,
        "out_of_bounds": 0,
    }


def test_preflight_preserves_mixed_and_out_of_bounds_instead_of_hiding_them():
    GridMetadata, audit, _ = _api()
    planner_pairs, diagnostics = _payloads()
    planner_pairs["tests"][0]["forward"]["start"] = _pose(0.01, 0.01)
    planner_pairs["tests"][0]["forward"]["goal"] = _pose(2.05, 1.05)
    grid = _grid()
    grid[9:13, 19:23] = 205
    grid[10:12, 20:22] = 0
    metadata = GridMetadata(0.1, 0.0, 0.0, 40, 40)

    result = audit(grid, planner_pairs, diagnostics, _footprint(), metadata)
    item = next(r for r in result["requests"] if r["request_id"] == "A01-A02-entry-forward")

    assert item["start_preflight"]["pose_class"] == "out_of_bounds"
    assert item["goal_preflight"]["pose_class"] == "mixed_blocking_overlap"
    assert item["goal_preflight"]["occupied_overlap_cell_count"] > 0
    assert item["goal_preflight"]["unknown_overlap_cell_count"] > 0


def test_preflight_bundle_rejects_nonempty_output_and_writes_json_csv(tmp_path):
    GridMetadata, audit, write_bundle = _api()
    planner_pairs, diagnostics = _payloads()
    metadata = GridMetadata(0.1, 0.0, 0.0, 40, 40)
    result = audit(_grid(), planner_pairs, diagnostics, _footprint(), metadata)

    output = tmp_path / "preflight"
    paths = write_bundle(result, output)
    assert {path.name for path in output.iterdir()} == {
        "planner_request_footprint_preflight.json",
        "planner_request_footprint_preflight.csv",
    }
    assert set(paths) == {"json", "csv"}
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["summary"]["pose_count"] == 8

    with pytest.raises(FileExistsError, match="not empty"):
        write_bundle(result, output)


def test_preflight_cli_reads_frozen_artifacts_without_modifying_map(tmp_path):
    grid = _grid()
    pgm = tmp_path / "navigation_base_map.pgm"
    cv2.imwrite(str(pgm), np.flipud(grid))
    map_yaml = tmp_path / "navigation_base_map.yaml"
    map_yaml.write_text(
        yaml.safe_dump(
            {
                "image": pgm.name,
                "mode": "trinary",
                "resolution": 0.1,
                "origin": [0.0, 0.0, 0.0],
                "negate": 0,
                "occupied_thresh": 0.65,
                "free_thresh": 0.196,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    planner_pairs, diagnostics = _payloads()
    pairs_path = tmp_path / "planner_pairs.yaml"
    pairs_path.write_text(yaml.safe_dump(planner_pairs, sort_keys=False), encoding="utf-8")
    gap_path = tmp_path / "headland_gap_diagnostics.json"
    gap_path.write_text(json.dumps(diagnostics), encoding="utf-8")
    footprint_path = tmp_path / "mk_mini_footprint.json"
    footprint_path.write_text(
        json.dumps({"name": "test_robot", "polygon_xy_m": _footprint().tolist()}),
        encoding="utf-8",
    )
    before = pgm.read_bytes()
    output = tmp_path / "out"
    script = Path(__file__).resolve().parents[1] / "tools" / "evaluate_planner_request_footprint_preflight.py"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--map-pgm",
            str(pgm),
            "--map-yaml",
            str(map_yaml),
            "--planner-pairs",
            str(pairs_path),
            "--gap-diagnostics",
            str(gap_path),
            "--footprint",
            str(footprint_path),
            "--output",
            str(output),
        ],
        env=env,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert pgm.read_bytes() == before
    payload = json.loads(
        (output / "planner_request_footprint_preflight.json").read_text(encoding="utf-8")
    )
    assert payload["summary"]["request_count"] == 4
    assert "pose_class_counts" in completed.stdout
