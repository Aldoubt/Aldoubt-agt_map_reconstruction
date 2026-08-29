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
        from agt_map_reconstruction.maps.vehicle_handoff_anchor_lateral_recovery import (
            recover_lateral_vehicle_handoff_anchors,
            write_lateral_vehicle_handoff_anchor_recovery_bundle,
        )
    except ImportError as exc:
        pytest.fail(
            f"P1-G1.2b lateral-aware vehicle handoff recovery API is missing: {exc}"
        )
    return (
        GridMetadata,
        recover_lateral_vehicle_handoff_anchors,
        write_lateral_vehicle_handoff_anchor_recovery_bundle,
    )


def _metadata():
    from agt_map_reconstruction.maps.grid_geometry import GridMetadata

    return GridMetadata(0.1, 0.0, 0.0, 60, 60)


def _rect(label, aisle_id, y0, y1):
    return {
        "aisle_id": aisle_id,
        "label": label,
        "polygon_xy": [[10.0, y0], [40.0, y0], [40.0, y1], [10.0, y1]],
        "width_m": (y1 - y0) * 0.1,
        "length_m": 3.0,
    }


def _aisles():
    # 0.4 m wide aisles, axis at y=15 and y=35 respectively.
    return [_rect("A01", 1, 13.0, 17.0), _rect("A02", 2, 33.0, 37.0)]


def _world_pose(metadata, x_cell, y_cell, yaw):
    x, y = metadata.grid_to_world(x_cell, y_cell)
    return {"x": float(x), "y": float(y), "yaw": float(yaw)}


def _planner_pairs(metadata):
    import math

    a1_entry = _world_pose(metadata, 10, 15, math.pi)
    a2_entry = _world_pose(metadata, 10, 35, math.pi)
    a1_exit = _world_pose(metadata, 40, 15, 0.0)
    a2_exit = _world_pose(metadata, 40, 35, 0.0)

    return {
        "schema_version": 1,
        "method": "nav2_headland_adjacent_pair_smoke_tests",
        "radius_m": 0.2,
        "bidirectional": True,
        "tests": [
            {
                "id": "A01-A02-entry",
                "pair_id": "A01-A02",
                "side": "entry",
                "enabled": True,
                "baseline_connected": True,
                "conservative_connected": True,
                "forward": {
                    "start": a1_entry,
                    "goal": {**a2_entry, "yaw": 0.0},
                },
                "reverse": {
                    "start": a2_entry,
                    "goal": {**a1_entry, "yaw": 0.0},
                },
            },
            {
                "id": "A01-A02-exit",
                "pair_id": "A01-A02",
                "side": "exit",
                "enabled": True,
                "baseline_connected": True,
                "conservative_connected": True,
                "forward": {
                    "start": a1_exit,
                    "goal": {**a2_exit, "yaw": 3.141592653589793},
                },
                "reverse": {
                    "start": a2_exit,
                    "goal": {**a1_exit, "yaw": 3.141592653589793},
                },
            },
        ],
    }


def _footprint():
    # Cell-exact footprint for deterministic raster fixtures. The production
    # algorithm must use the same polygon semantics for the MK-mini footprint.
    return np.asarray(
        [[0.04, 0.04], [0.04, -0.04], [-0.04, -0.04], [-0.04, 0.04]],
        dtype=float,
    )


def _grid():
    return np.full((60, 60), 254, dtype=np.uint8)


def _by_id(result):
    return {item["anchor_id"]: item for item in result["anchors"]}


def test_g1_2b_prefers_smaller_longitudinal_inset_before_smaller_lateral_shift():
    GridMetadata, recover, _ = _api()
    metadata = GridMetadata(0.1, 0.0, 0.0, 60, 60)
    grid = _grid()

    # Original entry pose is blocked. Longitudinal-only G1.2 can recover at
    # +0.1 m (x=11,y=15), but a lateral candidate exists at the original
    # station (x=10,y=14). G1.2b must prefer zero longitudinal inset even
    # though it requires a 0.1 m lateral move.
    grid[15, 10] = 205
    grid[16, 10] = 0

    result = recover(
        grid,
        _aisles(),
        _planner_pairs(metadata),
        _footprint(),
        metadata,
        footprint_name="test_robot",
    )
    item = _by_id(result)["A01-entry"]

    assert item["recovery_status"] == "recovered_lateral"
    assert item["longitudinal_only_status"] == "recovered"
    assert item["longitudinal_only_inset_m"] == pytest.approx(0.1)
    assert item["longitudinal_inset_m"] == pytest.approx(0.0)
    assert item["lateral_shift_m"] == pytest.approx(-0.1)
    assert item["yaw_delta_rad"] == pytest.approx(0.0)
    assert item["search_station_step_m"] == pytest.approx(metadata.resolution)
    assert item["lateral_search_step_m"] == pytest.approx(metadata.resolution)

    expected_x, expected_y = metadata.grid_to_world(10, 14)
    assert item["vehicle_anchor"]["x"] == pytest.approx(expected_x)
    assert item["vehicle_anchor"]["y"] == pytest.approx(expected_y)


def test_g1_2b_uses_smallest_absolute_lateral_shift_within_same_station():
    GridMetadata, recover, _ = _api()
    metadata = GridMetadata(0.1, 0.0, 0.0, 60, 60)
    grid = _grid()

    # At the topology station, 0 and +0.1 m are blocked while -0.1 and wider
    # offsets are free. The tie-break inside the minimum longitudinal station
    # must choose the smallest |lateral shift|, not an aisle-edge candidate.
    grid[15, 10] = 205
    grid[16, 10] = 0

    result = recover(grid, _aisles(), _planner_pairs(metadata), _footprint(), metadata)
    item = _by_id(result)["A01-entry"]

    assert item["longitudinal_inset_m"] == pytest.approx(0.0)
    assert abs(item["lateral_shift_m"]) == pytest.approx(0.1)
    assert item["vehicle_anchor"]["heading_rad"] == pytest.approx(
        item["topology_anchor"]["heading_rad"]
    )
    assert item["yaw_delta_rad"] == pytest.approx(0.0)

    policy = result["policy"]
    assert policy["unknown_blocking"] is True
    assert policy["occupied_blocking"] is True
    assert policy["longitudinal_search"] == "inward_only"
    assert policy["lateral_search"] == "within_aisle_footprint_feasible_band"
    assert policy["yaw_search"] is False
    assert policy["map_editing"] is False
    assert policy["selection_order"] == [
        "minimum_longitudinal_inset",
        "minimum_absolute_lateral_shift",
    ]


def test_g1_2b_marks_unavailable_when_no_polygon_valid_pose_exists_in_aisle():
    GridMetadata, recover, _ = _api()
    metadata = GridMetadata(0.1, 0.0, 0.0, 60, 60)
    grid = _grid()

    # Block the complete A01 polygon by UNKNOWN and A02 by OCCUPIED. Neither
    # longitudinal nor lateral search may cross these semantics.
    grid[13:18, 10:41] = 205
    grid[33:38, 10:41] = 0

    result = recover(grid, _aisles(), _planner_pairs(metadata), _footprint(), metadata)
    anchors = _by_id(result)

    for anchor_id in ("A01-entry", "A01-exit", "A02-entry", "A02-exit"):
        item = anchors[anchor_id]
        assert item["recovery_status"] == "unavailable"
        assert item["vehicle_anchor"] is None
        assert item["longitudinal_inset_m"] is None
        assert item["lateral_shift_m"] is None

    assert result["summary"]["available_count"] == 0
    assert result["summary"]["unavailable_count"] == 4


def test_g1_2b_retains_longitudinal_only_comparison_and_writes_json_csv_geojson(tmp_path):
    GridMetadata, recover, write_bundle = _api()
    metadata = GridMetadata(0.1, 0.0, 0.0, 60, 60)
    grid = _grid()
    grid[15, 10] = 205
    grid[16, 10] = 0

    result = recover(grid, _aisles(), _planner_pairs(metadata), _footprint(), metadata)
    item = _by_id(result)["A01-entry"]

    assert item["topology_anchor"] is not None
    assert item["longitudinal_only_anchor"] is not None
    assert item["vehicle_anchor"] is not None
    assert item["longitudinal_only_inset_m"] == pytest.approx(0.1)
    assert item["longitudinal_inset_m"] == pytest.approx(0.0)
    assert item["inset_reduction_m"] == pytest.approx(0.1)
    assert item["tested_pose_count"] > 0
    assert item["search_station_count"] > 0

    output = tmp_path / "recovery"
    paths = write_bundle(result, output)
    assert set(paths) == {"json", "csv", "geojson"}
    assert {path.name for path in output.iterdir()} == {
        "vehicle_handoff_anchors.json",
        "vehicle_handoff_anchors.csv",
        "vehicle_handoff_anchors.geojson",
    }
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["method"] == "p1_g1_2b_vehicle_handoff_anchor_lateral_recovery"
    assert "improved_over_longitudinal_only_count" in payload["summary"]

    with pytest.raises(FileExistsError, match="not empty"):
        write_bundle(result, output)


def test_g1_2b_cli_is_read_only_and_exposes_no_search_threshold_sweeps(tmp_path):
    metadata = _metadata()
    grid = _grid()
    grid[15, 10] = 205
    grid[16, 10] = 0

    pgm = tmp_path / "navigation_base_map.pgm"
    cv2.imwrite(str(pgm), np.flipud(grid))
    map_yaml = tmp_path / "navigation_base_map.yaml"
    map_yaml.write_text(
        yaml.safe_dump(
            {
                "image": pgm.name,
                "mode": "trinary",
                "resolution": metadata.resolution,
                "origin": [metadata.origin_x, metadata.origin_y, metadata.origin_yaw],
                "negate": 0,
                "occupied_thresh": 0.65,
                "free_thresh": 0.196,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    aisles_path = tmp_path / "aisle_rectangles.json"
    aisles_path.write_text(json.dumps({"rectangles": _aisles()}), encoding="utf-8")
    pairs_path = tmp_path / "planner_pairs.yaml"
    pairs_path.write_text(
        yaml.safe_dump(_planner_pairs(metadata), sort_keys=False), encoding="utf-8"
    )
    footprint_path = tmp_path / "mk_mini_footprint.json"
    footprint_path.write_text(
        json.dumps({"name": "test_robot", "polygon_xy_m": _footprint().tolist()}),
        encoding="utf-8",
    )

    before = pgm.read_bytes()
    output = tmp_path / "out"
    script = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "recover_lateral_vehicle_handoff_anchors.py"
    )
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
            "--aisles",
            str(aisles_path),
            "--planner-pairs",
            str(pairs_path),
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
    assert (output / "vehicle_handoff_anchors.json").exists()
    assert "anchor_count" in completed.stdout
    assert "improved_over_longitudinal_only_count" in completed.stdout

    source = script.read_text(encoding="utf-8")
    assert "--max-inset" not in source
    assert "--search-step" not in source
    assert "--lateral-step" not in source
    assert "--max-lateral-shift" not in source
    assert "--allow-unknown" not in source
