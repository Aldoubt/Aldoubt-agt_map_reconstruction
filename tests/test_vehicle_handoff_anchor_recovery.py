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
        from agt_map_reconstruction.maps.vehicle_handoff_anchor_recovery import (
            recover_vehicle_handoff_anchors,
            write_vehicle_handoff_anchor_recovery_bundle,
        )
    except ImportError as exc:
        pytest.fail(f"P1-G1.2 vehicle handoff anchor recovery API is missing: {exc}")
    return GridMetadata, recover_vehicle_handoff_anchors, write_vehicle_handoff_anchor_recovery_bundle


def _rect(label, aisle_id, y0, y1):
    return {
        "aisle_id": aisle_id,
        "label": label,
        "polygon_xy": [[10.0, y0], [40.0, y0], [40.0, y1], [10.0, y1]],
        "width_m": (y1 - y0) * 0.1,
        "length_m": 3.0,
    }


def _metadata():
    from agt_map_reconstruction.maps.grid_geometry import GridMetadata

    return GridMetadata(0.1, 0.0, 0.0, 60, 60)


def _world_pose(metadata, x_cell, y_cell, yaw):
    x, y = metadata.grid_to_world(x_cell, y_cell)
    return {"x": float(x), "y": float(y), "yaw": float(yaw)}


def _planner_pairs(metadata):
    import math

    a1_entry = _world_pose(metadata, 10, 15, 0.0)
    a2_entry = _world_pose(metadata, 10, 35, 0.0)
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
                    "start": {**a1_entry, "yaw": math.pi},
                    "goal": {**a2_entry, "yaw": 0.0},
                },
                "reverse": {
                    "start": {**a2_entry, "yaw": math.pi},
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
                    "start": {**a1_exit, "yaw": 0.0},
                    "goal": {**a2_exit, "yaw": math.pi},
                },
                "reverse": {
                    "start": {**a2_exit, "yaw": 0.0},
                    "goal": {**a1_exit, "yaw": math.pi},
                },
            },
        ],
    }


def _aisles():
    return [_rect("A01", 1, 14.0, 16.0), _rect("A02", 2, 34.0, 36.0)]


def _footprint():
    # Tiny symmetric footprint keeps the fixture cell-exact while exercising
    # the same polygon semantics as the real MK-mini preview footprint.
    return np.asarray(
        [[0.04, 0.04], [0.04, -0.04], [-0.04, -0.04], [-0.04, 0.04]],
        dtype=float,
    )


def _grid():
    return np.full((60, 60), 254, dtype=np.uint8)


def _by_id(result):
    return {item["anchor_id"]: item for item in result["anchors"]}


def test_g1_2_recovers_nearest_valid_anchor_at_map_resolution_step():
    GridMetadata, recover, _ = _api()
    metadata = GridMetadata(0.1, 0.0, 0.0, 60, 60)
    grid = _grid()
    grid[15, 10] = 205

    result = recover(
        grid,
        _aisles(),
        _planner_pairs(metadata),
        _footprint(),
        metadata,
        footprint_name="test_robot",
    )
    anchors = _by_id(result)

    a01_entry = anchors["A01-entry"]
    assert a01_entry["recovery_status"] == "recovered"
    assert a01_entry["topology_pose_class"] == "unknown_overlap"
    assert a01_entry["longitudinal_inset_m"] == pytest.approx(0.1)
    assert a01_entry["search_step_m"] == pytest.approx(metadata.resolution)
    assert a01_entry["vehicle_anchor"] is not None

    expected_x, expected_y = metadata.grid_to_world(11, 15)
    assert a01_entry["vehicle_anchor"]["x"] == pytest.approx(expected_x)
    assert a01_entry["vehicle_anchor"]["y"] == pytest.approx(expected_y)

    a02_entry = anchors["A02-entry"]
    assert a02_entry["recovery_status"] == "already_valid"
    assert a02_entry["longitudinal_inset_m"] == pytest.approx(0.0)


def test_g1_2_exit_search_moves_inward_only_and_preserves_heading_and_lateral_position():
    GridMetadata, recover, _ = _api()
    metadata = GridMetadata(0.1, 0.0, 0.0, 60, 60)
    grid = _grid()
    grid[15, 40] = 0
    grid[15, 39] = 0

    result = recover(grid, _aisles(), _planner_pairs(metadata), _footprint(), metadata)
    item = _by_id(result)["A01-exit"]

    assert item["recovery_status"] == "recovered"
    assert item["longitudinal_inset_m"] == pytest.approx(0.2)
    assert item["lateral_shift_m"] == pytest.approx(0.0)
    assert item["yaw_delta_rad"] == pytest.approx(0.0)
    assert item["vehicle_anchor"]["heading_rad"] == pytest.approx(
        item["topology_anchor"]["heading_rad"]
    )

    expected_x, expected_y = metadata.grid_to_world(38, 15)
    assert item["vehicle_anchor"]["x"] == pytest.approx(expected_x)
    assert item["vehicle_anchor"]["y"] == pytest.approx(expected_y)


def test_g1_2_keeps_unknown_and_occupied_strict_and_marks_unavailable_when_axis_has_no_valid_pose():
    GridMetadata, recover, _ = _api()
    metadata = GridMetadata(0.1, 0.0, 0.0, 60, 60)
    grid = _grid()
    grid[15, 10:41] = 205
    grid[35, 10:41] = 0

    result = recover(grid, _aisles(), _planner_pairs(metadata), _footprint(), metadata)
    anchors = _by_id(result)

    a01_entry = anchors["A01-entry"]
    assert a01_entry["topology_pose_class"] == "unknown_overlap"
    assert a01_entry["recovery_status"] == "unavailable"
    assert a01_entry["vehicle_anchor"] is None

    a02_entry = anchors["A02-entry"]
    assert a02_entry["topology_pose_class"] == "occupied_overlap"
    assert a02_entry["recovery_status"] == "unavailable"
    assert a02_entry["vehicle_anchor"] is None

    assert result["policy"]["unknown_blocking"] is True
    assert result["policy"]["occupied_blocking"] is True
    assert result["policy"]["lateral_search"] is False
    assert result["policy"]["yaw_search"] is False
    assert result["policy"]["map_editing"] is False


def test_g1_2_deduplicates_unique_aisle_side_anchors_and_writes_json_csv_geojson(tmp_path):
    GridMetadata, recover, write_bundle = _api()
    metadata = GridMetadata(0.1, 0.0, 0.0, 60, 60)
    result = recover(_grid(), _aisles(), _planner_pairs(metadata), _footprint(), metadata)

    assert result["summary"]["anchor_count"] == 4
    assert result["summary"]["available_count"] == 4
    assert result["summary"]["unavailable_count"] == 0
    assert {item["anchor_id"] for item in result["anchors"]} == {
        "A01-entry",
        "A01-exit",
        "A02-entry",
        "A02-exit",
    }

    output = tmp_path / "recovery"
    paths = write_bundle(result, output)
    assert set(paths) == {"json", "csv", "geojson"}
    assert {path.name for path in output.iterdir()} == {
        "vehicle_handoff_anchors.json",
        "vehicle_handoff_anchors.csv",
        "vehicle_handoff_anchors.geojson",
    }
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["method"] == "p1_g1_2_vehicle_handoff_anchor_recovery"
    geojson = json.loads(paths["geojson"].read_text(encoding="utf-8"))
    assert geojson["type"] == "FeatureCollection"

    with pytest.raises(FileExistsError, match="not empty"):
        write_bundle(result, output)


def test_g1_2_cli_is_read_only_and_exposes_no_inset_or_step_sweep(tmp_path):
    metadata = _metadata()
    grid = _grid()
    grid[15, 10] = 205

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
    script = Path(__file__).resolve().parents[1] / "tools" / "recover_vehicle_handoff_anchors.py"
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
    assert "available_count" in completed.stdout

    source = script.read_text(encoding="utf-8")
    assert "--max-inset" not in source
    assert "--search-step" not in source
    assert "--allow-unknown" not in source
