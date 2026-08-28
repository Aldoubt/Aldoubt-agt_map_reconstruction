import json
from pathlib import Path

import numpy as np

from agt_map_reconstruction.maps.planner_path_scope_audit import (
    write_planner_path_scope_audit_bundle,
)


def _audit():
    return {
        "schema_version": 1,
        "method": "p1_f3_nav2_planner_path_scope_audit",
        "radius_m": 0.2,
        "summary": {
            "record_count": 2,
            "classification_counts": {
                "positive_local_match": 1,
                "negative_global_detour": 1,
            },
        },
        "records": [
            {
                "request_id": "A01-A02-entry-forward",
                "pair_id": "A01-A02",
                "side": "entry",
                "direction": "forward",
                "frozen_conservative_connected": True,
                "strict_connected_4": True,
                "strict_connected_8": True,
                "strict4_matches_frozen": True,
                "planner_success": True,
                "scope_class": "pair_domain",
                "pair_domain_contained": True,
                "finite_headland_contained": True,
                "path_cell_count": 3,
                "path_outside_pair_domain_cell_count": 0,
                "path_outside_pair_domain_fraction": 0.0,
                "path_outside_finite_headland_cell_count": 0,
                "path_outside_finite_headland_fraction": 0.0,
                "touches_unknown": False,
                "touches_occupied": False,
                "min_source_map_clearance_m": 0.25,
                "path_length_m": 2.0,
                "direct_distance_m": 2.0,
                "detour_ratio": 1.0,
                "classification": "positive_local_match",
                "path_xy": [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]],
            },
            {
                "request_id": "A13-A14-exit-forward",
                "pair_id": "A13-A14",
                "side": "exit",
                "direction": "forward",
                "frozen_conservative_connected": False,
                "strict_connected_4": False,
                "strict_connected_8": False,
                "strict4_matches_frozen": True,
                "planner_success": True,
                "scope_class": "global_outside_finite_headland",
                "pair_domain_contained": False,
                "finite_headland_contained": False,
                "path_cell_count": 20,
                "path_outside_pair_domain_cell_count": 15,
                "path_outside_pair_domain_fraction": 0.75,
                "path_outside_finite_headland_cell_count": 10,
                "path_outside_finite_headland_fraction": 0.5,
                "touches_unknown": False,
                "touches_occupied": False,
                "min_source_map_clearance_m": 0.2,
                "path_length_m": 54.2,
                "direct_distance_m": 4.7,
                "detour_ratio": 11.5,
                "classification": "negative_global_detour",
                "path_xy": [[10.0, 2.0], [1.0, -7.0], [29.0, 0.0]],
            },
        ],
    }


def test_bundle_writes_json_csv_and_geojson(tmp_path):
    paths = write_planner_path_scope_audit_bundle(_audit(), tmp_path / "out")
    output = tmp_path / "out"

    assert set(paths) == {"json", "csv", "geojson"}
    assert {item.name for item in output.iterdir()} == {
        "planner_path_scope_audit.json",
        "planner_path_scope_audit.csv",
        "planner_path_scope_audit.geojson",
    }
    payload = json.loads((output / "planner_path_scope_audit.json").read_text())
    assert payload["summary"]["record_count"] == 2
    geojson = json.loads((output / "planner_path_scope_audit.geojson").read_text())
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == 2
    assert geojson["features"][1]["properties"]["classification"] == "negative_global_detour"
    assert (output / "planner_path_scope_audit.csv").read_text().startswith("request_id,")


def test_bundle_rejects_nonempty_output_directory(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    (output / "old.txt").write_text("old")

    try:
        write_planner_path_scope_audit_bundle(_audit(), output)
    except FileExistsError as exc:
        assert "not empty" in str(exc)
    else:
        raise AssertionError("non-empty output directory was accepted")


def test_cli_declares_frozen_inputs_and_does_not_edit_maps():
    source = (
        Path(__file__).parents[1] / "tools" / "evaluate_planner_path_scope_audit.py"
    ).read_text(encoding="utf-8")
    for flag in (
        "--map",
        "--aisles",
        "--connectivity",
        "--depth-profile",
        "--planner-results",
        "--radius",
        "--output",
    ):
        assert flag in source
    assert "analyze_planner_path_scope_audit" in source
    assert "write_planner_path_scope_audit_bundle" in source
    assert "cv2.imwrite" not in source
