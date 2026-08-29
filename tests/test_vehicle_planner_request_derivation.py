import json
from pathlib import Path
import os
import subprocess
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _api():
    try:
        from agt_map_reconstruction.maps.vehicle_planner_request_derivation import (
            derive_vehicle_planner_requests,
            write_vehicle_planner_request_bundle,
        )
    except ImportError as exc:
        pytest.fail(f"P1-G1.3 vehicle planner request derivation API is missing: {exc}")
    return derive_vehicle_planner_requests, write_vehicle_planner_request_bundle


def _anchor(label, side, *, x=None, y=None, heading=0.0, inset=None, lateral=None):
    vehicle = None
    if x is not None and y is not None:
        vehicle = {"x": float(x), "y": float(y), "heading_rad": float(heading)}
    return {
        "anchor_id": f"{label}-{side}",
        "aisle_id": int(label[1:]),
        "label": label,
        "side": side,
        "vehicle_anchor": vehicle,
        "longitudinal_inset_m": inset,
        "lateral_shift_m": lateral,
    }


def _lateral_recovery():
    return {
        "schema_version": 1,
        "method": "p1_g1_2b_vehicle_handoff_anchor_lateral_recovery",
        "footprint": {
            "name": "mk_mini_preview",
            "polygon_xy_m": [
                [0.42, 0.30],
                [0.42, -0.30],
                [-0.42, -0.30],
                [-0.42, 0.30],
            ],
        },
        "anchors": [
            _anchor("A01", "entry", x=10.0, y=1.0, inset=0.25, lateral=0.0),
            _anchor("A02", "entry", x=20.0, y=2.0, inset=0.50, lateral=0.05),
            _anchor("A03", "entry", inset=None, lateral=None),
            _anchor("A03", "exit", x=30.0, y=3.0, inset=7.50, lateral=0.15),
            _anchor("A04", "exit", x=40.0, y=4.0, inset=0.35, lateral=0.05),
            _anchor("A05", "exit", inset=None, lateral=None),
        ],
    }


def _planner_test(case_id, *, enabled, connected, yaw_base):
    pair_id, side = case_id.rsplit("-", 1)
    return {
        "id": case_id,
        "pair_id": pair_id,
        "side": side,
        "radius_m": 0.2,
        "enabled": bool(enabled),
        "baseline_connected": bool(connected),
        "conservative_connected": bool(connected),
        "forward": {
            "start": {"x": 100.0, "y": 101.0, "yaw": yaw_base + 0.01},
            "goal": {"x": 110.0, "y": 111.0, "yaw": yaw_base + 0.02},
        },
        "reverse": {
            "start": {"x": 120.0, "y": 121.0, "yaw": yaw_base + 3.01},
            "goal": {"x": 130.0, "y": 131.0, "yaw": yaw_base + 3.02},
        },
    }


def _planner_pairs():
    return {
        "schema_version": 1,
        "method": "nav2_headland_adjacent_pair_smoke_tests",
        "radius_m": 0.2,
        "bidirectional": True,
        "tests": [
            _planner_test("A01-A02-entry", enabled=True, connected=True, yaw_base=0.10),
            _planner_test("A02-A03-entry", enabled=True, connected=True, yaw_base=0.20),
            _planner_test("A03-A04-exit", enabled=False, connected=False, yaw_base=0.30),
            _planner_test("A04-A05-exit", enabled=False, connected=False, yaw_base=0.40),
        ],
    }


def _gap_diagnostics():
    return {
        "schema_version": 2,
        "method": "scoped_headland_gap_diagnostics",
        "radius_m": 0.2,
        "records": [
            {
                "pair_id": "A03-A04",
                "side": "exit",
                "evaluation_status": "evaluated",
                "strict_connected": False,
                "bridge_type": "mixed_bridge",
                "frozen_evidence_marker": "keep-me",
            },
            {
                "pair_id": "A04-A05",
                "side": "exit",
                "evaluation_status": "evaluated",
                "strict_connected": False,
                "bridge_type": "clearance_only_bridge",
                "frozen_evidence_marker": "exclude-me",
            },
        ],
    }


def _pair_side(case_id, expectation, ready, *, negative_reason=None, inset=None, lateral=None):
    pair_id, side = case_id.rsplit("-", 1)
    first_label, second_label = pair_id.split("-")
    return {
        "case_id": case_id,
        "pair_id": pair_id,
        "side": side,
        "expectation_class": expectation,
        "negative_reason": negative_reason,
        "first_anchor_id": f"{first_label}-{side}",
        "second_anchor_id": f"{second_label}-{side}",
        "pair_vehicle_ready": bool(ready),
        "max_longitudinal_inset_m": inset,
        "max_abs_lateral_shift_m": lateral,
    }


def _feasibility_audit():
    pair_sides = [
        _pair_side(
            "A01-A02-entry", "positive", True, inset=0.50, lateral=0.05
        ),
        _pair_side(
            "A02-A03-entry", "positive", False, inset=None, lateral=None
        ),
        _pair_side(
            "A03-A04-exit",
            "negative_control",
            True,
            negative_reason="mixed_bridge",
            inset=7.50,
            lateral=0.15,
        ),
        _pair_side(
            "A04-A05-exit",
            "negative_control",
            False,
            negative_reason="clearance_only_bridge",
            inset=None,
            lateral=None,
        ),
    ]
    return {
        "schema_version": 1,
        "method": "p1_g1_2c_vehicle_handoff_feasibility_audit",
        "summary": {
            "pair_side_count": 4,
            "positive_pair_side_count": 2,
            "negative_pair_side_count": 2,
            "pair_vehicle_ready_count": 2,
            "pair_vehicle_not_ready_count": 2,
        },
        "pair_sides": pair_sides,
    }


def _derive():
    derive, _ = _api()
    return derive(
        _lateral_recovery(),
        _feasibility_audit(),
        _planner_pairs(),
        _gap_diagnostics(),
    )


def test_g1_3_derives_only_vehicle_ready_pair_sides_and_reports_excluded_cases():
    result = _derive()
    vehicle_pairs = result["vehicle_planner_pairs"]
    test_ids = [item["id"] for item in vehicle_pairs["tests"]]

    assert test_ids == ["A01-A02-entry", "A03-A04-exit"]
    assert result["summary"] == {
        "source_pair_side_count": 4,
        "ready_pair_side_count": 2,
        "excluded_pair_side_count": 2,
        "positive_pair_side_count": 1,
        "negative_pair_side_count": 1,
        "directional_request_count": 4,
        "positive_requests": 2,
        "negative_requests": 2,
    }

    selection = {item["case_id"]: item for item in result["pair_selection"]}
    assert selection["A01-A02-entry"]["included"] is True
    assert selection["A02-A03-entry"]["included"] is False
    assert selection["A02-A03-entry"]["exclusion_reason"] == "pair_vehicle_not_ready"
    assert selection["A03-A04-exit"]["included"] is True
    assert selection["A04-A05-exit"]["included"] is False


def test_g1_3_replaces_endpoint_xy_with_vehicle_anchors_but_preserves_original_directional_yaw():
    result = _derive()
    tests = {item["id"]: item for item in result["vehicle_planner_pairs"]["tests"]}
    original = {item["id"]: item for item in _planner_pairs()["tests"]}

    positive = tests["A01-A02-entry"]
    assert positive["forward"]["start"] == {
        "x": 10.0,
        "y": 1.0,
        "yaw": original["A01-A02-entry"]["forward"]["start"]["yaw"],
    }
    assert positive["forward"]["goal"] == {
        "x": 20.0,
        "y": 2.0,
        "yaw": original["A01-A02-entry"]["forward"]["goal"]["yaw"],
    }
    assert positive["reverse"]["start"] == {
        "x": 20.0,
        "y": 2.0,
        "yaw": original["A01-A02-entry"]["reverse"]["start"]["yaw"],
    }
    assert positive["reverse"]["goal"] == {
        "x": 10.0,
        "y": 1.0,
        "yaw": original["A01-A02-entry"]["reverse"]["goal"]["yaw"],
    }

    negative = tests["A03-A04-exit"]
    assert negative["forward"]["start"]["x"] == pytest.approx(30.0)
    assert negative["forward"]["goal"]["x"] == pytest.approx(40.0)
    assert negative["forward"]["start"]["yaw"] == pytest.approx(
        original["A03-A04-exit"]["forward"]["start"]["yaw"]
    )


def test_g1_3_builds_negative_compatibility_view_by_copying_only_ready_frozen_records():
    result = _derive()
    compatibility = result["vehicle_ready_gap_diagnostics"]

    assert compatibility["radius_m"] == pytest.approx(0.2)
    assert compatibility["method"] == "p1_g1_3_vehicle_ready_gap_diagnostics_selection"
    assert len(compatibility["records"]) == 1
    record = compatibility["records"][0]
    assert record["pair_id"] == "A03-A04"
    assert record["side"] == "exit"
    assert record["bridge_type"] == "mixed_bridge"
    assert record["frozen_evidence_marker"] == "keep-me"
    assert all(item.get("frozen_evidence_marker") != "exclude-me" for item in compatibility["records"])
    assert compatibility["policy"]["gap_recomputed"] is False


def test_g1_3_keeps_source_topology_radius_as_provenance_and_does_not_filter_deep_ready_anchor():
    result = _derive()

    assert result["source_topology_radius_m"] == pytest.approx(0.2)
    assert result["radius_role"] == "source_topology_clearance_proxy"
    assert "vehicle_radius_m" not in result
    assert result["vehicle_planner_pairs"]["radius_m"] == pytest.approx(0.2)

    selection = {item["case_id"]: item for item in result["pair_selection"]}
    deep = selection["A03-A04-exit"]
    assert deep["included"] is True
    assert deep["max_longitudinal_inset_m"] == pytest.approx(7.50)
    assert deep["max_abs_lateral_shift_m"] == pytest.approx(0.15)
    assert "max_allowed_inset_m" not in result["policy"]
    assert "max_allowed_lateral_shift_m" not in result["policy"]
    assert result["policy"]["acceptance_thresholds"] is False


def test_g1_3_bundle_writes_runtime_compatible_artifacts_and_rejects_nonempty_output(tmp_path):
    _, write_bundle = _api()
    result = _derive()

    output = tmp_path / "g1_3"
    paths = write_bundle(result, output)
    assert set(paths) == {
        "vehicle_planner_pairs",
        "vehicle_ready_gap_diagnostics",
        "derivation",
        "selection_csv",
    }
    assert {path.name for path in output.iterdir()} == {
        "vehicle_planner_pairs.yaml",
        "vehicle_ready_gap_diagnostics.json",
        "vehicle_planner_request_derivation.json",
        "vehicle_planner_pair_selection.csv",
    }

    pairs = yaml.safe_load(paths["vehicle_planner_pairs"].read_text(encoding="utf-8"))
    assert [item["id"] for item in pairs["tests"]] == [
        "A01-A02-entry",
        "A03-A04-exit",
    ]
    gap = json.loads(paths["vehicle_ready_gap_diagnostics"].read_text(encoding="utf-8"))
    assert len(gap["records"]) == 1

    with pytest.raises(FileExistsError, match="not empty"):
        write_bundle(result, output)


def test_g1_3_cli_consumes_only_frozen_g1_2_artifacts_and_exposes_no_map_or_acceptance_thresholds(tmp_path):
    lateral_path = tmp_path / "vehicle_handoff_anchors.json"
    lateral_path.write_text(json.dumps(_lateral_recovery()), encoding="utf-8")
    feasibility_path = tmp_path / "vehicle_handoff_feasibility.json"
    feasibility_path.write_text(json.dumps(_feasibility_audit()), encoding="utf-8")
    pairs_path = tmp_path / "planner_pairs.yaml"
    pairs_path.write_text(yaml.safe_dump(_planner_pairs(), sort_keys=False), encoding="utf-8")
    gap_path = tmp_path / "headland_gap_diagnostics.json"
    gap_path.write_text(json.dumps(_gap_diagnostics()), encoding="utf-8")

    output = tmp_path / "out"
    script = Path(__file__).resolve().parents[1] / "tools" / "derive_vehicle_planner_requests.py"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--lateral-recovery",
            str(lateral_path),
            "--feasibility-audit",
            str(feasibility_path),
            "--planner-pairs",
            str(pairs_path),
            "--gap-diagnostics",
            str(gap_path),
            "--output",
            str(output),
        ],
        env=env,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output / "vehicle_planner_pairs.yaml").exists()
    assert (output / "vehicle_ready_gap_diagnostics.json").exists()
    assert "ready_pair_side_count" in completed.stdout

    source = script.read_text(encoding="utf-8")
    assert "--map-pgm" not in source
    assert "--map-yaml" not in source
    assert "--max-inset" not in source
    assert "--max-lateral" not in source
    assert "--min-ready-pairs" not in source
