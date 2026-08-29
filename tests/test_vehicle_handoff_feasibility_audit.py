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
        from agt_map_reconstruction.maps.vehicle_handoff_feasibility_audit import (
            audit_vehicle_handoff_feasibility,
            write_vehicle_handoff_feasibility_bundle,
        )
    except ImportError as exc:
        pytest.fail(f"P1-G1.2c vehicle handoff feasibility audit API is missing: {exc}")
    return audit_vehicle_handoff_feasibility, write_vehicle_handoff_feasibility_bundle


def _rect(label, aisle_id, width_m, length_m=3.0):
    return {
        "aisle_id": aisle_id,
        "label": label,
        "width_m": float(width_m),
        "length_m": float(length_m),
        "polygon_xy": [[0.0, 0.0], [30.0, 0.0], [30.0, 10.0], [0.0, 10.0]],
    }


def _aisles():
    return [
        _rect("A01", 1, 0.50),
        _rect("A02", 2, 0.80),
        _rect("A03", 3, 0.80),
        _rect("A04", 4, 0.80),
    ]


def _anchor(label, side, *, status, vehicle, inset, lateral=0.0, band=None):
    return {
        "anchor_id": f"{label}-{side}",
        "aisle_id": int(label[1:]),
        "label": label,
        "side": side,
        "recovery_status": status,
        "topology_anchor": {"x": 0.0, "y": 0.0, "heading_rad": 0.0},
        "topology_pose_class": "mixed_blocking_overlap",
        "longitudinal_only_status": "recovered" if vehicle is not None else "unavailable",
        "longitudinal_only_anchor": vehicle,
        "longitudinal_only_inset_m": inset,
        "vehicle_anchor": vehicle,
        "vehicle_pose_class": None if vehicle is None else "valid",
        "longitudinal_inset_m": inset,
        "lateral_shift_m": None if vehicle is None else lateral,
        "yaw_delta_rad": None if vehicle is None else 0.0,
        "lateral_feasible_band_m": band,
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
            # A01 is narrower than the 0.60 m vehicle footprint.
            _anchor("A01", "entry", status="unavailable", vehicle=None, inset=None, band=None),
            # A02/A03 entry anchors are usable and form one vehicle-ready positive pair.
            _anchor(
                "A02",
                "entry",
                status="recovered_longitudinal",
                vehicle={"x": 1.0, "y": 2.0, "heading_rad": 0.0},
                inset=0.30,
                lateral=0.0,
                band=[-0.10, 0.10],
            ),
            _anchor(
                "A03",
                "entry",
                status="recovered_lateral",
                vehicle={"x": 2.0, "y": 2.0, "heading_rad": 0.0},
                inset=0.60,
                lateral=-0.10,
                band=[-0.10, 0.10],
            ),
            # A02 exit is unavailable even though the aisle is wide enough.
            _anchor("A02", "exit", status="unavailable", vehicle=None, inset=None, band=[-0.10, 0.10]),
            _anchor(
                "A03",
                "exit",
                status="recovered_longitudinal",
                vehicle={"x": 2.0, "y": 3.0, "heading_rad": 0.0},
                inset=1.20,
                lateral=0.0,
                band=[-0.10, 0.10],
            ),
            # Ignored pair-side fixture.
            _anchor(
                "A04",
                "entry",
                status="recovered_longitudinal",
                vehicle={"x": 3.0, "y": 2.0, "heading_rad": 0.0},
                inset=0.20,
                lateral=0.0,
                band=[-0.10, 0.10],
            ),
        ],
    }


def _planner_test(case_id, *, enabled=True):
    pair_id, side = case_id.rsplit("-", 1)
    return {
        "id": case_id,
        "pair_id": pair_id,
        "side": side,
        "radius_m": 0.2,
        "enabled": bool(enabled),
        "baseline_connected": bool(enabled),
        "conservative_connected": bool(enabled),
        "forward": {"start": {"x": 0.0, "y": 0.0, "yaw": 0.0}, "goal": {"x": 1.0, "y": 0.0, "yaw": 0.0}},
        "reverse": {"start": {"x": 1.0, "y": 0.0, "yaw": 3.14}, "goal": {"x": 0.0, "y": 0.0, "yaw": 3.14}},
    }


def _planner_pairs():
    return {
        "schema_version": 1,
        "method": "nav2_headland_adjacent_pair_smoke_tests",
        "radius_m": 0.2,
        "bidirectional": True,
        "tests": [
            _planner_test("A01-A02-entry", enabled=True),
            _planner_test("A02-A03-entry", enabled=True),
            _planner_test("A02-A03-exit", enabled=False),
            _planner_test("A03-A04-entry", enabled=False),
        ],
    }


def _gap_diagnostics():
    return {
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


def test_g1_2c_classifies_width_failure_and_map_band_failure_without_searching_new_poses():
    audit, _ = _api()
    result = audit(_lateral_recovery(), _aisles(), _planner_pairs(), _gap_diagnostics())
    by_id = {item["anchor_id"]: item for item in result["anchors"]}

    assert by_id["A01-entry"]["feasibility_class"] == "footprint_wider_than_aisle"
    assert by_id["A01-entry"]["footprint_width_m"] == pytest.approx(0.60)
    assert by_id["A01-entry"]["aisle_width_m"] == pytest.approx(0.50)

    assert by_id["A02-exit"]["feasibility_class"] == "no_map_valid_pose_in_aisle_band"
    assert by_id["A02-exit"]["footprint_width_m"] == pytest.approx(0.60)
    assert by_id["A02-exit"]["aisle_width_m"] == pytest.approx(0.80)

    assert by_id["A02-entry"]["feasibility_class"] == "vehicle_anchor_valid"
    assert by_id["A02-entry"]["vehicle_anchor"] == {"x": 1.0, "y": 2.0, "heading_rad": 0.0}
    assert result["policy"]["pose_search"] is False
    assert result["policy"]["map_editing"] is False


def test_g1_2c_reports_continuous_depth_and_lateral_metrics_without_deep_recovery_threshold():
    audit, _ = _api()
    result = audit(_lateral_recovery(), _aisles(), _planner_pairs(), _gap_diagnostics())
    by_id = {item["anchor_id"]: item for item in result["anchors"]}

    a02 = by_id["A02-entry"]
    assert a02["longitudinal_inset_m"] == pytest.approx(0.30)
    assert a02["inset_over_aisle_length"] == pytest.approx(0.10)
    assert a02["lateral_shift_m"] == pytest.approx(0.0)
    assert a02["lateral_feasible_band_width_m"] == pytest.approx(0.20)

    a03 = by_id["A03-exit"]
    assert a03["longitudinal_inset_m"] == pytest.approx(1.20)
    assert a03["inset_over_aisle_length"] == pytest.approx(0.40)

    assert "deep_recovery_threshold_m" not in result["policy"]
    assert "deep_recovery_threshold_ratio" not in result["policy"]
    assert all("deep_row_recovery" not in item for item in result["anchors"])


def test_g1_2c_audits_same_positive_and_diagnostic_pair_sides_and_marks_vehicle_ready_subset():
    audit, _ = _api()
    result = audit(_lateral_recovery(), _aisles(), _planner_pairs(), _gap_diagnostics())
    pairs = {item["case_id"]: item for item in result["pair_sides"]}

    assert set(pairs) == {"A01-A02-entry", "A02-A03-entry", "A02-A03-exit"}
    assert pairs["A01-A02-entry"]["expectation_class"] == "positive"
    assert pairs["A01-A02-entry"]["pair_vehicle_ready"] is False
    assert pairs["A01-A02-entry"]["first_anchor_class"] == "footprint_wider_than_aisle"

    assert pairs["A02-A03-entry"]["expectation_class"] == "positive"
    assert pairs["A02-A03-entry"]["pair_vehicle_ready"] is True
    assert pairs["A02-A03-entry"]["max_longitudinal_inset_m"] == pytest.approx(0.60)
    assert pairs["A02-A03-entry"]["max_abs_lateral_shift_m"] == pytest.approx(0.10)

    assert pairs["A02-A03-exit"]["expectation_class"] == "negative_control"
    assert pairs["A02-A03-exit"]["negative_reason"] == "mixed_bridge"
    assert pairs["A02-A03-exit"]["pair_vehicle_ready"] is False

    summary = result["summary"]
    assert summary["pair_side_count"] == 3
    assert summary["positive_pair_side_count"] == 2
    assert summary["negative_pair_side_count"] == 1
    assert summary["pair_vehicle_ready_count"] == 1
    assert summary["pair_vehicle_not_ready_count"] == 2


def test_g1_2c_bundle_writes_anchor_and_pair_tables_and_rejects_nonempty_output(tmp_path):
    audit, write_bundle = _api()
    result = audit(_lateral_recovery(), _aisles(), _planner_pairs(), _gap_diagnostics())

    output = tmp_path / "audit"
    paths = write_bundle(result, output)
    assert set(paths) == {"json", "anchors_csv", "pairs_csv"}
    assert {path.name for path in output.iterdir()} == {
        "vehicle_handoff_feasibility.json",
        "vehicle_handoff_anchor_feasibility.csv",
        "vehicle_handoff_pair_feasibility.csv",
    }
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["method"] == "p1_g1_2c_vehicle_handoff_feasibility_audit"
    assert payload["summary"]["pair_vehicle_ready_count"] == 1

    with pytest.raises(FileExistsError, match="not empty"):
        write_bundle(result, output)


def test_g1_2c_cli_consumes_frozen_artifacts_only_and_exposes_no_acceptance_thresholds(tmp_path):
    lateral_path = tmp_path / "vehicle_handoff_anchors.json"
    lateral_path.write_text(json.dumps(_lateral_recovery()), encoding="utf-8")
    aisles_path = tmp_path / "aisle_rectangles.json"
    aisles_path.write_text(json.dumps({"rectangles": _aisles()}), encoding="utf-8")
    pairs_path = tmp_path / "planner_pairs.yaml"
    pairs_path.write_text(yaml.safe_dump(_planner_pairs(), sort_keys=False), encoding="utf-8")
    gap_path = tmp_path / "headland_gap_diagnostics.json"
    gap_path.write_text(json.dumps(_gap_diagnostics()), encoding="utf-8")

    output = tmp_path / "out"
    script = Path(__file__).resolve().parents[1] / "tools" / "audit_vehicle_handoff_feasibility.py"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--lateral-recovery",
            str(lateral_path),
            "--aisles",
            str(aisles_path),
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
    assert (output / "vehicle_handoff_feasibility.json").exists()
    assert "pair_vehicle_ready_count" in completed.stdout

    source = script.read_text(encoding="utf-8")
    assert "--map-pgm" not in source
    assert "--max-inset" not in source
    assert "--deep-inset-threshold" not in source
    assert "--deep-ratio-threshold" not in source
    assert "--min-ready-pairs" not in source
