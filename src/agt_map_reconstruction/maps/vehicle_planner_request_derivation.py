"""Derive vehicle-ready planner requests from frozen G1.2b/G1.2c artifacts.

P1-G1.3 is a read-only compatibility stage.  It does not read or edit the map,
search for new poses, or apply acceptance thresholds.  It selects only pair
sides already marked vehicle-ready by G1.2c, replaces the topology endpoint
x/y values with the recovered G1.2b vehicle anchors, and preserves each frozen
directional yaw from the original planner-pair artifact.
"""

from __future__ import annotations

import csv
import copy
import json
import math
from pathlib import Path

import yaml


def _finite_float(value, label):
    try:
        result = float(value)
    except Exception as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _radius(payload, label):
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must be an object")
    if "radius_m" not in payload:
        raise ValueError(f"{label}.radius_m is required")
    value = _finite_float(payload["radius_m"], f"{label}.radius_m")
    if value <= 0.0:
        raise ValueError(f"{label}.radius_m must be > 0")
    return value


def _anchor_lookup(lateral_recovery):
    if not isinstance(lateral_recovery, dict):
        raise TypeError("lateral_recovery must be an object")
    anchors = lateral_recovery.get("anchors")
    if not isinstance(anchors, list):
        raise ValueError("lateral_recovery.anchors must be a list")

    lookup = {}
    for item in anchors:
        anchor_id = str(item.get("anchor_id", ""))
        if not anchor_id:
            raise ValueError("each lateral recovery anchor requires anchor_id")
        if anchor_id in lookup:
            raise ValueError(f"duplicate lateral recovery anchor: {anchor_id}")
        lookup[anchor_id] = item
    return lookup


def _test_lookup(planner_pairs):
    tests = planner_pairs.get("tests")
    if not isinstance(tests, list):
        raise ValueError("planner_pairs.tests must be a list")
    lookup = {}
    for item in tests:
        case_id = str(item.get("id") or f"{item.get('pair_id')}-{item.get('side')}")
        if case_id in lookup:
            raise ValueError(f"duplicate planner pair test: {case_id}")
        lookup[case_id] = item
    return lookup


def _vehicle_xy(anchor, anchor_id):
    vehicle = anchor.get("vehicle_anchor")
    if not isinstance(vehicle, dict):
        raise ValueError(f"vehicle-ready anchor {anchor_id} has no vehicle_anchor")
    return {
        "x": _finite_float(vehicle.get("x"), f"{anchor_id} vehicle_anchor.x"),
        "y": _finite_float(vehicle.get("y"), f"{anchor_id} vehicle_anchor.y"),
    }


def _pose_with_vehicle_xy(original_pose, vehicle_xy, label):
    if not isinstance(original_pose, dict):
        raise ValueError(f"{label} must be an object")
    yaw = _finite_float(original_pose.get("yaw"), f"{label}.yaw")
    return {"x": float(vehicle_xy["x"]), "y": float(vehicle_xy["y"]), "yaw": yaw}


def _derive_test(original, pair_audit, anchors):
    pair_id = str(original.get("pair_id", ""))
    labels = pair_id.split("-")
    if len(labels) != 2 or not all(labels):
        raise ValueError(f"invalid pair_id: {pair_id}")
    side = str(original.get("side", ""))
    if side not in {"entry", "exit"}:
        raise ValueError(f"invalid pair side for {pair_id}: {side}")

    first_id = str(pair_audit.get("first_anchor_id") or f"{labels[0]}-{side}")
    second_id = str(pair_audit.get("second_anchor_id") or f"{labels[1]}-{side}")
    first = anchors.get(first_id)
    second = anchors.get(second_id)
    if first is None or second is None:
        raise ValueError(f"pair {pair_id}-{side} references missing G1.2b anchor")

    first_xy = _vehicle_xy(first, first_id)
    second_xy = _vehicle_xy(second, second_id)

    forward = original.get("forward")
    reverse = original.get("reverse")
    if not isinstance(forward, dict) or not isinstance(reverse, dict):
        raise ValueError(f"planner pair {pair_id}-{side} requires forward and reverse poses")

    result = copy.deepcopy(original)
    result["forward"] = {
        "start": _pose_with_vehicle_xy(forward.get("start"), first_xy, f"{pair_id}-{side} forward start"),
        "goal": _pose_with_vehicle_xy(forward.get("goal"), second_xy, f"{pair_id}-{side} forward goal"),
    }
    result["reverse"] = {
        "start": _pose_with_vehicle_xy(reverse.get("start"), second_xy, f"{pair_id}-{side} reverse start"),
        "goal": _pose_with_vehicle_xy(reverse.get("goal"), first_xy, f"{pair_id}-{side} reverse goal"),
    }
    return result


def _gap_record_lookup(gap_diagnostics):
    records = gap_diagnostics.get("records")
    if not isinstance(records, list):
        raise ValueError("gap_diagnostics.records must be a list")
    lookup = {}
    for record in records:
        case_id = f"{record.get('pair_id')}-{record.get('side')}"
        if case_id in lookup:
            raise ValueError(f"duplicate gap diagnostic case: {case_id}")
        lookup[case_id] = record
    return lookup


def derive_vehicle_planner_requests(
    lateral_recovery,
    feasibility_audit,
    planner_pairs,
    gap_diagnostics,
):
    """Build runtime-compatible vehicle planner inputs from frozen artifacts."""
    if not isinstance(feasibility_audit, dict):
        raise TypeError("feasibility_audit must be an object")
    if not isinstance(planner_pairs, dict) or not isinstance(gap_diagnostics, dict):
        raise TypeError("planner_pairs and gap_diagnostics must be objects")

    planner_radius = _radius(planner_pairs, "planner_pairs")
    gap_radius = _radius(gap_diagnostics, "gap_diagnostics")
    if not math.isclose(planner_radius, gap_radius, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("planner_pairs and gap_diagnostics radius_m must match")

    pair_sides = feasibility_audit.get("pair_sides")
    if not isinstance(pair_sides, list):
        raise ValueError("feasibility_audit.pair_sides must be a list")

    anchors = _anchor_lookup(lateral_recovery)
    planner_by_case = _test_lookup(planner_pairs)
    gap_by_case = _gap_record_lookup(gap_diagnostics)

    included_tests = []
    included_negative_records = []
    pair_selection = []
    positive_pair_sides = 0
    negative_pair_sides = 0

    for item in pair_sides:
        case_id = str(item.get("case_id", ""))
        if not case_id:
            raise ValueError("each feasibility pair-side requires case_id")
        expectation = str(item.get("expectation_class", ""))
        if expectation not in {"positive", "negative_control"}:
            raise ValueError(f"unsupported expectation_class for {case_id}: {expectation}")

        ready = bool(item.get("pair_vehicle_ready"))
        selection = {
            "case_id": case_id,
            "pair_id": item.get("pair_id"),
            "side": item.get("side"),
            "expectation_class": expectation,
            "negative_reason": item.get("negative_reason"),
            "included": ready,
            "exclusion_reason": None if ready else "pair_vehicle_not_ready",
            "first_anchor_id": item.get("first_anchor_id"),
            "second_anchor_id": item.get("second_anchor_id"),
            "max_longitudinal_inset_m": item.get("max_longitudinal_inset_m"),
            "max_abs_lateral_shift_m": item.get("max_abs_lateral_shift_m"),
        }
        pair_selection.append(selection)

        if not ready:
            continue

        original = planner_by_case.get(case_id)
        if original is None:
            raise ValueError(f"vehicle-ready feasibility case has no planner pair: {case_id}")
        included_tests.append(_derive_test(original, item, anchors))

        if expectation == "positive":
            positive_pair_sides += 1
        else:
            negative_pair_sides += 1
            record = gap_by_case.get(case_id)
            if record is None:
                raise ValueError(f"vehicle-ready negative case has no frozen gap record: {case_id}")
            included_negative_records.append(copy.deepcopy(record))

    ready_count = len(included_tests)
    source_count = len(pair_sides)

    vehicle_planner_pairs = copy.deepcopy(planner_pairs)
    vehicle_planner_pairs["method"] = "p1_g1_3_vehicle_planner_pairs"
    vehicle_planner_pairs["radius_m"] = planner_radius
    vehicle_planner_pairs["tests"] = included_tests
    vehicle_planner_pairs["source_topology_radius_m"] = planner_radius
    vehicle_planner_pairs["radius_role"] = "source_topology_clearance_proxy"

    vehicle_ready_gap = copy.deepcopy(gap_diagnostics)
    vehicle_ready_gap["method"] = "p1_g1_3_vehicle_ready_gap_diagnostics_selection"
    vehicle_ready_gap["radius_m"] = gap_radius
    vehicle_ready_gap["records"] = included_negative_records
    vehicle_ready_gap["policy"] = {
        "gap_recomputed": False,
        "selection_only": True,
        "source": "frozen_gap_diagnostics",
    }

    return {
        "schema_version": 1,
        "method": "p1_g1_3_vehicle_planner_request_derivation",
        "source_topology_radius_m": planner_radius,
        "radius_role": "source_topology_clearance_proxy",
        "policy": {
            "map_reading": False,
            "map_editing": False,
            "pose_search": False,
            "yaw_recomputed": False,
            "gap_recomputed": False,
            "acceptance_thresholds": False,
            "pair_selection": "G1.2c pair_vehicle_ready only",
        },
        "sources": {
            "lateral_recovery_method": lateral_recovery.get("method"),
            "feasibility_audit_method": feasibility_audit.get("method"),
            "planner_pairs_method": planner_pairs.get("method"),
            "gap_diagnostics_method": gap_diagnostics.get("method"),
        },
        "summary": {
            "source_pair_side_count": source_count,
            "ready_pair_side_count": ready_count,
            "excluded_pair_side_count": source_count - ready_count,
            "positive_pair_side_count": positive_pair_sides,
            "negative_pair_side_count": negative_pair_sides,
            "directional_request_count": 2 * ready_count,
            "positive_requests": 2 * positive_pair_sides,
            "negative_requests": 2 * negative_pair_sides,
        },
        "pair_selection": pair_selection,
        "vehicle_planner_pairs": vehicle_planner_pairs,
        "vehicle_ready_gap_diagnostics": vehicle_ready_gap,
    }


def write_vehicle_planner_request_bundle(result, output_dir):
    """Write the frozen G1.3 compatibility bundle without overwriting results."""
    out = Path(output_dir).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    pairs_path = out / "vehicle_planner_pairs.yaml"
    pairs_path.write_text(
        yaml.safe_dump(result["vehicle_planner_pairs"], sort_keys=False),
        encoding="utf-8",
    )

    gap_path = out / "vehicle_ready_gap_diagnostics.json"
    gap_path.write_text(
        json.dumps(result["vehicle_ready_gap_diagnostics"], indent=2) + "\n",
        encoding="utf-8",
    )

    derivation_path = out / "vehicle_planner_request_derivation.json"
    derivation_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    selection_path = out / "vehicle_planner_pair_selection.csv"
    fields = [
        "case_id",
        "pair_id",
        "side",
        "expectation_class",
        "negative_reason",
        "included",
        "exclusion_reason",
        "first_anchor_id",
        "second_anchor_id",
        "max_longitudinal_inset_m",
        "max_abs_lateral_shift_m",
    ]
    with selection_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in result.get("pair_selection", []):
            writer.writerow({key: item.get(key) for key in fields})

    return {
        "vehicle_planner_pairs": pairs_path,
        "vehicle_ready_gap_diagnostics": gap_path,
        "derivation": derivation_path,
        "selection_csv": selection_path,
    }
