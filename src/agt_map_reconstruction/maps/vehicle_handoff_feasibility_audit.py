"""Read-only feasibility audit for recovered vehicle handoff anchors.

P1-G1.2c consumes frozen P1-G1.2b recovery output and structural aisle / pair
artifacts.  It does not read or edit the map and performs no pose search.  The
audit separates geometric width incompatibility from failure to find a valid
pose inside the aisle-feasible band, while preserving recovery depth and lateral
shift as continuous diagnostics rather than acceptance thresholds.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


NEGATIVE_BRIDGE_TYPES = {"mixed_bridge", "clearance_only_bridge"}
FEASIBILITY_CLASSES = {
    "vehicle_anchor_valid",
    "footprint_wider_than_aisle",
    "no_map_valid_pose_in_aisle_band",
}


def _finite_float(value, label):
    try:
        result = float(value)
    except Exception as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _footprint_width(lateral_recovery):
    footprint = lateral_recovery.get("footprint") or {}
    polygon = footprint.get("polygon_xy_m")
    if not isinstance(polygon, list) or len(polygon) < 3:
        raise ValueError("lateral_recovery.footprint.polygon_xy_m must be a polygon")
    ys = []
    for index, point in enumerate(polygon):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"footprint point {index} must be [x, y]")
        ys.append(_finite_float(point[1], f"footprint point {index} y"))
    width = max(ys) - min(ys)
    if width <= 0.0:
        raise ValueError("vehicle footprint width must be > 0")
    return float(width)


def _aisle_lookup(aisle_rectangles):
    lookup = {}
    for item in aisle_rectangles:
        label = str(item.get("label") or f"A{int(item['aisle_id']):02d}")
        if label in lookup:
            raise ValueError(f"duplicate aisle label: {label}")
        width = _finite_float(item.get("width_m"), f"{label} width_m")
        length = _finite_float(item.get("length_m"), f"{label} length_m")
        if width <= 0.0 or length <= 0.0:
            raise ValueError(f"{label} width_m and length_m must be > 0")
        record = dict(item)
        record["label"] = label
        record["width_m"] = width
        record["length_m"] = length
        lookup[label] = record
    return lookup


def _anchor_audit(item, aisle, footprint_width_m):
    vehicle_anchor = item.get("vehicle_anchor")
    aisle_width_m = float(aisle["width_m"])
    aisle_length_m = float(aisle["length_m"])

    if aisle_width_m + 1e-12 < footprint_width_m:
        feasibility_class = "footprint_wider_than_aisle"
    elif vehicle_anchor is None:
        feasibility_class = "no_map_valid_pose_in_aisle_band"
    else:
        feasibility_class = "vehicle_anchor_valid"

    inset = item.get("longitudinal_inset_m")
    if inset is None:
        inset_m = None
        inset_ratio = None
    else:
        inset_m = _finite_float(inset, f"{item.get('anchor_id')} longitudinal_inset_m")
        inset_ratio = inset_m / aisle_length_m

    lateral = item.get("lateral_shift_m")
    lateral_m = None if lateral is None else _finite_float(
        lateral, f"{item.get('anchor_id')} lateral_shift_m"
    )

    band = item.get("lateral_feasible_band_m")
    if band is None:
        band_width = None
    else:
        if not isinstance(band, (list, tuple)) or len(band) != 2:
            raise ValueError(f"{item.get('anchor_id')} lateral_feasible_band_m must be [low, high]")
        low = _finite_float(band[0], f"{item.get('anchor_id')} lateral band low")
        high = _finite_float(band[1], f"{item.get('anchor_id')} lateral band high")
        band_width = max(0.0, high - low)

    result = dict(item)
    result.update(
        {
            "feasibility_class": feasibility_class,
            "vehicle_ready": feasibility_class == "vehicle_anchor_valid",
            "footprint_width_m": float(footprint_width_m),
            "aisle_width_m": aisle_width_m,
            "aisle_length_m": aisle_length_m,
            "longitudinal_inset_m": inset_m,
            "inset_over_aisle_length": inset_ratio,
            "lateral_shift_m": lateral_m,
            "lateral_feasible_band_width_m": band_width,
        }
    )
    return result


def _selected_pair_sides(planner_pairs, gap_diagnostics):
    tests = planner_pairs.get("tests")
    records = gap_diagnostics.get("records")
    if not isinstance(tests, list) or not isinstance(records, list):
        raise ValueError("planner_pairs.tests and gap_diagnostics.records must be lists")

    planner_radius = planner_pairs.get("radius_m")
    gap_radius = gap_diagnostics.get("radius_m")
    if planner_radius is not None and gap_radius is not None:
        a = _finite_float(planner_radius, "planner_pairs radius_m")
        b = _finite_float(gap_radius, "gap_diagnostics radius_m")
        if not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("planner_pairs and gap_diagnostics radius_m must match")

    by_case = {}
    for test in tests:
        case_id = str(test.get("id") or f"{test.get('pair_id')}-{test.get('side')}")
        if case_id in by_case:
            raise ValueError(f"duplicate planner test id: {case_id}")
        by_case[case_id] = test

    selected = []
    for test in tests:
        if bool(test.get("enabled")) and bool(test.get("conservative_connected")):
            case_id = str(test.get("id") or f"{test.get('pair_id')}-{test.get('side')}")
            selected.append((case_id, test, "positive", None))

    seen_negative = set()
    for record in records:
        bridge_type = str(record.get("bridge_type", ""))
        if (
            str(record.get("evaluation_status")) != "evaluated"
            or bool(record.get("strict_connected"))
            or bridge_type not in NEGATIVE_BRIDGE_TYPES
        ):
            continue
        case_id = f"{record.get('pair_id')}-{record.get('side')}"
        if case_id in seen_negative:
            raise ValueError(f"duplicate diagnostic negative case: {case_id}")
        seen_negative.add(case_id)
        test = by_case.get(case_id)
        if test is None:
            raise ValueError(f"diagnostic negative case has no planner pair: {case_id}")
        selected.append((case_id, test, "negative_control", bridge_type))

    return selected


def audit_vehicle_handoff_feasibility(
    lateral_recovery,
    aisle_rectangles,
    planner_pairs,
    gap_diagnostics,
):
    """Audit G1.2b anchors and pair-side readiness without searching new poses."""
    if not isinstance(lateral_recovery, dict):
        raise TypeError("lateral_recovery must be an object")
    anchors_source = lateral_recovery.get("anchors")
    if not isinstance(anchors_source, list):
        raise ValueError("lateral_recovery.anchors must be a list")

    footprint_width_m = _footprint_width(lateral_recovery)
    aisles = _aisle_lookup(aisle_rectangles)

    anchors = []
    anchor_by_id = {}
    for source in anchors_source:
        anchor_id = str(source.get("anchor_id", ""))
        label = str(source.get("label", ""))
        side = str(source.get("side", ""))
        if not anchor_id or not label or side not in {"entry", "exit"}:
            raise ValueError("each lateral recovery anchor needs anchor_id, label, entry/exit side")
        if anchor_id in anchor_by_id:
            raise ValueError(f"duplicate lateral recovery anchor: {anchor_id}")
        aisle = aisles.get(label)
        if aisle is None:
            raise ValueError(f"lateral recovery references missing aisle: {label}")
        audited = _anchor_audit(source, aisle, footprint_width_m)
        anchors.append(audited)
        anchor_by_id[anchor_id] = audited

    pair_sides = []
    for case_id, test, expectation_class, negative_reason in _selected_pair_sides(
        planner_pairs, gap_diagnostics
    ):
        pair_id = str(test.get("pair_id", ""))
        labels = pair_id.split("-")
        if len(labels) != 2 or not all(labels):
            raise ValueError(f"invalid pair_id: {pair_id}")
        first_label, second_label = labels
        side = str(test.get("side", ""))
        if side not in {"entry", "exit"}:
            raise ValueError(f"invalid pair side for {case_id}: {side}")

        first_id = f"{first_label}-{side}"
        second_id = f"{second_label}-{side}"
        first = anchor_by_id.get(first_id)
        second = anchor_by_id.get(second_id)
        if first is None or second is None:
            raise ValueError(f"pair {case_id} references missing G1.2b anchor")

        pair_ready = bool(first["vehicle_ready"] and second["vehicle_ready"])
        valid_anchors = [item for item in (first, second) if item["vehicle_ready"]]
        if len(valid_anchors) == 2:
            max_inset = max(float(item["longitudinal_inset_m"]) for item in valid_anchors)
            max_lateral = max(abs(float(item["lateral_shift_m"])) for item in valid_anchors)
        else:
            max_inset = None
            max_lateral = None

        pair_sides.append(
            {
                "case_id": case_id,
                "pair_id": pair_id,
                "side": side,
                "expectation_class": expectation_class,
                "negative_reason": negative_reason,
                "first_anchor_id": first_id,
                "second_anchor_id": second_id,
                "first_anchor_class": first["feasibility_class"],
                "second_anchor_class": second["feasibility_class"],
                "first_anchor_ready": bool(first["vehicle_ready"]),
                "second_anchor_ready": bool(second["vehicle_ready"]),
                "pair_vehicle_ready": pair_ready,
                "max_longitudinal_inset_m": max_inset,
                "max_abs_lateral_shift_m": max_lateral,
            }
        )

    valid_anchor_count = sum(item["vehicle_ready"] for item in anchors)
    width_failure_count = sum(
        item["feasibility_class"] == "footprint_wider_than_aisle" for item in anchors
    )
    map_band_failure_count = sum(
        item["feasibility_class"] == "no_map_valid_pose_in_aisle_band" for item in anchors
    )
    ready_pair_count = sum(item["pair_vehicle_ready"] for item in pair_sides)

    return {
        "schema_version": 1,
        "method": "p1_g1_2c_vehicle_handoff_feasibility_audit",
        "footprint": lateral_recovery.get("footprint"),
        "policy": {
            "pose_search": False,
            "map_reading": False,
            "map_editing": False,
            "unknown_reclassification": False,
            "acceptance_thresholds": False,
            "continuous_recovery_metrics_only": True,
            "pair_selection": "runtime positives plus mixed/clearance diagnostic negatives",
        },
        "summary": {
            "anchor_count": len(anchors),
            "vehicle_anchor_valid_count": int(valid_anchor_count),
            "footprint_wider_than_aisle_count": int(width_failure_count),
            "no_map_valid_pose_in_aisle_band_count": int(map_band_failure_count),
            "pair_side_count": len(pair_sides),
            "positive_pair_side_count": int(
                sum(item["expectation_class"] == "positive" for item in pair_sides)
            ),
            "negative_pair_side_count": int(
                sum(item["expectation_class"] == "negative_control" for item in pair_sides)
            ),
            "pair_vehicle_ready_count": int(ready_pair_count),
            "pair_vehicle_not_ready_count": int(len(pair_sides) - ready_pair_count),
        },
        "anchors": anchors,
        "pair_sides": pair_sides,
    }


def write_vehicle_handoff_feasibility_bundle(result, output_dir):
    out = Path(output_dir).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "vehicle_handoff_feasibility.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    anchors_csv = out / "vehicle_handoff_anchor_feasibility.csv"
    anchor_fields = [
        "anchor_id", "aisle_id", "label", "side", "feasibility_class",
        "vehicle_ready", "recovery_status", "footprint_width_m", "aisle_width_m",
        "aisle_length_m", "longitudinal_inset_m", "inset_over_aisle_length",
        "lateral_shift_m", "lateral_feasible_band_width_m",
    ]
    with anchors_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=anchor_fields)
        writer.writeheader()
        for item in result.get("anchors", []):
            writer.writerow({key: item.get(key) for key in anchor_fields})

    pairs_csv = out / "vehicle_handoff_pair_feasibility.csv"
    pair_fields = [
        "case_id", "pair_id", "side", "expectation_class", "negative_reason",
        "first_anchor_id", "second_anchor_id", "first_anchor_class",
        "second_anchor_class", "first_anchor_ready", "second_anchor_ready",
        "pair_vehicle_ready", "max_longitudinal_inset_m", "max_abs_lateral_shift_m",
    ]
    with pairs_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=pair_fields)
        writer.writeheader()
        for item in result.get("pair_sides", []):
            writer.writerow({key: item.get(key) for key in pair_fields})

    return {"json": json_path, "anchors_csv": anchors_csv, "pairs_csv": pairs_csv}
