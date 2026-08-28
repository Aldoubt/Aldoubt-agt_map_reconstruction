"""Read-only polygon-footprint preflight for frozen planner request poses."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

from .grid_geometry import GridMetadata

OCCUPIED_VALUE = np.uint8(0)
UNKNOWN_VALUE = np.uint8(205)
FREE_VALUE = np.uint8(254)
NEGATIVE_BRIDGE_TYPES = {"mixed_bridge", "clearance_only_bridge"}
POSE_CLASSES = (
    "valid",
    "unknown_overlap",
    "occupied_overlap",
    "mixed_blocking_overlap",
    "out_of_bounds",
)


def _canonical_map(value):
    grid = np.asarray(value, dtype=np.uint8)
    if grid.ndim != 2:
        raise ValueError("base_map must be a 2D array")
    if not np.isin(grid, [OCCUPIED_VALUE, UNKNOWN_VALUE, FREE_VALUE]).all():
        raise ValueError("base_map contains unsupported gray values")
    return grid


def _footprint(value):
    polygon = np.asarray(value, dtype=float)
    if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
        raise ValueError("footprint_xy_m must be an Nx2 polygon with at least 3 vertices")
    if not np.isfinite(polygon).all():
        raise ValueError("footprint_xy_m must be finite")
    return polygon


def _radius(payload, name):
    try:
        radius = float(payload["radius_m"])
    except Exception as exc:
        raise ValueError(f"{name} radius_m is required") from exc
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError(f"{name} radius_m must be positive and finite")
    return radius


def _pose(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    result = {}
    for key in ("x", "y", "yaw"):
        try:
            result[key] = float(value[key])
        except Exception as exc:
            raise ValueError(f"{label} requires finite {key}") from exc
        if not math.isfinite(result[key]):
            raise ValueError(f"{label} requires finite {key}")
    return result


def _selected_requests(planner_pairs, gap_diagnostics):
    planner_radius = _radius(planner_pairs, "planner_pairs")
    diagnostic_radius = _radius(gap_diagnostics, "gap_diagnostics")
    if not math.isclose(planner_radius, diagnostic_radius, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("planner_pairs and gap_diagnostics radius_m must match")

    tests = planner_pairs.get("tests")
    records = gap_diagnostics.get("records")
    if not isinstance(tests, list) or not isinstance(records, list):
        raise ValueError("planner_pairs.tests and gap_diagnostics.records must be lists")

    by_case = {}
    for test in tests:
        case_id = str(test.get("id") or f"{test.get('pair_id')}-{test.get('side')}")
        if case_id in by_case:
            raise ValueError(f"duplicate planner pair test id: {case_id}")
        by_case[case_id] = test

    positives = [
        test
        for test in tests
        if bool(test.get("enabled")) and bool(test.get("conservative_connected"))
    ]

    negatives = []
    negative_reason = {}
    for record in records:
        bridge_type = str(record.get("bridge_type", ""))
        if (
            str(record.get("evaluation_status")) != "evaluated"
            or bool(record.get("strict_connected"))
            or bridge_type not in NEGATIVE_BRIDGE_TYPES
        ):
            continue
        case_id = f"{record.get('pair_id')}-{record.get('side')}"
        if case_id not in by_case:
            raise ValueError(f"negative diagnostic case {case_id} has no planner pair poses")
        if case_id in negative_reason:
            raise ValueError(f"duplicate negative diagnostic case: {case_id}")
        negatives.append(by_case[case_id])
        negative_reason[case_id] = bridge_type

    requests = []
    seen = set()
    for test, expectation_class in (
        [(item, "positive") for item in positives]
        + [(item, "negative_control") for item in negatives]
    ):
        case_id = str(test.get("id") or f"{test['pair_id']}-{test['side']}")
        for direction in ("forward", "reverse"):
            segment = test.get(direction)
            if not isinstance(segment, dict):
                raise ValueError(f"{case_id} is missing {direction} start/goal")
            request_id = f"{case_id}-{direction}"
            if request_id in seen:
                raise ValueError(f"duplicate planner request id: {request_id}")
            seen.add(request_id)
            requests.append(
                {
                    "request_id": request_id,
                    "case_id": case_id,
                    "pair_id": str(test["pair_id"]),
                    "side": str(test["side"]),
                    "direction": direction,
                    "radius_m": planner_radius,
                    "expectation_class": expectation_class,
                    "expected_success": expectation_class == "positive",
                    "negative_reason": (
                        None
                        if expectation_class == "positive"
                        else negative_reason[case_id]
                    ),
                    "start": _pose(segment.get("start"), f"{request_id} start"),
                    "goal": _pose(segment.get("goal"), f"{request_id} goal"),
                }
            )
    return planner_radius, requests


def _pose_footprint_preflight(base_map, footprint_xy_m, metadata, pose):
    yaw = float(pose["yaw"])
    c = math.cos(yaw)
    s = math.sin(yaw)
    rotation = np.asarray([[c, -s], [s, c]], dtype=float)
    world_vertices = np.asarray([pose["x"], pose["y"]], dtype=float) + (
        footprint_xy_m @ rotation.T
    )
    grid_vertices = np.asarray(
        [metadata.world_to_grid(x, y) for x, y in world_vertices], dtype=float
    )

    height, width = base_map.shape
    out_of_bounds = bool(
        np.any(grid_vertices[:, 0] < 0)
        or np.any(grid_vertices[:, 0] >= width)
        or np.any(grid_vertices[:, 1] < 0)
        or np.any(grid_vertices[:, 1] >= height)
    )

    mask = np.zeros(base_map.shape, dtype=np.uint8)
    points = np.rint(grid_vertices).astype(np.int32)
    cv2.fillPoly(mask, [points], 1)
    footprint_cells = mask.astype(bool)

    occupied_overlap = int(
        np.count_nonzero(footprint_cells & (base_map == OCCUPIED_VALUE))
    )
    unknown_overlap = int(
        np.count_nonzero(footprint_cells & (base_map == UNKNOWN_VALUE))
    )
    free_overlap = int(np.count_nonzero(footprint_cells & (base_map == FREE_VALUE)))

    if out_of_bounds:
        pose_class = "out_of_bounds"
    elif occupied_overlap and unknown_overlap:
        pose_class = "mixed_blocking_overlap"
    elif occupied_overlap:
        pose_class = "occupied_overlap"
    elif unknown_overlap:
        pose_class = "unknown_overlap"
    else:
        pose_class = "valid"

    return {
        "pose_class": pose_class,
        "valid": pose_class == "valid",
        "out_of_bounds": out_of_bounds,
        "footprint_cell_count": int(np.count_nonzero(footprint_cells)),
        "free_overlap_cell_count": free_overlap,
        "unknown_overlap_cell_count": unknown_overlap,
        "occupied_overlap_cell_count": occupied_overlap,
    }


def audit_planner_request_footprints(
    base_map,
    planner_pairs,
    gap_diagnostics,
    footprint_xy_m,
    metadata,
    *,
    footprint_name="robot",
):
    """Classify start/goal polygon-footprint validity for frozen planner requests.

    This is a read-only diagnostic. It selects the same positive and diagnostic
    negative requests as the runtime smoke test, but performs no path search and
    never edits map semantics or request poses.
    """
    grid = _canonical_map(base_map)
    polygon = _footprint(footprint_xy_m)
    if not isinstance(metadata, GridMetadata):
        raise TypeError("metadata must be GridMetadata")
    if (metadata.height, metadata.width) != grid.shape:
        raise ValueError("GridMetadata dimensions must match base_map")

    radius_m, selected = _selected_requests(planner_pairs, gap_diagnostics)
    requests = []
    pose_counts = {name: 0 for name in POSE_CLASSES}
    start_invalid = 0
    goal_invalid = 0
    request_valid = 0

    for request in selected:
        start = _pose_footprint_preflight(grid, polygon, metadata, request["start"])
        goal = _pose_footprint_preflight(grid, polygon, metadata, request["goal"])
        pose_counts[start["pose_class"]] += 1
        pose_counts[goal["pose_class"]] += 1
        start_invalid += int(not start["valid"])
        goal_invalid += int(not goal["valid"])
        valid = bool(start["valid"] and goal["valid"])
        request_valid += int(valid)

        item = dict(request)
        item.update(
            {
                "start_preflight": start,
                "goal_preflight": goal,
                "request_valid": valid,
            }
        )
        requests.append(item)

    summary = {
        "request_count": len(requests),
        "pose_count": 2 * len(requests),
        "positive_request_count": int(
            sum(item["expectation_class"] == "positive" for item in requests)
        ),
        "negative_request_count": int(
            sum(item["expectation_class"] == "negative_control" for item in requests)
        ),
        "request_valid_count": request_valid,
        "request_invalid_count": len(requests) - request_valid,
        "start_invalid_request_count": start_invalid,
        "goal_invalid_request_count": goal_invalid,
        "pose_class_counts": pose_counts,
    }

    return {
        "schema_version": 1,
        "method": "p1_g1_1_planner_request_footprint_preflight",
        "radius_m": radius_m,
        "grid": metadata.to_dict(),
        "footprint": {
            "name": str(footprint_name),
            "polygon_xy_m": polygon.tolist(),
        },
        "policy": {
            "unknown_blocking": True,
            "occupied_blocking": True,
            "map_editing": False,
            "pose_relocation": False,
            "request_selection": "runtime positive plus mixed/clearance diagnostic negatives",
        },
        "summary": summary,
        "requests": requests,
    }


def write_planner_request_footprint_preflight_bundle(result, output_dir):
    out = Path(output_dir).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "planner_request_footprint_preflight.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    csv_path = out / "planner_request_footprint_preflight.csv"
    fields = [
        "request_id",
        "pair_id",
        "side",
        "direction",
        "expectation_class",
        "negative_reason",
        "request_valid",
        "start_pose_class",
        "start_unknown_overlap_cell_count",
        "start_occupied_overlap_cell_count",
        "start_out_of_bounds",
        "goal_pose_class",
        "goal_unknown_overlap_cell_count",
        "goal_occupied_overlap_cell_count",
        "goal_out_of_bounds",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in result.get("requests", []):
            start = item.get("start_preflight") or {}
            goal = item.get("goal_preflight") or {}
            writer.writerow(
                {
                    "request_id": item.get("request_id"),
                    "pair_id": item.get("pair_id"),
                    "side": item.get("side"),
                    "direction": item.get("direction"),
                    "expectation_class": item.get("expectation_class"),
                    "negative_reason": item.get("negative_reason"),
                    "request_valid": item.get("request_valid"),
                    "start_pose_class": start.get("pose_class"),
                    "start_unknown_overlap_cell_count": start.get("unknown_overlap_cell_count"),
                    "start_occupied_overlap_cell_count": start.get("occupied_overlap_cell_count"),
                    "start_out_of_bounds": start.get("out_of_bounds"),
                    "goal_pose_class": goal.get("pose_class"),
                    "goal_unknown_overlap_cell_count": goal.get("unknown_overlap_cell_count"),
                    "goal_occupied_overlap_cell_count": goal.get("occupied_overlap_cell_count"),
                    "goal_out_of_bounds": goal.get("out_of_bounds"),
                }
            )

    return {"json": json_path, "csv": csv_path}
