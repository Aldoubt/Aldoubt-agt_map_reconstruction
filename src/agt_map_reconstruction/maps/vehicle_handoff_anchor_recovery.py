"""Recover polygon-valid vehicle handoff anchors by longitudinal inset only.

P1-G1.2 deliberately keeps the frozen navigation map and topology handoffs
unchanged.  Each unique aisle-side topology anchor is projected onto the
recovered aisle axis and tested at map-resolution increments moving only toward
that aisle's interior.  UNKNOWN and OCCUPIED remain blocking; no lateral or yaw
search is performed.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from .grid_geometry import GridMetadata
from .planner_request_footprint_preflight import (
    _canonical_map,
    _footprint,
    _pose_footprint_preflight,
)


def _aisle_lookup(aisles):
    lookup = {}
    for item in aisles:
        label = str(item.get("label") or f"A{int(item['aisle_id']):02d}")
        if label in lookup:
            raise ValueError(f"duplicate aisle label: {label}")
        rectangle = dict(item)
        rectangle["label"] = label
        lookup[label] = rectangle
    return lookup


def _aisle_axis_world(rectangle, metadata):
    polygon = np.asarray(rectangle.get("polygon_xy"), dtype=float)
    if polygon.shape != (4, 2):
        raise ValueError(f"aisle {rectangle.get('label')} polygon_xy must be 4x2")
    if not np.isfinite(polygon).all():
        raise ValueError(f"aisle {rectangle.get('label')} polygon_xy must be finite")

    start_grid = 0.5 * (polygon[0] + polygon[3])
    end_grid = 0.5 * (polygon[1] + polygon[2])
    start_world = np.asarray(metadata.grid_to_world(*start_grid), dtype=float)
    end_world = np.asarray(metadata.grid_to_world(*end_grid), dtype=float)
    delta = end_world - start_world
    length_m = float(np.linalg.norm(delta))
    if length_m <= 1e-12:
        raise ValueError(f"aisle {rectangle.get('label')} centerline has zero length")
    unit = delta / length_m
    return {
        "start_world": start_world,
        "end_world": end_world,
        "unit_world": unit,
        "length_m": length_m,
        "heading_rad": float(math.atan2(unit[1], unit[0])),
    }


def _position(value, label):
    if not isinstance(value, dict):
        return None
    try:
        x = float(value["x"])
        y = float(value["y"])
    except Exception:
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError(f"{label} position must be finite")
    return np.asarray([x, y], dtype=float)


def _collect_topology_positions(planner_pairs):
    tests = planner_pairs.get("tests")
    if not isinstance(tests, list):
        raise ValueError("planner_pairs.tests must be a list")

    occurrences = {}
    for test in tests:
        pair_id = str(test.get("pair_id", ""))
        labels = pair_id.split("-")
        if len(labels) != 2 or not all(labels):
            raise ValueError(f"invalid planner pair_id: {pair_id}")
        first, second = labels
        side = str(test.get("side", ""))
        if side not in {"entry", "exit"}:
            raise ValueError(f"planner pair {pair_id} has invalid side: {side}")

        forward = test.get("forward") or {}
        reverse = test.get("reverse") or {}
        mapping = (
            (first, _position(forward.get("start"), f"{pair_id}-{side} forward start")),
            (second, _position(forward.get("goal"), f"{pair_id}-{side} forward goal")),
            (second, _position(reverse.get("start"), f"{pair_id}-{side} reverse start")),
            (first, _position(reverse.get("goal"), f"{pair_id}-{side} reverse goal")),
        )
        for label, xy in mapping:
            if xy is None:
                continue
            occurrences.setdefault((label, side), []).append(xy)

    anchors = {}
    for key, values in occurrences.items():
        reference = values[0]
        for value in values[1:]:
            if float(np.linalg.norm(value - reference)) > 1e-6:
                raise ValueError(
                    f"inconsistent planner handoff positions for {key[0]}-{key[1]}"
                )
        anchors[key] = reference
    return anchors


def _max_inward_distance(topology_xy, axis, side):
    target = axis["end_world"] if side == "entry" else axis["start_world"]
    inward = axis["unit_world"] if side == "entry" else -axis["unit_world"]
    projected = float(np.dot(target - topology_xy, inward))
    return max(0.0, projected), inward


def recover_vehicle_handoff_anchors(
    base_map,
    aisle_rectangles,
    planner_pairs,
    footprint_xy_m,
    metadata,
    *,
    footprint_name="robot",
):
    """Recover nearest vehicle-valid handoff poses along each aisle axis.

    Search is intentionally parameter-free beyond the frozen map resolution:
    candidates are tested at 0, resolution, 2*resolution, ... from each frozen
    topology handoff toward the aisle interior.  The aisle centerline and
    heading are fixed, therefore lateral_shift_m and yaw_delta_rad are always 0.
    """
    grid = _canonical_map(base_map)
    footprint = _footprint(footprint_xy_m)
    if not isinstance(metadata, GridMetadata):
        raise TypeError("metadata must be GridMetadata")
    if (metadata.height, metadata.width) != grid.shape:
        raise ValueError("GridMetadata dimensions must match base_map")

    aisles = _aisle_lookup(aisle_rectangles)
    topology_positions = _collect_topology_positions(planner_pairs)
    anchors = []

    for (label, side), topology_xy in sorted(topology_positions.items()):
        rectangle = aisles.get(label)
        if rectangle is None:
            raise ValueError(f"planner topology anchor references missing aisle: {label}")
        axis = _aisle_axis_world(rectangle, metadata)
        heading = axis["heading_rad"]
        topology_pose = {
            "x": float(topology_xy[0]),
            "y": float(topology_xy[1]),
            "heading_rad": heading,
        }
        topology_check = _pose_footprint_preflight(
            grid,
            footprint,
            metadata,
            {"x": topology_pose["x"], "y": topology_pose["y"], "yaw": heading},
        )

        max_inset, inward = _max_inward_distance(topology_xy, axis, side)
        step = float(metadata.resolution)
        candidate_count = int(math.floor(max_inset / step + 1e-9)) + 1

        recovered_pose = None
        recovered_check = None
        recovered_inset = None
        tested_count = 0
        for index in range(candidate_count):
            inset = min(float(index) * step, max_inset)
            candidate_xy = topology_xy + inward * inset
            pose = {
                "x": float(candidate_xy[0]),
                "y": float(candidate_xy[1]),
                "yaw": heading,
            }
            check = _pose_footprint_preflight(grid, footprint, metadata, pose)
            tested_count += 1
            if check["valid"]:
                recovered_pose = {
                    "x": pose["x"],
                    "y": pose["y"],
                    "heading_rad": heading,
                }
                recovered_check = check
                recovered_inset = inset
                break

        if recovered_pose is None:
            recovery_status = "unavailable"
            inset_value = None
        elif recovered_inset <= 1e-12:
            recovery_status = "already_valid"
            inset_value = 0.0
        else:
            recovery_status = "recovered"
            inset_value = float(recovered_inset)

        anchors.append(
            {
                "anchor_id": f"{label}-{side}",
                "aisle_id": int(rectangle["aisle_id"]),
                "label": label,
                "side": side,
                "recovery_status": recovery_status,
                "topology_anchor": topology_pose,
                "topology_pose_class": topology_check["pose_class"],
                "topology_preflight": topology_check,
                "vehicle_anchor": recovered_pose,
                "vehicle_pose_class": (
                    None if recovered_check is None else recovered_check["pose_class"]
                ),
                "vehicle_preflight": recovered_check,
                "longitudinal_inset_m": inset_value,
                "search_step_m": step,
                "max_axis_search_m": float(max_inset),
                "tested_pose_count": tested_count,
                "lateral_shift_m": 0.0,
                "yaw_delta_rad": 0.0,
            }
        )

    available = [item for item in anchors if item["recovery_status"] != "unavailable"]
    recovered = [item for item in anchors if item["recovery_status"] == "recovered"]
    already_valid = [item for item in anchors if item["recovery_status"] == "already_valid"]
    unavailable = [item for item in anchors if item["recovery_status"] == "unavailable"]

    return {
        "schema_version": 1,
        "method": "p1_g1_2_vehicle_handoff_anchor_recovery",
        "grid": metadata.to_dict(),
        "footprint": {
            "name": str(footprint_name),
            "polygon_xy_m": footprint.tolist(),
        },
        "policy": {
            "unknown_blocking": True,
            "occupied_blocking": True,
            "search_step_source": "map_resolution",
            "longitudinal_search": "inward_only",
            "lateral_search": False,
            "yaw_search": False,
            "map_editing": False,
            "nearest_valid_candidate": True,
        },
        "summary": {
            "anchor_count": len(anchors),
            "available_count": len(available),
            "unavailable_count": len(unavailable),
            "already_valid_count": len(already_valid),
            "recovered_count": len(recovered),
        },
        "anchors": anchors,
    }


def write_vehicle_handoff_anchor_recovery_bundle(result, output_dir):
    out = Path(output_dir).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "vehicle_handoff_anchors.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    csv_path = out / "vehicle_handoff_anchors.csv"
    fields = [
        "anchor_id",
        "aisle_id",
        "label",
        "side",
        "recovery_status",
        "topology_pose_class",
        "topology_x",
        "topology_y",
        "heading_rad",
        "vehicle_x",
        "vehicle_y",
        "longitudinal_inset_m",
        "search_step_m",
        "max_axis_search_m",
        "tested_pose_count",
        "lateral_shift_m",
        "yaw_delta_rad",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in result.get("anchors", []):
            topology = item.get("topology_anchor") or {}
            vehicle = item.get("vehicle_anchor") or {}
            writer.writerow(
                {
                    "anchor_id": item.get("anchor_id"),
                    "aisle_id": item.get("aisle_id"),
                    "label": item.get("label"),
                    "side": item.get("side"),
                    "recovery_status": item.get("recovery_status"),
                    "topology_pose_class": item.get("topology_pose_class"),
                    "topology_x": topology.get("x"),
                    "topology_y": topology.get("y"),
                    "heading_rad": topology.get("heading_rad"),
                    "vehicle_x": vehicle.get("x"),
                    "vehicle_y": vehicle.get("y"),
                    "longitudinal_inset_m": item.get("longitudinal_inset_m"),
                    "search_step_m": item.get("search_step_m"),
                    "max_axis_search_m": item.get("max_axis_search_m"),
                    "tested_pose_count": item.get("tested_pose_count"),
                    "lateral_shift_m": item.get("lateral_shift_m"),
                    "yaw_delta_rad": item.get("yaw_delta_rad"),
                }
            )

    features = []
    for item in result.get("anchors", []):
        vehicle = item.get("vehicle_anchor")
        topology = item.get("topology_anchor") or {}
        point = vehicle if vehicle is not None else topology
        if "x" not in point or "y" not in point:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(point["x"]), float(point["y"])],
                },
                "properties": {
                    "anchor_id": item.get("anchor_id"),
                    "label": item.get("label"),
                    "side": item.get("side"),
                    "recovery_status": item.get("recovery_status"),
                    "topology_pose_class": item.get("topology_pose_class"),
                    "longitudinal_inset_m": item.get("longitudinal_inset_m"),
                    "coordinate_role": (
                        "vehicle_anchor" if vehicle is not None else "topology_anchor"
                    ),
                },
            }
        )
    geojson_path = out / "vehicle_handoff_anchors.geojson"
    geojson_path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2)
        + "\n",
        encoding="utf-8",
    )

    return {"json": json_path, "csv": csv_path, "geojson": geojson_path}
