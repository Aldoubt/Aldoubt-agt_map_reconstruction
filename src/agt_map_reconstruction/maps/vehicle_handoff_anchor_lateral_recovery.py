"""Recover vehicle handoff anchors with bounded lateral search inside each aisle.

P1-G1.2b keeps the frozen map, topology anchors, aisle geometry, and vehicle
heading unchanged. Search proceeds at map-resolution stations moving only toward
the aisle interior. At each station, polygon-valid lateral offsets inside the
aisle footprint-feasible band are tested in increasing absolute offset order.
UNKNOWN and OCCUPIED remain blocking.
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
from .vehicle_handoff_anchor_recovery import (
    _aisle_axis_world,
    _aisle_lookup,
    _collect_topology_positions,
    _max_inward_distance,
    recover_vehicle_handoff_anchors,
)


def _aisle_lateral_band_world(rectangle, axis, footprint_xy_m, metadata):
    """Return centre offsets that keep the fixed-yaw footprint inside the aisle.

    The aisle polygon is expressed in repository grid-cell coordinates. Because
    the vehicle yaw is fixed to the aisle axis, the robot-frame Y coordinates are
    exactly the footprint extents along the aisle normal.
    """
    polygon = np.asarray(rectangle.get("polygon_xy"), dtype=float)
    if polygon.shape != (4, 2) or not np.isfinite(polygon).all():
        raise ValueError(f"aisle {rectangle.get('label')} polygon_xy must be finite 4x2")

    world = np.asarray([metadata.grid_to_world(*point) for point in polygon], dtype=float)
    unit = np.asarray(axis["unit_world"], dtype=float)
    normal = np.asarray([-unit[1], unit[0]], dtype=float)
    origin = np.asarray(axis["start_world"], dtype=float)
    aisle_offsets = (world - origin) @ normal
    aisle_low = float(np.min(aisle_offsets))
    aisle_high = float(np.max(aisle_offsets))

    footprint = np.asarray(footprint_xy_m, dtype=float)
    footprint_low = float(np.min(footprint[:, 1]))
    footprint_high = float(np.max(footprint[:, 1]))

    low = aisle_low - footprint_low
    high = aisle_high - footprint_high
    if low > high + 1e-12:
        return None, normal
    return (float(low), float(high)), normal


def _lateral_candidates(low, high, step):
    """Map-resolution offsets ordered by |offset|, then negative before positive."""
    if step <= 0.0:
        raise ValueError("lateral search step must be > 0")
    maximum = max(abs(float(low)), abs(float(high)))
    count = int(math.floor(maximum / float(step) + 1e-9))
    values = []
    for index in range(-count, count + 1):
        value = float(index) * float(step)
        if value < float(low) - 1e-9 or value > float(high) + 1e-9:
            continue
        values.append(value)
    values.sort(key=lambda value: (abs(value), value))
    return values


def _anchor_index(result):
    return {str(item["anchor_id"]): item for item in result.get("anchors", [])}


def recover_lateral_vehicle_handoff_anchors(
    base_map,
    aisle_rectangles,
    planner_pairs,
    footprint_xy_m,
    metadata,
    *,
    footprint_name="robot",
):
    """Recover nearest fixed-yaw vehicle anchors with lateral aisle search.

    Selection is lexicographic and parameter-free beyond map resolution:
      1. minimum inward longitudinal inset;
      2. minimum absolute lateral shift.

    The longitudinal-only P1-G1.2 result is retained per anchor for direct
    comparison. No map cell, topology handoff, or yaw is modified.
    """
    grid = _canonical_map(base_map)
    footprint = _footprint(footprint_xy_m)
    if not isinstance(metadata, GridMetadata):
        raise TypeError("metadata must be GridMetadata")
    if (metadata.height, metadata.width) != grid.shape:
        raise ValueError("GridMetadata dimensions must match base_map")

    aisles = _aisle_lookup(aisle_rectangles)
    topology_positions = _collect_topology_positions(planner_pairs)
    longitudinal = recover_vehicle_handoff_anchors(
        grid,
        aisle_rectangles,
        planner_pairs,
        footprint,
        metadata,
        footprint_name=footprint_name,
    )
    longitudinal_by_id = _anchor_index(longitudinal)

    step = float(metadata.resolution)
    anchors = []

    for (label, side), topology_xy in sorted(topology_positions.items()):
        rectangle = aisles.get(label)
        if rectangle is None:
            raise ValueError(f"planner topology anchor references missing aisle: {label}")
        axis = _aisle_axis_world(rectangle, metadata)
        heading = float(axis["heading_rad"])
        anchor_id = f"{label}-{side}"
        long_item = longitudinal_by_id.get(anchor_id)
        if long_item is None:
            raise RuntimeError(f"missing longitudinal-only comparison for {anchor_id}")

        band, normal = _aisle_lateral_band_world(rectangle, axis, footprint, metadata)
        offsets = [] if band is None else _lateral_candidates(band[0], band[1], step)
        max_inset, inward = _max_inward_distance(topology_xy, axis, side)
        station_count_total = int(math.floor(max_inset / step + 1e-9)) + 1

        vehicle_anchor = None
        vehicle_check = None
        chosen_inset = None
        chosen_offset = None
        tested_pose_count = 0
        search_station_count = 0

        for station_index in range(station_count_total):
            inset = min(float(station_index) * step, max_inset)
            station_xy = topology_xy + inward * inset
            search_station_count += 1
            found_at_station = False
            for lateral_shift in offsets:
                candidate_xy = station_xy + normal * float(lateral_shift)
                pose = {
                    "x": float(candidate_xy[0]),
                    "y": float(candidate_xy[1]),
                    "yaw": heading,
                }
                check = _pose_footprint_preflight(grid, footprint, metadata, pose)
                tested_pose_count += 1
                if not check["valid"]:
                    continue
                vehicle_anchor = {
                    "x": pose["x"],
                    "y": pose["y"],
                    "heading_rad": heading,
                }
                vehicle_check = check
                chosen_inset = float(inset)
                chosen_offset = float(lateral_shift)
                found_at_station = True
                break
            if found_at_station:
                break

        if vehicle_anchor is None:
            recovery_status = "unavailable"
            chosen_inset = None
            chosen_offset = None
        elif chosen_inset <= 1e-12 and abs(chosen_offset) <= 1e-12:
            recovery_status = "already_valid"
        elif abs(chosen_offset) > 1e-12:
            recovery_status = "recovered_lateral"
        else:
            recovery_status = "recovered_longitudinal"

        long_inset = long_item.get("longitudinal_inset_m")
        current_available = vehicle_anchor is not None
        longitudinal_available = long_item.get("vehicle_anchor") is not None
        if current_available and longitudinal_available:
            inset_reduction = float(long_inset) - float(chosen_inset)
        else:
            inset_reduction = None

        anchors.append(
            {
                "anchor_id": anchor_id,
                "aisle_id": int(rectangle["aisle_id"]),
                "label": label,
                "side": side,
                "recovery_status": recovery_status,
                "topology_anchor": long_item.get("topology_anchor"),
                "topology_pose_class": long_item.get("topology_pose_class"),
                "topology_preflight": long_item.get("topology_preflight"),
                "longitudinal_only_status": long_item.get("recovery_status"),
                "longitudinal_only_anchor": long_item.get("vehicle_anchor"),
                "longitudinal_only_inset_m": long_inset,
                "vehicle_anchor": vehicle_anchor,
                "vehicle_pose_class": (
                    None if vehicle_check is None else vehicle_check.get("pose_class")
                ),
                "vehicle_preflight": vehicle_check,
                "longitudinal_inset_m": chosen_inset,
                "lateral_shift_m": chosen_offset,
                "yaw_delta_rad": (None if vehicle_anchor is None else 0.0),
                "search_station_step_m": step,
                "lateral_search_step_m": step,
                "max_axis_search_m": float(max_inset),
                "lateral_feasible_band_m": (
                    None if band is None else [float(band[0]), float(band[1])]
                ),
                "search_station_count": int(search_station_count),
                "tested_pose_count": int(tested_pose_count),
                "inset_reduction_m": inset_reduction,
            }
        )

    available = [item for item in anchors if item["vehicle_anchor"] is not None]
    unavailable = [item for item in anchors if item["vehicle_anchor"] is None]
    improved = [
        item
        for item in anchors
        if item["vehicle_anchor"] is not None
        and (
            item["longitudinal_only_anchor"] is None
            or (
                item["inset_reduction_m"] is not None
                and item["inset_reduction_m"] > 1e-9
            )
        )
    ]

    return {
        "schema_version": 1,
        "method": "p1_g1_2b_vehicle_handoff_anchor_lateral_recovery",
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
            "lateral_search": "within_aisle_footprint_feasible_band",
            "yaw_search": False,
            "map_editing": False,
            "selection_order": [
                "minimum_longitudinal_inset",
                "minimum_absolute_lateral_shift",
            ],
        },
        "summary": {
            "anchor_count": len(anchors),
            "available_count": len(available),
            "unavailable_count": len(unavailable),
            "already_valid_count": int(
                sum(item["recovery_status"] == "already_valid" for item in anchors)
            ),
            "recovered_longitudinal_count": int(
                sum(item["recovery_status"] == "recovered_longitudinal" for item in anchors)
            ),
            "recovered_lateral_count": int(
                sum(item["recovery_status"] == "recovered_lateral" for item in anchors)
            ),
            "improved_over_longitudinal_only_count": len(improved),
            "recovered_from_longitudinal_unavailable_count": int(
                sum(
                    item["vehicle_anchor"] is not None
                    and item["longitudinal_only_anchor"] is None
                    for item in anchors
                )
            ),
        },
        "anchors": anchors,
    }


def write_lateral_vehicle_handoff_anchor_recovery_bundle(result, output_dir):
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
        "longitudinal_only_status",
        "longitudinal_only_x",
        "longitudinal_only_y",
        "longitudinal_only_inset_m",
        "vehicle_x",
        "vehicle_y",
        "longitudinal_inset_m",
        "lateral_shift_m",
        "inset_reduction_m",
        "search_station_step_m",
        "lateral_search_step_m",
        "search_station_count",
        "tested_pose_count",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in result.get("anchors", []):
            topology = item.get("topology_anchor") or {}
            longitudinal_anchor = item.get("longitudinal_only_anchor") or {}
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
                    "longitudinal_only_status": item.get("longitudinal_only_status"),
                    "longitudinal_only_x": longitudinal_anchor.get("x"),
                    "longitudinal_only_y": longitudinal_anchor.get("y"),
                    "longitudinal_only_inset_m": item.get("longitudinal_only_inset_m"),
                    "vehicle_x": vehicle.get("x"),
                    "vehicle_y": vehicle.get("y"),
                    "longitudinal_inset_m": item.get("longitudinal_inset_m"),
                    "lateral_shift_m": item.get("lateral_shift_m"),
                    "inset_reduction_m": item.get("inset_reduction_m"),
                    "search_station_step_m": item.get("search_station_step_m"),
                    "lateral_search_step_m": item.get("lateral_search_step_m"),
                    "search_station_count": item.get("search_station_count"),
                    "tested_pose_count": item.get("tested_pose_count"),
                }
            )

    features = []
    for item in result.get("anchors", []):
        vehicle = item.get("vehicle_anchor")
        if vehicle is None:
            continue
        properties = {
            key: item.get(key)
            for key in (
                "anchor_id",
                "aisle_id",
                "label",
                "side",
                "recovery_status",
                "topology_pose_class",
                "longitudinal_only_status",
                "longitudinal_only_inset_m",
                "longitudinal_inset_m",
                "lateral_shift_m",
                "inset_reduction_m",
            )
        }
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(vehicle["x"]), float(vehicle["y"])],
                },
                "properties": properties,
            }
        )
    geojson_path = out / "vehicle_handoff_anchors.geojson"
    geojson_path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2) + "\n",
        encoding="utf-8",
    )

    return {"json": json_path, "csv": csv_path, "geojson": geojson_path}
