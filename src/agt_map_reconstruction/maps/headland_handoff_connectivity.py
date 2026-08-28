"""Scoped adjacent-aisle connectivity through finite headland domains."""

from __future__ import annotations

import math

import cv2
import numpy as np
from scipy import ndimage

OCCUPIED_VALUE = np.uint8(0)
UNKNOWN_VALUE = np.uint8(205)
FREE_VALUE = np.uint8(254)


def _require_grid(name, value, shape=None, *, dtype=None):
    array = np.asarray(value, dtype=dtype)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2D")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must match map shape")
    return array


def _normalize_direction(name, value):
    direction = np.asarray(value, dtype=float).reshape(-1)
    if direction.size != 2:
        raise ValueError(f"{name} must contain two values")
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ValueError(f"{name} must be non-zero")
    return direction / norm


def _canonical_map(name, value, shape=None):
    grid = _require_grid(name, value, shape, dtype=np.uint8)
    if not np.isin(grid, [OCCUPIED_VALUE, UNKNOWN_VALUE, FREE_VALUE]).all():
        raise ValueError(f"{name} contains unsupported gray values")
    return grid


def _handoff_lookup(records):
    lookup = {}
    for item in records:
        label = str(item.get("label", ""))
        if not label:
            raise ValueError("handoff record missing label")
        if label in lookup:
            raise ValueError(f"duplicate handoff label: {label}")
        lookup[label] = item
    return lookup


def _sorted_aisles(aisles):
    result = [dict(item) for item in aisles]
    for item in result:
        if "aisle_id" not in item:
            raise ValueError("aisle missing aisle_id")
        if not str(item.get("label", "")):
            item["label"] = f"A{int(item['aisle_id']):02d}"
        centerline = np.asarray(item.get("centerline_xy"), dtype=float)
        if centerline.shape != (2, 2):
            raise ValueError(f"aisle {item['label']} centerline_xy must be 2x2")
    return sorted(result, key=lambda item: int(item["aisle_id"]))


def _safe_mask(base_map, resolution, radius_m):
    free = base_map == FREE_VALUE
    distance = ndimage.distance_transform_edt(free) * float(resolution)
    return free & (distance + 1e-12 >= float(radius_m))


def _depth_side_mask(payload, masks, side, shape):
    side_payload = dict(payload.get(side) or {})
    combined = np.zeros(shape, dtype=bool)
    for item in side_payload.get("bands") or []:
        key = str(item.get("mask_key", ""))
        if not key or key not in masks:
            raise ValueError(f"missing {side} depth mask: {key}")
        combined |= _require_grid(key, masks[key], shape, dtype=bool)

    boundary_key = str(
        side_payload.get("boundary_uncertainty_mask_key")
        or f"{side}_boundary_uncertainty"
    )
    if boundary_key not in masks:
        raise ValueError(f"missing {side} boundary mask: {boundary_key}")
    combined |= _require_grid(boundary_key, masks[boundary_key], shape, dtype=bool)

    unresolved = _require_grid(
        "structurally_unresolved_cross",
        masks.get("structurally_unresolved_cross", np.zeros(shape, dtype=bool)),
        shape,
        dtype=bool,
    )
    return combined & ~unresolved


def _cross_coordinate(aisle, cross_direction):
    centerline = np.asarray(aisle["centerline_xy"], dtype=float)
    return float((0.5 * (centerline[0] + centerline[1])) @ cross_direction)


def _pair_cross_window(shape, first, second, cross_direction, resolution, radius_m):
    yy, xx = np.indices(shape, dtype=float)
    v = xx * cross_direction[0] + yy * cross_direction[1]

    first_half = max(
        float(first.get("width_m", 0.0)) / (2.0 * float(resolution)),
        float(radius_m) / float(resolution),
    )
    second_half = max(
        float(second.get("width_m", 0.0)) / (2.0 * float(resolution)),
        float(radius_m) / float(resolution),
    )
    first_v = _cross_coordinate(first, cross_direction)
    second_v = _cross_coordinate(second, cross_direction)
    lower = min(first_v - first_half, second_v - second_half)
    upper = max(first_v + first_half, second_v + second_half)
    return (v >= lower - 1e-12) & (v <= upper + 1e-12)


def _approach_mask(shape, aisle, anchor_xy, side, resolution):
    centerline = np.asarray(aisle["centerline_xy"], dtype=float)
    endpoint = centerline[0] if side == "entry" else centerline[1]
    anchor = np.asarray(anchor_xy, dtype=float)
    if anchor.shape != (2,):
        raise ValueError("handoff grid_xy must contain two values")

    canvas = np.zeros(shape, dtype=np.uint8)
    p0 = tuple(np.rint(anchor).astype(int))
    p1 = tuple(np.rint(endpoint).astype(int))
    width_cells = max(
        1,
        int(math.ceil(float(aisle.get("width_m", 0.0)) / float(resolution))),
    )
    cv2.line(canvas, p0, p1, 1, thickness=width_cells, lineType=cv2.LINE_8)
    return canvas.astype(bool)


def _anchor_xy(record, side):
    if not record or str(record.get("status", "")) != "ok":
        return None
    pose = record.get(f"{side}_handoff") or {}
    xy = pose.get("grid_xy")
    if xy is None:
        return None
    array = np.asarray(xy, dtype=float)
    if array.shape != (2,):
        return None
    return array


def _point_component(labels, xy):
    x, y = np.rint(np.asarray(xy, dtype=float)).astype(int)
    if y < 0 or x < 0 or y >= labels.shape[0] or x >= labels.shape[1]:
        return 0
    return int(labels[y, x])


def _connectivity(safe, domain, first_xy, second_xy):
    labels, _ = ndimage.label(safe & domain)
    first_id = _point_component(labels, first_xy)
    second_id = _point_component(labels, second_xy)
    connected = first_id > 0 and first_id == second_id
    if first_id == 0 or second_id == 0:
        reason = "anchor_not_safe"
    elif connected:
        reason = "connected"
    else:
        reason = "disconnected_in_scoped_domain"
    return bool(connected), reason, int(first_id), int(second_id)


def analyze_headland_handoff_connectivity(
    baseline_map,
    conservative_map,
    aisles,
    baseline_handoffs,
    conservative_handoffs,
    depth_profile_payload,
    depth_masks,
    *,
    resolution,
    radius_m=0.20,
):
    """Compare baseline and conservative connectivity for consecutive aisle pairs.

    Connectivity anchors are frozen to the baseline row-core handoff cells so
    that map A/B differences are attributable to the conservative headland
    overlay rather than to moving start/goal anchors. The search domain is the
    finite side-specific headland envelope plus only the two approach corridors
    from those frozen handoffs to the corresponding raw aisle endpoints.
    """
    resolution = float(resolution)
    radius_m = float(radius_m)
    if not np.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("resolution must be finite and > 0")
    if not np.isfinite(radius_m) or radius_m < 0.0:
        raise ValueError("radius_m must be finite and >= 0")

    baseline = _canonical_map("baseline_map", baseline_map)
    conservative = _canonical_map("conservative_map", conservative_map, baseline.shape)
    shape = baseline.shape

    payload = dict(depth_profile_payload)
    expected_shape = tuple(int(v) for v in payload.get("grid_shape_yx", shape))
    if expected_shape != shape:
        raise ValueError("depth profile grid shape does not match maps")

    _normalize_direction("row_axis_direction", payload.get("row_axis_direction"))
    cross = _normalize_direction("cross_row_direction", payload.get("cross_row_direction"))
    side_masks = {
        side: _depth_side_mask(payload, depth_masks, side, shape)
        for side in ("entry", "exit")
    }

    ordered = _sorted_aisles(aisles)
    baseline_lookup = _handoff_lookup(baseline_handoffs)
    conservative_lookup = _handoff_lookup(conservative_handoffs)
    baseline_safe = _safe_mask(baseline, resolution, radius_m)
    conservative_safe = _safe_mask(conservative, resolution, radius_m)

    records = []
    adjacent_pairs = list(zip(ordered[:-1], ordered[1:]))
    for first, second in adjacent_pairs:
        first_label = str(first["label"])
        second_label = str(second["label"])
        pair_id = f"{first_label}-{second_label}"
        first_base = baseline_lookup.get(first_label)
        second_base = baseline_lookup.get(second_label)
        first_cons = conservative_lookup.get(first_label)
        second_cons = conservative_lookup.get(second_label)

        width_eligible = bool(
            first_base
            and second_base
            and first_base.get("width_clearance_eligible") is True
            and second_base.get("width_clearance_eligible") is True
        )

        pair_window = _pair_cross_window(
            shape, first, second, cross, resolution, radius_m
        )
        for side in ("entry", "exit"):
            first_anchor = _anchor_xy(first_base, side)
            second_anchor = _anchor_xy(second_base, side)

            if not width_eligible:
                evaluation_status = "width_ineligible"
            elif first_anchor is None or second_anchor is None:
                evaluation_status = "missing_baseline_handoff"
            else:
                evaluation_status = "evaluated"

            if first_anchor is not None and second_anchor is not None:
                approach = (
                    _approach_mask(shape, first, first_anchor, side, resolution)
                    | _approach_mask(shape, second, second_anchor, side, resolution)
                )
            else:
                approach = np.zeros(shape, dtype=bool)
            domain = (side_masks[side] & pair_window) | approach

            new_free = (
                (baseline != FREE_VALUE)
                & (conservative == FREE_VALUE)
                & domain
            )
            new_safe = (~baseline_safe) & conservative_safe & domain

            if evaluation_status == "evaluated":
                baseline_connected, baseline_reason, b1, b2 = _connectivity(
                    baseline_safe, domain, first_anchor, second_anchor
                )
                conservative_connected, conservative_reason, c1, c2 = _connectivity(
                    conservative_safe, domain, first_anchor, second_anchor
                )
            else:
                baseline_connected = False
                conservative_connected = False
                baseline_reason = evaluation_status
                conservative_reason = evaluation_status
                b1 = b2 = c1 = c2 = 0

            gained = bool((not baseline_connected) and conservative_connected)
            lost = bool(baseline_connected and (not conservative_connected))
            records.append(
                {
                    "pair_id": pair_id,
                    "first_aisle": first_label,
                    "second_aisle": second_label,
                    "side": side,
                    "radius_m": radius_m,
                    "width_clearance_eligible": width_eligible,
                    "evaluation_status": evaluation_status,
                    "anchor_source": "baseline_handoff",
                    "first_anchor_grid_xy": (
                        None if first_anchor is None else [float(v) for v in first_anchor]
                    ),
                    "second_anchor_grid_xy": (
                        None if second_anchor is None else [float(v) for v in second_anchor]
                    ),
                    "baseline_connected": bool(baseline_connected),
                    "conservative_connected": bool(conservative_connected),
                    "gained_by_trusted_overlay": gained,
                    "lost_by_trusted_overlay": lost,
                    "baseline_reason": baseline_reason,
                    "conservative_reason": conservative_reason,
                    "baseline_component_ids": [b1, b2],
                    "conservative_component_ids": [c1, c2],
                    "domain_cell_count": int(np.count_nonzero(domain)),
                    "new_free_cell_count_in_domain": int(np.count_nonzero(new_free)),
                    "new_safe_cell_count_in_domain": int(np.count_nonzero(new_safe)),
                    "first_conservative_handoff": (
                        None if first_cons is None else first_cons.get(f"{side}_handoff")
                    ),
                    "second_conservative_handoff": (
                        None if second_cons is None else second_cons.get(f"{side}_handoff")
                    ),
                }
            )

    evaluated = [item for item in records if item["evaluation_status"] == "evaluated"]
    return {
        "schema_version": 1,
        "method": "adjacent_aisle_scoped_headland_handoff_connectivity",
        "grid_shape_yx": list(shape),
        "resolution_m": resolution,
        "radius_m": radius_m,
        "adjacent_pair_count": len(adjacent_pairs),
        "pair_side_record_count": len(records),
        "connectivity_counts": {
            "baseline_connected": int(sum(item["baseline_connected"] for item in evaluated)),
            "conservative_connected": int(sum(item["conservative_connected"] for item in evaluated)),
            "gained_by_trusted_overlay": int(sum(item["gained_by_trusted_overlay"] for item in evaluated)),
            "lost_by_trusted_overlay": int(sum(item["lost_by_trusted_overlay"] for item in evaluated)),
            "width_ineligible": int(sum(item["evaluation_status"] == "width_ineligible" for item in records)),
            "missing_baseline_handoff": int(sum(item["evaluation_status"] == "missing_baseline_handoff" for item in records)),
        },
        "policy": {
            "adjacency": "consecutive aisle_id order only",
            "connectivity_anchor": "baseline clearance-safe handoff cell",
            "search_domain": (
                "side-specific finite headland depth/boundary envelope, unresolved "
                "cross strip excluded, plus only the two baseline-handoff approach corridors"
            ),
            "pair_cross_scope": "between the two adjacent aisle centerlines plus half-width/radius padding",
            "safe_definition": "free && distance_to_nonfree >= radius",
            "unknown_traversable": False,
            "map_editing": False,
            "semantic_promotion": False,
        },
        "pairs": records,
    }


def _normalize_angle(angle):
    return float((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)


def _pose_for_direction(handoff, side, *, outward):
    if not handoff:
        return None
    xy = handoff.get("map_xy_m")
    if xy is None:
        return None
    heading = float(handoff.get("heading_rad", 0.0))
    if side == "entry":
        yaw = heading + math.pi if outward else heading
    elif side == "exit":
        yaw = heading if outward else heading + math.pi
    else:
        raise ValueError("side must be entry or exit")
    return {
        "x": float(xy[0]),
        "y": float(xy[1]),
        "yaw": _normalize_angle(yaw),
    }


def build_planner_pairs(connectivity_result):
    tests = []
    for item in connectivity_result.get("pairs", []):
        side = str(item["side"])
        first = item.get("first_conservative_handoff")
        second = item.get("second_conservative_handoff")
        first_out = _pose_for_direction(first, side, outward=True)
        first_in = _pose_for_direction(first, side, outward=False)
        second_out = _pose_for_direction(second, side, outward=True)
        second_in = _pose_for_direction(second, side, outward=False)
        poses_valid = all(v is not None for v in (first_out, first_in, second_out, second_in))
        enabled = bool(
            item.get("evaluation_status") == "evaluated"
            and item.get("conservative_connected") is True
            and poses_valid
        )
        tests.append(
            {
                "id": f"{item['pair_id']}-{side}",
                "pair_id": item["pair_id"],
                "side": side,
                "radius_m": float(item["radius_m"]),
                "enabled": enabled,
                "baseline_connected": bool(item["baseline_connected"]),
                "conservative_connected": bool(item["conservative_connected"]),
                "gained_by_trusted_overlay": bool(item["gained_by_trusted_overlay"]),
                "forward": {"start": first_out, "goal": second_in},
                "reverse": {"start": second_out, "goal": first_in},
            }
        )
    return {
        "schema_version": 1,
        "method": "nav2_headland_adjacent_pair_smoke_tests",
        "radius_m": float(connectivity_result["radius_m"]),
        "bidirectional": True,
        "tests": tests,
    }


def build_headland_gates_geojson(connectivity_result):
    gates = {}
    for item in connectivity_result.get("pairs", []):
        side = str(item["side"])
        endpoints = (
            (str(item["first_aisle"]), item.get("first_anchor_grid_xy"), item.get("first_conservative_handoff")),
            (str(item["second_aisle"]), item.get("second_anchor_grid_xy"), item.get("second_conservative_handoff")),
        )
        for label, anchor, handoff in endpoints:
            key = (label, side)
            record = gates.setdefault(
                key,
                {
                    "label": label,
                    "side": side,
                    "baseline_anchor_grid_xy": anchor,
                    "conservative_handoff": handoff,
                    "adjacent_pairs": [],
                    "connected_pairs": [],
                    "gained_pairs": [],
                },
            )
            record["adjacent_pairs"].append(item["pair_id"])
            if item.get("conservative_connected"):
                record["connected_pairs"].append(item["pair_id"])
            if item.get("gained_by_trusted_overlay"):
                record["gained_pairs"].append(item["pair_id"])

    features = []
    for (label, side), item in sorted(gates.items()):
        handoff = item.get("conservative_handoff") or {}
        map_xy = handoff.get("map_xy_m")
        grid_xy = handoff.get("grid_xy")
        if map_xy is not None:
            geometry = {"type": "Point", "coordinates": [float(map_xy[0]), float(map_xy[1])]}
            coordinate_space = "map_m"
        elif grid_xy is not None:
            geometry = {"type": "Point", "coordinates": [float(grid_xy[0]), float(grid_xy[1])]}
            coordinate_space = "grid_cells"
        else:
            geometry = None
            coordinate_space = "unavailable"
        features.append(
            {
                "type": "Feature",
                "geometry": geometry,
                "properties": {
                    "label": label,
                    "side": side,
                    "coordinate_space": coordinate_space,
                    "baseline_anchor_grid_xy": item.get("baseline_anchor_grid_xy"),
                    "conservative_grid_xy": grid_xy,
                    "heading_rad": handoff.get("heading_rad"),
                    "clearance_m": handoff.get("clearance_m"),
                    "adjacent_pairs": sorted(set(item["adjacent_pairs"])),
                    "connected_pairs": sorted(set(item["connected_pairs"])),
                    "gained_pairs": sorted(set(item["gained_pairs"])),
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def write_headland_connectivity_bundle(
    connectivity_result,
    conservative_map,
    output_dir,
    *,
    source_map_yaml=None,
):
    import json
    from pathlib import Path

    import yaml

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    (output / "headland_connectivity.json").write_text(
        json.dumps(connectivity_result, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    geojson = build_headland_gates_geojson(connectivity_result)
    (output / "headland_gates.geojson").write_text(
        json.dumps(geojson, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    planner = build_planner_pairs(connectivity_result)
    if source_map_yaml is not None:
        planner["map_yaml"] = str(source_map_yaml)
    with (output / "planner_pairs.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(planner, stream, sort_keys=False)

    _plot_headland_connectivity(
        conservative_map,
        connectivity_result,
        output / "headland_connectivity.png",
    )
    return {
        "connectivity": connectivity_result,
        "geojson": geojson,
        "planner_pairs": planner,
    }


def _plot_headland_connectivity(base_map, result, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.asarray(base_map, dtype=np.uint8)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(grid, origin="lower", cmap="gray", vmin=0, vmax=255)
    for item in result.get("pairs", []):
        first = item.get("first_anchor_grid_xy")
        second = item.get("second_anchor_grid_xy")
        if first is None or second is None:
            continue
        linestyle = "-" if item.get("conservative_connected") else ":"
        linewidth = 2.0 if item.get("gained_by_trusted_overlay") else 1.0
        marker = "o" if item.get("side") == "entry" else "x"
        x = [float(first[0]), float(second[0])]
        y = [float(first[1]), float(second[1])]
        ax.plot(x, y, linestyle=linestyle, linewidth=linewidth, marker=marker)
    counts = result.get("connectivity_counts", {})
    ax.set_title(
        "Scoped headland handoff connectivity "
        f"(baseline={counts.get('baseline_connected', 0)}, "
        f"conservative={counts.get('conservative_connected', 0)}, "
        f"gained={counts.get('gained_by_trusted_overlay', 0)})"
    )
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
