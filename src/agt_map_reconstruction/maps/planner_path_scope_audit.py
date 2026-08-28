"""Audit Nav2 planner paths against frozen P1 headland topology scopes."""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from .grid_geometry import GridMetadata
from .headland_gap_diagnostics import _aisle_lookup, _domain_parts
from .headland_handoff_connectivity import (
    FREE_VALUE,
    OCCUPIED_VALUE,
    UNKNOWN_VALUE,
    _canonical_map,
    _depth_side_mask,
    _normalize_direction,
    _safe_mask,
)


def _finite_positive(value, name, *, allow_zero=False):
    value = float(value)
    if not math.isfinite(value) or value < 0.0 or (not allow_zero and value <= 0.0):
        relation = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{name} must be finite and {relation}")
    return value


def _radius_matches(payload, radius_m, name):
    if "radius_m" not in payload:
        return
    other = float(payload["radius_m"])
    if not math.isclose(other, radius_m, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{name} radius_m {other} does not match requested radius {radius_m}")


def _point_component(labels, xy):
    x, y = np.rint(np.asarray(xy, dtype=float)).astype(int)
    if y < 0 or x < 0 or y >= labels.shape[0] or x >= labels.shape[1]:
        return 0
    return int(labels[y, x])


def _connected(mask, domain, first_xy, second_xy, *, connectivity):
    if connectivity == 4:
        structure = ndimage.generate_binary_structure(2, 1)
    elif connectivity == 8:
        structure = ndimage.generate_binary_structure(2, 2)
    else:
        raise ValueError("connectivity must be 4 or 8")
    labels, _ = ndimage.label(np.asarray(mask, dtype=bool) & np.asarray(domain, dtype=bool), structure=structure)
    first_id = _point_component(labels, first_xy)
    second_id = _point_component(labels, second_xy)
    return bool(first_id > 0 and first_id == second_id)


def _line_cells(x0, y0, x1, y1):
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        yield x0, y0
        if x0 == x1 and y0 == y1:
            break
        twice = 2 * err
        if twice >= dy:
            err += dy
            x0 += sx
        if twice <= dx:
            err += dx
            y0 += sy


def _rasterize_world_path(path_xy, metadata):
    points = []
    for index, point in enumerate(path_xy or []):
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            raise ValueError(f"path point {index} must contain x and y")
        x, y = float(point[0]), float(point[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"path point {index} must be finite")
        points.append((x, y))
    if not points:
        return set()
    grid = [metadata.world_to_grid(x, y) for x, y in points]
    cells = {grid[0]}
    for first, second in zip(grid, grid[1:]):
        cells.update(_line_cells(first[0], first[1], second[0], second[1]))
    return cells


def _path_length(path_xy):
    points = [(float(item[0]), float(item[1])) for item in (path_xy or [])]
    return float(
        sum(
            math.hypot(b[0] - a[0], b[1] - a[1])
            for a, b in zip(points, points[1:])
        )
    )


def _direct_distance(result):
    start = dict(result.get("start") or {})
    goal = dict(result.get("goal") or {})
    try:
        sx, sy = float(start["x"]), float(start["y"])
        gx, gy = float(goal["x"]), float(goal["y"])
    except Exception as exc:
        raise ValueError(f"{result.get('request_id')} requires start/goal x,y") from exc
    values = (sx, sy, gx, gy)
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"{result.get('request_id')} start/goal must be finite")
    return float(math.hypot(gx - sx, gy - sy))


def _classify(*, frozen_positive, planner_success, infrastructure_error, pair_contained, strict8, strict4_matches):
    if strict4_matches is False:
        return "frozen_topology_recompute_mismatch"
    if infrastructure_error:
        return "infrastructure_error"
    if not planner_success:
        return "positive_no_plan" if frozen_positive else "negative_no_plan"
    if frozen_positive:
        return "positive_local_match" if pair_contained else "positive_global_detour"
    if not pair_contained:
        return "negative_global_detour"
    if strict8:
        return "negative_local_8connect_match"
    return "negative_local_clearance_mismatch"


def analyze_planner_path_scope_audit(
    conservative_map,
    aisles,
    connectivity_result,
    depth_profile_payload,
    depth_masks,
    planner_results,
    *,
    metadata,
    resolution,
    radius_m=0.20,
):
    """Explain planner/topology agreement using the frozen pair-side domains.

    P1-F1 remains a 4-connected clearance-safe component test. P1-F3 does not
    modify that result; it recomputes the same 4-connected contract, adds an
    8-connected diagnostic aligned with grid-planner neighbor semantics, and
    audits successful Nav2 paths for pair-domain / finite-headland containment.
    """
    resolution = _finite_positive(resolution, "resolution")
    radius_m = _finite_positive(radius_m, "radius_m", allow_zero=True)
    if not isinstance(metadata, GridMetadata):
        raise TypeError("metadata must be GridMetadata")
    if not math.isclose(float(metadata.resolution), resolution, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("metadata resolution does not match resolution")

    base = _canonical_map("conservative_map", conservative_map)
    shape = base.shape
    if (int(metadata.height), int(metadata.width)) != shape:
        raise ValueError("metadata dimensions do not match map")

    profile = dict(depth_profile_payload)
    expected_shape = tuple(int(v) for v in profile.get("grid_shape_yx", shape))
    if expected_shape != shape:
        raise ValueError("depth profile grid shape does not match map")
    _radius_matches(dict(connectivity_result), radius_m, "connectivity")
    _radius_matches(dict(planner_results), radius_m, "planner results")

    cross = _normalize_direction("cross_row_direction", profile.get("cross_row_direction"))
    side_masks = {
        side: _depth_side_mask(profile, depth_masks, side, shape)
        for side in ("entry", "exit")
    }
    aisle_by_label = _aisle_lookup(aisles)
    safe = _safe_mask(base, resolution, radius_m)
    free = base == FREE_VALUE
    clearance = ndimage.distance_transform_edt(free) * resolution

    source_pairs = {}
    for item in connectivity_result.get("pairs", []):
        key = (str(item.get("pair_id")), str(item.get("side")))
        if key in source_pairs:
            raise ValueError(f"duplicate connectivity pair-side: {key[0]} {key[1]}")
        source_pairs[key] = dict(item)

    records = []
    for planner_item in planner_results.get("results", []):
        item = dict(planner_item)
        key = (str(item.get("pair_id")), str(item.get("side")))
        source = source_pairs.get(key)
        if source is None:
            raise ValueError(f"planner result has no connectivity record: {key[0]} {key[1]}")

        pair_domain, finite_headland_domain = _domain_parts(
            shape,
            source,
            aisle_by_label,
            side_masks,
            cross,
            resolution,
            radius_m,
        )
        first_anchor = source.get("first_anchor_grid_xy")
        second_anchor = source.get("second_anchor_grid_xy")
        evaluated = str(source.get("evaluation_status")) == "evaluated"
        if evaluated and first_anchor is not None and second_anchor is not None:
            strict4 = _connected(
                safe, pair_domain, first_anchor, second_anchor, connectivity=4
            )
            strict8 = _connected(
                safe, pair_domain, first_anchor, second_anchor, connectivity=8
            )
            frozen = bool(source.get("conservative_connected"))
            strict4_matches = bool(strict4 == frozen)
        else:
            strict4 = None
            strict8 = None
            strict4_matches = None
            frozen = bool(source.get("conservative_connected"))

        planner_success = bool(item.get("planner_success"))
        infrastructure_error = bool(item.get("infrastructure_error", False))
        path_xy = item.get("path_xy") or []
        path_cells = _rasterize_world_path(path_xy, metadata) if planner_success else set()
        in_map = {
            (x, y)
            for x, y in path_cells
            if 0 <= x < shape[1] and 0 <= y < shape[0]
        }
        out_of_map = path_cells - in_map

        pair_inside = {(x, y) for x, y in in_map if pair_domain[y, x]}
        finite_inside = {(x, y) for x, y in in_map if finite_headland_domain[y, x]}
        outside_pair = (in_map - pair_inside) | out_of_map
        outside_finite = (in_map - finite_inside) | out_of_map

        path_count = len(path_cells)
        pair_contained = bool(planner_success and path_count > 0 and not outside_pair)
        finite_contained = bool(planner_success and path_count > 0 and not outside_finite)
        if not planner_success:
            scope_class = "no_path"
        elif pair_contained:
            scope_class = "pair_domain"
        elif finite_contained:
            scope_class = "finite_headland_outside_pair"
        else:
            scope_class = "global_outside_finite_headland"

        unknown_count = sum(base[y, x] == UNKNOWN_VALUE for x, y in in_map)
        occupied_count = sum(base[y, x] == OCCUPIED_VALUE for x, y in in_map)
        if in_map:
            min_clearance = float(min(clearance[y, x] for x, y in in_map))
        else:
            min_clearance = None

        length_m = _path_length(path_xy) if planner_success else 0.0
        direct_m = _direct_distance(item)
        detour_ratio = (
            float(length_m / direct_m)
            if planner_success and direct_m > 1e-12
            else (None if not planner_success else float("inf"))
        )
        classification = _classify(
            frozen_positive=frozen,
            planner_success=planner_success,
            infrastructure_error=infrastructure_error,
            pair_contained=pair_contained,
            strict8=bool(strict8),
            strict4_matches=strict4_matches,
        )

        records.append(
            {
                "request_id": item.get("request_id"),
                "pair_id": key[0],
                "side": key[1],
                "direction": item.get("direction"),
                "radius_m": radius_m,
                "evaluation_status": source.get("evaluation_status"),
                "frozen_conservative_connected": frozen,
                "strict_connected_4": strict4,
                "strict_connected_8": strict8,
                "strict4_matches_frozen": strict4_matches,
                "planner_success": planner_success,
                "infrastructure_error": infrastructure_error,
                "expected_success": item.get("expected_success"),
                "negative_reason": item.get("negative_reason"),
                "scope_class": scope_class,
                "pair_domain_contained": pair_contained,
                "finite_headland_contained": finite_contained,
                "path_cell_count": int(path_count),
                "path_in_map_cell_count": int(len(in_map)),
                "path_out_of_map_cell_count": int(len(out_of_map)),
                "path_outside_pair_domain_cell_count": int(len(outside_pair)),
                "path_outside_pair_domain_fraction": (
                    float(len(outside_pair) / path_count) if path_count else 0.0
                ),
                "path_outside_finite_headland_cell_count": int(len(outside_finite)),
                "path_outside_finite_headland_fraction": (
                    float(len(outside_finite) / path_count) if path_count else 0.0
                ),
                "touches_unknown": bool(unknown_count > 0),
                "touches_unknown_cell_count": int(unknown_count),
                "touches_occupied": bool(occupied_count > 0),
                "touches_occupied_cell_count": int(occupied_count),
                "min_source_map_clearance_m": min_clearance,
                "path_length_m": float(length_m),
                "direct_distance_m": float(direct_m),
                "detour_ratio": detour_ratio,
                "classification": classification,
                "failure_reason": item.get("failure_reason"),
                "path_xy": path_xy,
            }
        )

    class_counts = {}
    scope_counts = {}
    for item in records:
        class_counts[item["classification"]] = class_counts.get(item["classification"], 0) + 1
        scope_counts[item["scope_class"]] = scope_counts.get(item["scope_class"], 0) + 1
    return {
        "schema_version": 1,
        "method": "p1_f3_nav2_planner_path_scope_audit",
        "radius_m": radius_m,
        "grid_shape_yx": list(shape),
        "resolution_m": resolution,
        "policy": {
            "frozen_topology_connectivity": 4,
            "planner_aligned_diagnostic_connectivity": 8,
            "map_editing": False,
            "semantic_promotion": False,
            "successful_path_scope": "rasterized Nav2 path cells against frozen pair/finite-headland domains",
            "source_map_clearance": "distance to nearest non-FREE cell on conservative source map",
        },
        "summary": {
            "record_count": len(records),
            "planner_success": int(sum(item["planner_success"] for item in records)),
            "planner_failure": int(sum(not item["planner_success"] for item in records)),
            "infrastructure_error": int(sum(item["infrastructure_error"] for item in records)),
            "strict4_contract_mismatch": int(sum(item["strict4_matches_frozen"] is False for item in records)),
            "classification_counts": class_counts,
            "scope_counts": scope_counts,
        },
        "records": records,
    }
