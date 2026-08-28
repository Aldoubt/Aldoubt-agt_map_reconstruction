"""Diagnose why scoped headland handoff pairs remain disconnected."""

from __future__ import annotations

import heapq

import cv2
import numpy as np
from scipy import ndimage

from .headland_handoff_connectivity import (
    FREE_VALUE,
    OCCUPIED_VALUE,
    UNKNOWN_VALUE,
    _approach_mask,
    _canonical_map,
    _depth_side_mask,
    _normalize_direction,
    _pair_cross_window,
    _safe_mask,
    _sorted_aisles,
)


def _aisle_lookup(aisles):
    return {str(item["label"]): item for item in _sorted_aisles(aisles)}


def _relaxed_safe_mask(base_map, resolution, radius_m):
    hard = base_map == OCCUPIED_VALUE
    nonhard = ~hard
    distance_to_hard = ndimage.distance_transform_edt(nonhard) * float(resolution)
    return nonhard & (distance_to_hard + 1e-12 >= float(radius_m))


def _point_component(labels, xy):
    x, y = np.rint(np.asarray(xy, dtype=float)).astype(int)
    if y < 0 or x < 0 or y >= labels.shape[0] or x >= labels.shape[1]:
        return 0
    return int(labels[y, x])


def _connected(mask, domain, first_xy, second_xy):
    labels, _ = ndimage.label(mask & domain)
    first_id = _point_component(labels, first_xy)
    second_id = _point_component(labels, second_xy)
    return bool(first_id > 0 and first_id == second_id)


def _shortest_bridge_path(allowed, strict_safe, unknown, first_xy, second_xy, resolution):
    allowed = np.asarray(allowed, dtype=bool)
    strict = np.asarray(strict_safe, dtype=bool)
    unknown = np.asarray(unknown, dtype=bool)
    height, width = allowed.shape
    sx, sy = np.rint(np.asarray(first_xy, dtype=float)).astype(int)
    gx, gy = np.rint(np.asarray(second_xy, dtype=float)).astype(int)
    if not (0 <= sx < width and 0 <= sy < height and 0 <= gx < width and 0 <= gy < height):
        return None
    if not allowed[sy, sx] or not allowed[gy, gx]:
        return None

    bridge_dist = np.full((height, width), np.inf, dtype=float)
    total_dist = np.full((height, width), np.inf, dtype=float)
    previous = np.full((height, width, 2), -1, dtype=np.int32)
    bridge_dist[sy, sx] = 0.0
    total_dist[sy, sx] = 0.0
    heap = [(0.0, 0.0, sy, sx)]

    while heap:
        bridge_cost, total_cost, y, x = heapq.heappop(heap)
        if bridge_cost > bridge_dist[y, x] + 1e-12:
            continue
        if abs(bridge_cost - bridge_dist[y, x]) <= 1e-12 and total_cost > total_dist[y, x] + 1e-12:
            continue
        if x == gx and y == gy:
            break
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if ny < 0 or nx < 0 or ny >= height or nx >= width or not allowed[ny, nx]:
                continue
            step = float(resolution)
            next_bridge = bridge_cost + (0.0 if strict[ny, nx] else step)
            next_total = total_cost + step
            better = next_bridge < bridge_dist[ny, nx] - 1e-12 or (
                abs(next_bridge - bridge_dist[ny, nx]) <= 1e-12
                and next_total < total_dist[ny, nx] - 1e-12
            )
            if better:
                bridge_dist[ny, nx] = next_bridge
                total_dist[ny, nx] = next_total
                previous[ny, nx] = (y, x)
                heapq.heappush(heap, (next_bridge, next_total, ny, nx))

    if not np.isfinite(bridge_dist[gy, gx]):
        return None

    path = []
    y, x = gy, gx
    while True:
        path.append((x, y))
        if x == sx and y == sy:
            break
        py, px = previous[y, x]
        if py < 0:
            return None
        y, x = int(py), int(px)
    path.reverse()

    unknown_m = 0.0
    non_strict_m = 0.0
    unknown_cells = 0
    for index, (x, y) in enumerate(path):
        if unknown[y, x]:
            unknown_cells += 1
        if index == 0:
            continue
        if not strict[y, x]:
            non_strict_m += float(resolution)
        if unknown[y, x]:
            unknown_m += float(resolution)
    return {
        "path_grid_xy": [[int(x), int(y)] for x, y in path],
        "shortest_non_strict_bridge_m": float(non_strict_m),
        "shortest_unknown_bridge_m": float(unknown_m),
        "shortest_unknown_bridge_cell_count": int(unknown_cells),
        "relaxed_path_length_m": float(total_dist[gy, gx]),
    }


def _classify_bridge_type(bridge):
    if bridge is None:
        return "not_available"
    unknown_m = float(bridge.get("shortest_unknown_bridge_m") or 0.0)
    non_strict_m = float(bridge.get("shortest_non_strict_bridge_m") or 0.0)
    eps = 1e-9
    if unknown_m <= eps and non_strict_m > eps:
        return "clearance_only_bridge"
    if unknown_m > eps and non_strict_m <= unknown_m + eps:
        return "unknown_bridge"
    if unknown_m > eps and non_strict_m > unknown_m + eps:
        return "mixed_bridge"
    return "zero_cost_bridge"


def _domain_parts(shape, record, aisles, side_masks, cross, resolution, radius_m):
    first = aisles.get(str(record.get("first_aisle")))
    second = aisles.get(str(record.get("second_aisle")))
    if first is None or second is None:
        raise ValueError(f"missing aisle for {record.get('pair_id')}")
    side = str(record.get("side"))
    if side not in {"entry", "exit"}:
        raise ValueError("side must be entry or exit")
    first_anchor = record.get("first_anchor_grid_xy")
    second_anchor = record.get("second_anchor_grid_xy")
    if first_anchor is None or second_anchor is None:
        empty = np.zeros(shape, dtype=bool)
        return empty, empty

    pair_window = _pair_cross_window(shape, first, second, cross, resolution, radius_m)
    approach = (
        _approach_mask(shape, first, first_anchor, side, resolution)
        | _approach_mask(shape, second, second_anchor, side, resolution)
    )
    finite_side = side_masks[side]
    pair_domain = (finite_side & pair_window) | approach
    finite_headland_domain = finite_side | approach
    return pair_domain, finite_headland_domain


def _domain_for_record(shape, record, aisles, side_masks, cross, resolution, radius_m):
    pair_domain, _ = _domain_parts(
        shape, record, aisles, side_masks, cross, resolution, radius_m
    )
    return pair_domain


def analyze_headland_gap_diagnostics(
    baseline_map,
    conservative_map,
    aisles,
    connectivity_result,
    depth_profile_payload,
    depth_masks,
    *,
    resolution,
    radius_m=0.20,
):
    """Classify strict failures and quantify bridge, clearance, and scope causes."""
    resolution = float(resolution)
    radius_m = float(radius_m)
    if not np.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("resolution must be finite and > 0")
    if not np.isfinite(radius_m) or radius_m < 0.0:
        raise ValueError("radius_m must be finite and >= 0")

    baseline = _canonical_map("baseline_map", baseline_map)
    conservative = _canonical_map("conservative_map", conservative_map, baseline.shape)
    shape = baseline.shape
    profile = dict(depth_profile_payload)
    expected_shape = tuple(int(v) for v in profile.get("grid_shape_yx", shape))
    if expected_shape != shape:
        raise ValueError("depth profile grid shape does not match maps")
    cross = _normalize_direction("cross_row_direction", profile.get("cross_row_direction"))
    side_masks = {
        side: _depth_side_mask(profile, depth_masks, side, shape)
        for side in ("entry", "exit")
    }
    aisle_by_label = _aisle_lookup(aisles)

    baseline_safe = _safe_mask(baseline, resolution, radius_m)
    conservative_safe = _safe_mask(conservative, resolution, radius_m)
    conservative_free = conservative == FREE_VALUE
    baseline_free = baseline == FREE_VALUE
    conservative_clearance = ndimage.distance_transform_edt(conservative_free) * resolution
    relaxed_safe = _relaxed_safe_mask(conservative, resolution, radius_m)
    unknown = conservative == UNKNOWN_VALUE

    records = []
    for source in connectivity_result.get("pairs", []):
        item = dict(source)
        domain, finite_headland_domain = _domain_parts(
            shape, item, aisle_by_label, side_masks, cross, resolution, radius_m
        )

        promoted_free = (~baseline_free) & conservative_free & domain
        promoted_free_strict_safe = promoted_free & conservative_safe
        baseline_free_newly_safe = baseline_free & (~baseline_safe) & conservative_safe & domain
        newly_safe = (~baseline_safe) & conservative_safe & domain

        promoted_count = int(np.count_nonzero(promoted_free))
        promoted_safe_count = int(np.count_nonzero(promoted_free_strict_safe))
        baseline_new_safe_count = int(np.count_nonzero(baseline_free_newly_safe))
        newly_safe_count = int(np.count_nonzero(newly_safe))
        promoted_survival = (
            float(promoted_safe_count / promoted_count) if promoted_count else 0.0
        )
        max_promoted_clearance = (
            float(np.max(conservative_clearance[promoted_free])) if promoted_count else 0.0
        )

        evaluated = str(item.get("evaluation_status")) == "evaluated"
        first_anchor = item.get("first_anchor_grid_xy")
        second_anchor = item.get("second_anchor_grid_xy")
        strict_connected = bool(item.get("conservative_connected"))
        relaxed_connected = False
        finite_headland_relaxed_connected = False
        bridge = None

        if evaluated and first_anchor is not None and second_anchor is not None:
            relaxed_connected = _connected(
                relaxed_safe, domain, first_anchor, second_anchor
            )
            finite_headland_relaxed_connected = _connected(
                relaxed_safe,
                finite_headland_domain,
                first_anchor,
                second_anchor,
            )
            if relaxed_connected and not strict_connected:
                bridge = _shortest_bridge_path(
                    relaxed_safe & domain,
                    conservative_safe & domain,
                    unknown,
                    first_anchor,
                    second_anchor,
                    resolution,
                )

        bridge_type = _classify_bridge_type(bridge)
        pair_window_scope_blocked = False
        hard_or_finite_headland_blocked = False

        if not evaluated:
            failure_class = str(item.get("evaluation_status"))
            bridge_type = failure_class
        elif strict_connected:
            failure_class = "connected"
            bridge_type = "strict_connected"
        elif not relaxed_connected:
            if finite_headland_relaxed_connected:
                failure_class = "pair_window_scope_blocked"
                pair_window_scope_blocked = True
            else:
                failure_class = "hard_or_finite_headland_blocked"
                hard_or_finite_headland_blocked = True
            bridge_type = "not_available"
        elif promoted_count == 0:
            failure_class = "no_overlay_in_domain"
        elif promoted_safe_count == 0 and baseline_new_safe_count > 0:
            failure_class = "overlay_indirect_clearance_gain_only"
        elif promoted_safe_count == 0:
            failure_class = "overlay_eroded_by_clearance"
        else:
            failure_class = "safe_overlay_not_bridging"

        record = {
            "pair_id": item.get("pair_id"),
            "first_aisle": item.get("first_aisle"),
            "second_aisle": item.get("second_aisle"),
            "side": item.get("side"),
            "evaluation_status": item.get("evaluation_status"),
            "radius_m": radius_m,
            "strict_connected": strict_connected,
            "relaxed_connected": bool(relaxed_connected),
            "finite_headland_relaxed_connected": bool(
                finite_headland_relaxed_connected
            ),
            "bridge_type": bridge_type,
            "bridge_class": bridge_type,
            "failure_class": failure_class,
            "pair_window_scope_blocked": bool(pair_window_scope_blocked),
            "hard_or_finite_headland_blocked": bool(
                hard_or_finite_headland_blocked
            ),
            "domain_cell_count": int(np.count_nonzero(domain)),
            "finite_headland_domain_cell_count": int(
                np.count_nonzero(finite_headland_domain)
            ),
            "promoted_free_cell_count_in_domain": promoted_count,
            "promoted_free_strict_safe_cell_count_in_domain": promoted_safe_count,
            "baseline_free_newly_safe_cell_count_in_domain": baseline_new_safe_count,
            "newly_safe_cell_count_in_domain": newly_safe_count,
            "promoted_free_survival_ratio": promoted_survival,
            "max_promoted_free_clearance_m": max_promoted_clearance,
            "new_free_cell_count_in_domain": promoted_count,
            "new_safe_cell_count_in_domain": newly_safe_count,
            "max_new_free_clearance_m": max_promoted_clearance,
            "first_anchor_grid_xy": first_anchor,
            "second_anchor_grid_xy": second_anchor,
            "shortest_unknown_bridge_m": None,
            "shortest_unknown_bridge_cell_count": None,
            "shortest_non_strict_bridge_m": None,
            "relaxed_path_length_m": None,
            "bridge_path_grid_xy": None,
        }
        if bridge is not None:
            record.update(
                {
                    "shortest_unknown_bridge_m": bridge[
                        "shortest_unknown_bridge_m"
                    ],
                    "shortest_unknown_bridge_cell_count": bridge[
                        "shortest_unknown_bridge_cell_count"
                    ],
                    "shortest_non_strict_bridge_m": bridge[
                        "shortest_non_strict_bridge_m"
                    ],
                    "relaxed_path_length_m": bridge["relaxed_path_length_m"],
                    "bridge_path_grid_xy": bridge["path_grid_xy"],
                }
            )
        records.append(record)

    failure_counts = {}
    bridge_type_counts = {}
    for record in records:
        failure_counts[record["failure_class"]] = (
            failure_counts.get(record["failure_class"], 0) + 1
        )
        bridge_type_counts[record["bridge_type"]] = (
            bridge_type_counts.get(record["bridge_type"], 0) + 1
        )
    return {
        "schema_version": 2,
        "method": "scoped_headland_gap_diagnostics",
        "grid_shape_yx": list(shape),
        "resolution_m": resolution,
        "radius_m": radius_m,
        "record_count": len(records),
        "evaluated_record_count": int(
            sum(r["evaluation_status"] == "evaluated" for r in records)
        ),
        "failure_counts": failure_counts,
        "bridge_type_counts": bridge_type_counts,
        "policy": {
            "strict_safe": "free && distance_to_nonfree >= radius",
            "relaxed_diagnostic_safe": "not hard && distance_to_hard >= radius",
            "relaxed_unknown_is_navigation_acceptance": False,
            "promoted_free_definition": "baseline not-free -> conservative free inside pair-scoped domain",
            "promoted_free_strict_safe_definition": "promoted free && conservative strict-safe",
            "baseline_free_newly_safe_definition": "baseline free && not baseline strict-safe && conservative strict-safe",
            "pair_scope_diagnostic": "compare pair cross-window domain against the same finite side headland envelope with pair cross-window removed",
            "hard_or_finite_headland_blocked_does_not_prove_hard_obstacle": True,
            "bridge_cost": "minimum 4-connected geometric distance through relaxed-only cells; direct unknown distance reported separately",
            "map_editing": False,
            "semantic_promotion": False,
        },
        "records": records,
    }


def _overlay_image(conservative_map, diagnostics):
    base = np.asarray(conservative_map, dtype=np.uint8)
    image = cv2.cvtColor(np.flipud(base), cv2.COLOR_GRAY2BGR)
    height = int(base.shape[0])
    for record in diagnostics.get("records", []):
        path = record.get("bridge_path_grid_xy") or []
        if len(path) >= 2:
            points = np.asarray(
                [[int(x), height - 1 - int(y)] for x, y in path], dtype=np.int32
            )
            cv2.polylines(image, [points], False, (0, 0, 255), 1, lineType=cv2.LINE_AA)
        for key in ("first_anchor_grid_xy", "second_anchor_grid_xy"):
            xy = record.get(key)
            if xy is None:
                continue
            x, y = np.rint(np.asarray(xy, dtype=float)).astype(int)
            cv2.circle(image, (int(x), height - 1 - int(y)), 2, (0, 255, 0), -1)
    return image


def write_headland_gap_diagnostics_bundle(diagnostics, conservative_map, output_dir):
    import csv
    import json
    from pathlib import Path

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "headland_gap_diagnostics.json"
    json_path.write_text(
        json.dumps(diagnostics, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    fields = [
        "pair_id",
        "side",
        "evaluation_status",
        "strict_connected",
        "relaxed_connected",
        "finite_headland_relaxed_connected",
        "bridge_type",
        "failure_class",
        "pair_window_scope_blocked",
        "hard_or_finite_headland_blocked",
        "promoted_free_cell_count_in_domain",
        "promoted_free_strict_safe_cell_count_in_domain",
        "baseline_free_newly_safe_cell_count_in_domain",
        "newly_safe_cell_count_in_domain",
        "promoted_free_survival_ratio",
        "max_promoted_free_clearance_m",
        "shortest_unknown_bridge_m",
        "shortest_unknown_bridge_cell_count",
        "shortest_non_strict_bridge_m",
        "relaxed_path_length_m",
    ]
    csv_path = output / "headland_gap_diagnostics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in diagnostics.get("records", []):
            writer.writerow({key: record.get(key) for key in fields})
    overlay_path = output / "headland_gap_diagnostics.png"
    cv2.imwrite(str(overlay_path), _overlay_image(conservative_map, diagnostics))
    return {"json": json_path, "csv": csv_path, "overlay": overlay_path}
