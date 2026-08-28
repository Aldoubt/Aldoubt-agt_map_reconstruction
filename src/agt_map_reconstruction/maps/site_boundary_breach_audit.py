"""Diagnose HARD-boundary leaks using trusted row-lattice interior anchors.

The audit is intentionally diagnostic. It never closes wall gaps, modifies the
navigation map, or promotes any cell to semantic free space. A single border
flood fill is used to classify trusted anchors. For every leaked anchor, the
stored BFS parent chain is traced back to one map-border seed. Overlapping leak
paths are accumulated so common bottlenecks can be localized without inventing
a physical boundary.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .navigation_export import FREE_VALUE, OCCUPIED_VALUE, UNKNOWN_VALUE


def _validate_base_map(base_map):
    base = np.asarray(base_map, dtype=np.uint8)
    if base.ndim != 2 or min(base.shape) < 1:
        raise ValueError("base_map must be a non-empty 2D array")
    if not np.isin(base, [OCCUPIED_VALUE, UNKNOWN_VALUE, FREE_VALUE]).all():
        raise ValueError("base_map contains unsupported gray values")
    return base


def _border_seeds(nonhard):
    height, width = nonhard.shape
    seeds = []
    seen = set()
    for x in range(width):
        for y in (0, height - 1):
            if bool(nonhard[y, x]) and (y, x) not in seen:
                seeds.append((y, x))
                seen.add((y, x))
    for y in range(height):
        for x in (0, width - 1):
            if bool(nonhard[y, x]) and (y, x) not in seen:
                seeds.append((y, x))
                seen.add((y, x))
    return seeds


def _flood_with_parents(nonhard):
    exterior = np.zeros(nonhard.shape, dtype=bool)
    parent_y = np.full(nonhard.shape, -2, dtype=np.int32)
    parent_x = np.full(nonhard.shape, -2, dtype=np.int32)
    queue = deque(_border_seeds(nonhard))
    for y, x in queue:
        exterior[y, x] = True
        parent_y[y, x] = -1
        parent_x[y, x] = -1

    height, width = nonhard.shape
    while queue:
        y, x = queue.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if (
                0 <= ny < height
                and 0 <= nx < width
                and nonhard[ny, nx]
                and not exterior[ny, nx]
            ):
                exterior[ny, nx] = True
                parent_y[ny, nx] = y
                parent_x[ny, nx] = x
                queue.append((ny, nx))
    return exterior, parent_y, parent_x


def _trace(parent_y, parent_x, start):
    y, x = int(start[0]), int(start[1])
    path = []
    seen = set()
    while True:
        key = (y, x)
        if key in seen:
            raise RuntimeError("flood-fill parent chain contains a cycle")
        seen.add(key)
        path.append(key)
        py = int(parent_y[y, x])
        px = int(parent_x[y, x])
        if py < 0 or px < 0:
            break
        y, x = py, px
    return path


def _border_side(y, x, shape):
    height, width = shape
    sides = []
    if y == 0:
        sides.append("bottom")
    if y == height - 1:
        sides.append("top")
    if x == 0:
        sides.append("left")
    if x == width - 1:
        sides.append("right")
    return "+".join(sides) if sides else "not_border"


def _normalize_anchor_records(anchor_records, shape):
    height, width = shape
    normalized = []
    seen = set()
    for index, item in enumerate(anchor_records):
        record = dict(item)
        xy = record.get("grid_xy")
        if xy is None or len(xy) != 2:
            raise ValueError("each anchor must contain grid_xy=[x,y]")
        x = int(xy[0])
        y = int(xy[1])
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"anchor {record.get('slot_id', index)} lies outside map")
        key = (x, y)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(
            {
                "slot_id": str(record.get("slot_id", f"anchor_{index}")),
                "source": str(record.get("source", "")),
                "grid_xy": [x, y],
            }
        )
    if not normalized:
        raise ValueError("at least one anchor is required")
    return normalized


def audit_site_boundary_breaches(base_map, anchor_records, *, resolution_m):
    """Classify anchors and localize common border-reachable leak paths."""
    base = _validate_base_map(base_map)
    resolution = float(resolution_m)
    if resolution <= 0.0:
        raise ValueError("resolution_m must be > 0")
    anchors = _normalize_anchor_records(anchor_records, base.shape)

    hard = base == OCCUPIED_VALUE
    nonhard = ~hard
    exterior, parent_y, parent_x = _flood_with_parents(nonhard)

    path_support = np.zeros(base.shape, dtype=np.uint16)
    leaked_mask = np.zeros(base.shape, dtype=bool)
    anchor_mask = np.zeros(base.shape, dtype=bool)
    hard_anchor_mask = np.zeros(base.shape, dtype=bool)
    records = []
    exit_side_counts = {}

    leaked_count = 0
    hard_count = 0
    enclosed_count = 0
    for anchor in anchors:
        x, y = anchor["grid_xy"]
        anchor_mask[y, x] = True
        item = dict(anchor)
        item["map_value"] = int(base[y, x])
        if hard[y, x]:
            hard_count += 1
            hard_anchor_mask[y, x] = True
            item.update(
                {
                    "classification": "anchor_on_hard",
                    "path_cell_count": 0,
                    "path_length_m": 0.0,
                    "border_exit_xy": None,
                    "border_exit_side": None,
                }
            )
        elif exterior[y, x]:
            leaked_count += 1
            path = _trace(parent_y, parent_x, (y, x))
            for py, px in path:
                if path_support[py, px] < np.iinfo(np.uint16).max:
                    path_support[py, px] += 1
                leaked_mask[py, px] = True
            ey, ex = path[-1]
            side = _border_side(ey, ex, base.shape)
            exit_side_counts[side] = int(exit_side_counts.get(side, 0) + 1)
            item.update(
                {
                    "classification": "exterior_reachable",
                    "path_cell_count": len(path),
                    "path_length_m": float(max(0, len(path) - 1) * resolution),
                    "border_exit_xy": [int(ex), int(ey)],
                    "border_exit_side": side,
                }
            )
        else:
            enclosed_count += 1
            item.update(
                {
                    "classification": "enclosed_nonhard",
                    "path_cell_count": 0,
                    "path_length_m": 0.0,
                    "border_exit_xy": None,
                    "border_exit_side": None,
                }
            )
        records.append(item)

    max_support = int(path_support.max()) if leaked_count else 0
    bottleneck = (path_support == max_support) & (path_support > 0)
    max_cells_yx = np.argwhere(bottleneck)
    max_cells_xy = [[int(x), int(y)] for y, x in max_cells_yx]
    unique_exits = {
        tuple(record["border_exit_xy"])
        for record in records
        if record["border_exit_xy"] is not None
    }

    leaked_lengths = [
        float(record["path_length_m"])
        for record in records
        if record["classification"] == "exterior_reachable"
    ]
    result = {
        "schema_version": 1,
        "method": "trusted_anchor_outer_boundary_breach_audit",
        "status": "breach_confirmed" if leaked_count > 0 else "no_anchor_reachable_breach",
        "grid_shape_yx": list(base.shape),
        "resolution_m": resolution,
        "anchor_count": len(anchors),
        "leaked_anchor_count": leaked_count,
        "hard_anchor_count": hard_count,
        "enclosed_anchor_count": enclosed_count,
        "anchor_quality_warning": hard_count > 0,
        "unique_border_exit_count": len(unique_exits),
        "border_exit_side_counts": exit_side_counts,
        "path_length_m": {
            "min": None if not leaked_lengths else float(np.min(leaked_lengths)),
            "median": None if not leaked_lengths else float(np.median(leaked_lengths)),
            "max": None if not leaked_lengths else float(np.max(leaked_lengths)),
        },
        "max_path_support_count": max_support,
        "max_path_support_fraction_of_leaked_anchors": (
            0.0 if leaked_count == 0 else float(max_support / leaked_count)
        ),
        "max_path_support_cells_xy": max_cells_xy,
        "anchors": records,
        "policy": {
            "automatic_wall_gap_closure": False,
            "automatic_boundary_repair": False,
            "automatic_doorway_classification": False,
            "anchor_on_hard_auto_shifted": False,
            "navigation_map_modified": False,
            "semantic_promotion": False,
        },
    }
    masks = {
        "anchor": anchor_mask,
        "hard_anchor": hard_anchor_mask,
        "leaked_paths": leaked_mask,
        "path_support_count": path_support,
        "max_path_support": bottleneck,
        "exterior_reachable_nonhard": exterior,
        "hard_barrier": hard,
    }
    return result, masks
