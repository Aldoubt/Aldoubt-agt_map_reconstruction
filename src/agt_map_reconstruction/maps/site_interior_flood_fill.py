"""Recover enclosed greenhouse non-HARD interior by border flood fill.

HARD occupied cells are impermeable barriers. Starting from every non-HARD
border cell, a 4-connected flood fill marks the exterior-reachable domain. Any
remaining non-HARD cells are enclosed interior candidates. Optional trusted
interior anchors may validate that the border flood has not leaked into the
known greenhouse structural core. No morphology, wall-gap closure, semantic
promotion, or navigation-map modification occurs.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .navigation_export import FREE_VALUE, OCCUPIED_VALUE, UNKNOWN_VALUE


def _validate_base_map(base_map):
    base = np.asarray(base_map, dtype=np.uint8)
    if base.ndim != 2 or base.shape[0] < 1 or base.shape[1] < 1:
        raise ValueError("base_map must be a non-empty 2D array")
    if not np.isin(base, [OCCUPIED_VALUE, UNKNOWN_VALUE, FREE_VALUE]).all():
        raise ValueError("base_map contains unsupported gray values")
    return base


def _border_nonhard_seeds(nonhard):
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


def _connected_component_sizes(mask):
    target = np.asarray(mask, dtype=bool)
    visited = np.zeros(target.shape, dtype=bool)
    sizes = []
    height, width = target.shape
    for y in range(height):
        for x in range(width):
            if not target[y, x] or visited[y, x]:
                continue
            queue = deque([(y, x)])
            visited[y, x] = True
            size = 0
            while queue:
                cy, cx = queue.popleft()
                size += 1
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = cy + dy, cx + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and target[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        queue.append((ny, nx))
            sizes.append(size)
    return sorted(sizes, reverse=True)


def _prepare_anchor_mask(anchor_mask, shape):
    if anchor_mask is None:
        return None
    anchors = np.asarray(anchor_mask, dtype=bool)
    if anchors.shape != shape:
        raise ValueError("interior_anchor_mask must match base_map shape")
    return anchors


def _follow_parent_path(parent_y, parent_x, start):
    """Follow the border-flood parent chain from one leaked anchor to a border seed."""
    y, x = int(start[0]), int(start[1])
    path = []
    seen = set()
    while True:
        key = (y, x)
        if key in seen:
            raise RuntimeError("flood-fill parent chain contains a cycle")
        seen.add(key)
        path.append((y, x))
        py = int(parent_y[y, x])
        px = int(parent_x[y, x])
        if py < 0 or px < 0:
            break
        y, x = py, px
    return path


def build_site_interior_flood_fill(base_map, *, interior_anchor_mask=None):
    """Return enclosed non-HARD site interior and exterior-reachable masks.

    When ``interior_anchor_mask`` is provided, those cells are trusted to lie
    inside the greenhouse structural core. Any anchor reached by the border
    flood proves that the HARD boundary is open for this dataset, even if small
    unrelated enclosed pockets remain elsewhere in the map.
    """
    base = _validate_base_map(base_map)
    hard = base == OCCUPIED_VALUE
    nonhard = ~hard
    anchors = _prepare_anchor_mask(interior_anchor_mask, base.shape)

    exterior = np.zeros(base.shape, dtype=bool)
    parent_y = np.full(base.shape, -2, dtype=np.int32)
    parent_x = np.full(base.shape, -2, dtype=np.int32)
    queue = deque(_border_nonhard_seeds(nonhard))
    for y, x in queue:
        exterior[y, x] = True
        parent_y[y, x] = -1
        parent_x[y, x] = -1

    height, width = base.shape
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

    interior = nonhard & ~exterior
    component_sizes = _connected_component_sizes(interior)
    interior_count = int(np.count_nonzero(interior))
    nonhard_count = int(np.count_nonzero(nonhard))

    leak_path = np.zeros(base.shape, dtype=bool)
    leak_path_xy = []
    anchor_requested = anchors is not None
    anchor_count = 0
    anchor_hard_count = 0
    anchor_nonhard_count = 0
    anchor_exterior_count = 0
    anchor_enclosed_count = 0
    anchor_validation_passed = None
    leak_anchor_xy = None

    if anchors is not None:
        anchor_count = int(np.count_nonzero(anchors))
        anchor_hard_count = int(np.count_nonzero(anchors & hard))
        anchor_nonhard = anchors & nonhard
        anchor_nonhard_count = int(np.count_nonzero(anchor_nonhard))
        leaked_anchors = anchor_nonhard & exterior
        enclosed_anchors = anchor_nonhard & interior
        anchor_exterior_count = int(np.count_nonzero(leaked_anchors))
        anchor_enclosed_count = int(np.count_nonzero(enclosed_anchors))
        anchor_validation_passed = bool(
            anchor_count > 0
            and anchor_hard_count == 0
            and anchor_nonhard_count == anchor_count
            and anchor_exterior_count == 0
            and anchor_enclosed_count == anchor_nonhard_count
        )
        if anchor_exterior_count > 0:
            ay, ax = np.argwhere(leaked_anchors)[0]
            path = _follow_parent_path(parent_y, parent_x, (int(ay), int(ax)))
            for py, px in path:
                leak_path[py, px] = True
            leak_path_xy = [[int(px), int(py)] for py, px in path]
            leak_anchor_xy = [int(ax), int(ay)]

    topology_has_enclosed_nonhard = interior_count > 0
    if anchor_requested:
        status = (
            "ok"
            if topology_has_enclosed_nonhard and bool(anchor_validation_passed)
            else "leaked_or_unenclosed"
        )
        status_basis = "trusted_interior_anchor_validation"
    else:
        status = "ok" if topology_has_enclosed_nonhard else "leaked_or_unenclosed"
        status_basis = "topology_only_no_interior_anchor"

    yy, xx = np.indices(base.shape)
    border = (yy == 0) | (yy == height - 1) | (xx == 0) | (xx == width - 1)
    result = {
        "schema_version": 2,
        "method": "hard_boundary_border_flood_fill_site_interior",
        "status": status,
        "status_basis": status_basis,
        "grid_shape_yx": list(base.shape),
        "connectivity": 4,
        "hard_cell_count": int(np.count_nonzero(hard)),
        "nonhard_cell_count": nonhard_count,
        "border_seed_cell_count": int(np.count_nonzero(exterior & border)),
        "exterior_reachable_nonhard_cell_count": int(np.count_nonzero(exterior)),
        "interior_nonhard_cell_count": interior_count,
        "interior_fraction_of_nonhard": (
            0.0 if nonhard_count == 0 else float(interior_count / nonhard_count)
        ),
        "interior_component_count": len(component_sizes),
        "interior_component_sizes": component_sizes,
        "interior_anchor_validation_requested": anchor_requested,
        "interior_anchor_cell_count": anchor_count,
        "interior_anchor_hard_cell_count": anchor_hard_count,
        "interior_anchor_nonhard_cell_count": anchor_nonhard_count,
        "interior_anchor_exterior_reachable_cell_count": anchor_exterior_count,
        "interior_anchor_enclosed_cell_count": anchor_enclosed_count,
        "interior_anchor_validation_passed": anchor_validation_passed,
        "leak_anchor_xy": leak_anchor_xy,
        "leak_path_xy": leak_path_xy,
        "morphology_applied": False,
        "automatic_wall_gap_closure": False,
        "automatic_component_selection": False,
        "interior_anchor_used_to_construct_mask": False,
        "site_interior_mask_is_semantic_free": False,
        "navigation_map_modified": False,
        "semantic_promotion": False,
    }
    masks = {
        "site_interior_nonhard": interior,
        "exterior_reachable_nonhard": exterior,
        "hard_barrier": hard,
        "interior_anchor": (
            np.zeros(base.shape, dtype=bool) if anchors is None else anchors
        ),
        "leak_path": leak_path,
    }
    return result, masks
