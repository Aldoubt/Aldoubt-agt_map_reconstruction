"""Recover enclosed greenhouse non-HARD interior by border flood fill.

HARD occupied cells are impermeable barriers.  Starting from every non-HARD
border cell, a 4-connected flood fill marks the exterior-reachable domain.  Any
remaining non-HARD cells are enclosed interior candidates.  No morphology,
wall-gap closure, semantic promotion, or navigation-map modification occurs.
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


def build_site_interior_flood_fill(base_map):
    """Return enclosed non-HARD site interior and exterior-reachable masks."""
    base = _validate_base_map(base_map)
    hard = base == OCCUPIED_VALUE
    nonhard = ~hard
    exterior = np.zeros(base.shape, dtype=bool)
    queue = deque(_border_nonhard_seeds(nonhard))
    for y, x in queue:
        exterior[y, x] = True

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
                queue.append((ny, nx))

    interior = nonhard & ~exterior
    component_sizes = _connected_component_sizes(interior)
    interior_count = int(np.count_nonzero(interior))
    nonhard_count = int(np.count_nonzero(nonhard))
    result = {
        "schema_version": 1,
        "method": "hard_boundary_border_flood_fill_site_interior",
        "status": "ok" if interior_count > 0 else "leaked_or_unenclosed",
        "grid_shape_yx": list(base.shape),
        "connectivity": 4,
        "hard_cell_count": int(np.count_nonzero(hard)),
        "nonhard_cell_count": nonhard_count,
        "border_seed_cell_count": int(np.count_nonzero(exterior & (
            (np.indices(base.shape)[0] == 0)
            | (np.indices(base.shape)[0] == height - 1)
            | (np.indices(base.shape)[1] == 0)
            | (np.indices(base.shape)[1] == width - 1)
        ))),
        "exterior_reachable_nonhard_cell_count": int(np.count_nonzero(exterior)),
        "interior_nonhard_cell_count": interior_count,
        "interior_fraction_of_nonhard": (
            0.0 if nonhard_count == 0 else float(interior_count / nonhard_count)
        ),
        "interior_component_count": len(component_sizes),
        "interior_component_sizes": component_sizes,
        "morphology_applied": False,
        "automatic_wall_gap_closure": False,
        "automatic_component_selection": False,
        "site_interior_mask_is_semantic_free": False,
        "navigation_map_modified": False,
        "semantic_promotion": False,
    }
    masks = {
        "site_interior_nonhard": interior,
        "exterior_reachable_nonhard": exterior,
        "hard_barrier": hard,
    }
    return result, masks
