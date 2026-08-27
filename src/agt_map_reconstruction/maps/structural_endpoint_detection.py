"""Detect persistent structural row terminations from bilateral support profiles."""

from __future__ import annotations

import numpy as np


_VALID_STATUS = {
    "ok_bilateral",
    "ambiguous_single_side",
    "insufficient_structural_support",
}


def _close_short_internal_gaps(mask, max_gap_bins):
    values = np.asarray(mask, dtype=bool).copy()
    if max_gap_bins <= 0 or values.size == 0:
        return values
    index = 0
    n = values.size
    while index < n:
        if values[index]:
            index += 1
            continue
        start = index
        while index < n and not values[index]:
            index += 1
        end = index
        gap_len = end - start
        bounded = start > 0 and end < n and values[start - 1] and values[end]
        if bounded and gap_len <= int(max_gap_bins):
            values[start:end] = True
    return values


def _longest_persistent_run(mask, min_bins):
    values = np.asarray(mask, dtype=bool)
    best = None
    index = 0
    n = values.size
    while index < n:
        if not values[index]:
            index += 1
            continue
        start = index
        while index < n and values[index]:
            index += 1
        end = index
        length = end - start
        if length < int(min_bins):
            continue
        key = (length, -start)
        if best is None or key > best[0]:
            best = (key, start, end)
    if best is None:
        return None
    return {"start_bin": int(best[1]), "end_bin_exclusive": int(best[2])}


def _grid_xy(u_cells, profile):
    axis = np.asarray(profile["row_axis_direction"], dtype=np.float64)
    cross = np.asarray(profile["cross_row_direction"], dtype=np.float64)
    v_min, v_max = profile["cross_row_span_cells"]
    v_center = 0.5 * (float(v_min) + float(v_max))
    point = float(u_cells) * axis + v_center * cross
    return [float(point[0]), float(point[1])]


def _side_record(
    side,
    left_u,
    right_u,
    *,
    profile,
    max_disagreement_m,
    resolution_m,
):
    available = [value for value in (left_u, right_u) if value is not None]
    if not available:
        return {
            "status": "insufficient_structural_support",
            "structural_u_cells": None,
            "structural_grid_xy": None,
            "candidate_u_cells": None,
            "candidate_grid_xy": None,
            "candidate_source": None,
            "left_u_cells": None,
            "right_u_cells": None,
            "side_disagreement_m": None,
        }

    if left_u is None or right_u is None:
        source = "left_only" if left_u is not None else "right_only"
        candidate_u = float(available[0])
        return {
            "status": "ambiguous_single_side",
            "structural_u_cells": None,
            "structural_grid_xy": None,
            "candidate_u_cells": candidate_u,
            "candidate_grid_xy": _grid_xy(candidate_u, profile),
            "candidate_source": source,
            "left_u_cells": None if left_u is None else float(left_u),
            "right_u_cells": None if right_u is None else float(right_u),
            "side_disagreement_m": None,
        }

    disagreement_m = abs(float(left_u) - float(right_u)) * float(resolution_m)
    candidate_u = 0.5 * (float(left_u) + float(right_u))
    if disagreement_m > float(max_disagreement_m) + 1e-12:
        return {
            "status": "ambiguous_single_side",
            "structural_u_cells": None,
            "structural_grid_xy": None,
            "candidate_u_cells": candidate_u,
            "candidate_grid_xy": _grid_xy(candidate_u, profile),
            "candidate_source": "side_disagreement",
            "left_u_cells": float(left_u),
            "right_u_cells": float(right_u),
            "side_disagreement_m": float(disagreement_m),
        }

    return {
        "status": "ok_bilateral",
        "structural_u_cells": candidate_u,
        "structural_grid_xy": _grid_xy(candidate_u, profile),
        "candidate_u_cells": candidate_u,
        "candidate_grid_xy": _grid_xy(candidate_u, profile),
        "candidate_source": "bilateral",
        "left_u_cells": float(left_u),
        "right_u_cells": float(right_u),
        "side_disagreement_m": float(disagreement_m),
    }


def detect_structural_endpoints(
    profile,
    *,
    min_support_fraction,
    min_persistence_m,
    max_internal_gap_m,
    max_side_endpoint_disagreement_m,
):
    """Detect entry/exit structural termination without geometric fallbacks."""
    threshold = float(min_support_fraction)
    persistence_m = float(min_persistence_m)
    gap_m = float(max_internal_gap_m)
    disagreement_m = float(max_side_endpoint_disagreement_m)
    if not 0.0 < threshold <= 1.0:
        raise ValueError("min_support_fraction must be in (0, 1]")
    if persistence_m <= 0.0:
        raise ValueError("min_persistence_m must be > 0")
    if gap_m < 0.0:
        raise ValueError("max_internal_gap_m must be >= 0")
    if disagreement_m < 0.0:
        raise ValueError("max_side_endpoint_disagreement_m must be >= 0")

    left = np.asarray(profile["left_hard_support_fraction"], dtype=np.float64)
    right = np.asarray(profile["right_hard_support_fraction"], dtype=np.float64)
    edges = np.asarray(profile["bin_edges_u_cells"], dtype=np.float64)
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
        raise ValueError("left/right support arrays must be same-length 1D arrays")
    if edges.shape != (left.size + 1,):
        raise ValueError("bin_edges_u_cells must contain N+1 values")

    bin_size_m = float(profile["bin_size_m"])
    resolution_m = float(profile["resolution_m"])
    if bin_size_m <= 0.0 or resolution_m <= 0.0:
        raise ValueError("profile bin_size_m/resolution_m must be > 0")

    min_bins = max(1, int(np.ceil(persistence_m / bin_size_m - 1e-12)))
    max_gap_bins = int(np.floor(gap_m / bin_size_m + 1e-12))

    left_raw = left + 1e-12 >= threshold
    right_raw = right + 1e-12 >= threshold
    left_mask = _close_short_internal_gaps(left_raw, max_gap_bins)
    right_mask = _close_short_internal_gaps(right_raw, max_gap_bins)
    left_run = _longest_persistent_run(left_mask, min_bins)
    right_run = _longest_persistent_run(right_mask, min_bins)

    def _run_u(run, side):
        if run is None:
            return None
        if side == "entry":
            return float(edges[run["start_bin"]])
        if side == "exit":
            return float(edges[run["end_bin_exclusive"]])
        raise ValueError("side must be entry or exit")

    entry = _side_record(
        "entry",
        _run_u(left_run, "entry"),
        _run_u(right_run, "entry"),
        profile=profile,
        max_disagreement_m=disagreement_m,
        resolution_m=resolution_m,
    )
    exit_ = _side_record(
        "exit",
        _run_u(left_run, "exit"),
        _run_u(right_run, "exit"),
        profile=profile,
        max_disagreement_m=disagreement_m,
        resolution_m=resolution_m,
    )

    if entry["status"] not in _VALID_STATUS or exit_["status"] not in _VALID_STATUS:
        raise RuntimeError("unexpected structural endpoint status")

    return {
        "schema_version": 1,
        "label": str(profile.get("label", "")),
        "entry": entry,
        "exit": exit_,
        "support_runs": {
            "left": left_run,
            "right": right_run,
        },
        "support_masks": {
            "left_raw": left_raw.tolist(),
            "right_raw": right_raw.tolist(),
            "left_gap_closed": left_mask.tolist(),
            "right_gap_closed": right_mask.tolist(),
        },
        "parameters": {
            "min_support_fraction": threshold,
            "min_persistence_m": persistence_m,
            "max_internal_gap_m": gap_m,
            "max_side_endpoint_disagreement_m": disagreement_m,
            "min_persistence_bins": int(min_bins),
            "max_internal_gap_bins": int(max_gap_bins),
        },
        "policy": {
            "raw_endpoint_fallback": False,
            "handoff_fallback": False,
            "automatic_parameter_selection": False,
            "navigation_map_modified": False,
            "semantic_promotion": False,
        },
    }
