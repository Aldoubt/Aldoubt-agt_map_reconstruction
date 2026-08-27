"""Evaluate trajectory-aware observed-free support against frozen P1-D3 metrics.

This module never edits the canonical navigation map. It creates an in-memory
evaluation overlay where only UNKNOWN cells with explicit ray support are treated
as observed free, then reruns the frozen endpoint-envelope metric geometry.
"""

from __future__ import annotations

import numpy as np

from .headland_endpoint_envelope import analyze_endpoint_side_envelopes
from .headland_evidence_gap import attach_endpoint_evidence_gaps
from .navigation_export import FREE_VALUE, OCCUPIED_VALUE, UNKNOWN_VALUE


def build_observation_evaluation_overlay(base_map, ray_support_mask):
    """Return an evaluation-only map with ray-supported UNKNOWN cells set free."""
    base = np.asarray(base_map, dtype=np.uint8)
    support = np.asarray(ray_support_mask).astype(bool)
    if base.ndim != 2:
        raise ValueError("base_map must be 2D")
    if support.shape != base.shape:
        raise ValueError("ray_support_mask shape must match base_map")
    if not np.isin(base, [OCCUPIED_VALUE, UNKNOWN_VALUE, FREE_VALUE]).all():
        raise ValueError("base_map contains unsupported gray values")

    overlay = base.copy()
    promoted_unknown = support & (base == UNKNOWN_VALUE)
    overlay[promoted_unknown] = FREE_VALUE

    return overlay, {
        "ray_supported_cell_count": int(np.count_nonzero(support)),
        "ray_supported_unknown_cell_count": int(np.count_nonzero(promoted_unknown)),
        "ray_supported_existing_free_cell_count": int(
            np.count_nonzero(support & (base == FREE_VALUE))
        ),
        "ray_supported_occupied_cell_count_ignored": int(
            np.count_nonzero(support & (base == OCCUPIED_VALUE))
        ),
        "navigation_map_modified": False,
        "evaluation_overlay_only": True,
        "semantic_promotion": False,
    }


def _best_metrics(side):
    best = side.get("strict", {}).get("best_component")
    if best is None:
        return {
            "cross_row_coverage_fraction": None,
            "endpoint_distance_median_m": None,
            "max_outward_depth_m": None,
        }
    return {
        "cross_row_coverage_fraction": best.get("cross_row_coverage_fraction"),
        "endpoint_distance_median_m": best.get("endpoint_distance_median_m"),
        "max_outward_depth_m": best.get("max_outward_depth_m"),
    }


def _delta(candidate, baseline, *, smaller_is_better=False):
    if candidate is None or baseline is None:
        return None
    value = float(candidate) - float(baseline)
    return -value if smaller_is_better else value


def compare_endpoint_envelopes(baseline, candidate):
    """Compare strict D3 endpoint metrics while asserting frozen geometry."""
    if float(candidate["radius_m"]) != float(baseline["radius_m"]):
        raise ValueError("candidate radius differs from frozen baseline")
    if candidate["eligible_row_labels"] != baseline["eligible_row_labels"]:
        raise ValueError("candidate eligible rows differ from frozen baseline")
    if not np.allclose(
        candidate["row_axis_direction"], baseline["row_axis_direction"], atol=1e-10, rtol=0.0
    ):
        raise ValueError("candidate row axis differs from frozen baseline")
    if not np.allclose(
        candidate["row_cross_span"], baseline["row_cross_span"], atol=1e-8, rtol=0.0
    ):
        raise ValueError("candidate cross-row span differs from frozen baseline")

    sides = {}
    for side_name in ("entry", "exit"):
        base_metrics = _best_metrics(baseline["sides"][side_name])
        candidate_metrics = _best_metrics(candidate["sides"][side_name])
        sides[side_name] = {
            "baseline": base_metrics,
            "candidate": candidate_metrics,
            "delta": {
                "cross_row_coverage_fraction": _delta(
                    candidate_metrics["cross_row_coverage_fraction"],
                    base_metrics["cross_row_coverage_fraction"],
                ),
                "endpoint_distance_reduction_m": _delta(
                    candidate_metrics["endpoint_distance_median_m"],
                    base_metrics["endpoint_distance_median_m"],
                    smaller_is_better=True,
                ),
                "max_outward_depth_gain_m": _delta(
                    candidate_metrics["max_outward_depth_m"],
                    base_metrics["max_outward_depth_m"],
                ),
            },
        }
    return {
        "geometry_frozen": True,
        "sides": sides,
        "automatic_acceptance": False,
        "semantic_promotion": False,
    }


def evaluate_ray_supported_endpoint_envelope(
    base_map,
    ray_support_mask,
    row_aisles,
    handoffs,
    *,
    resolution,
    radius_m,
    baseline_envelope,
):
    """Run the D3 metric on an evaluation-only ray-supported free overlay."""
    overlay, overlay_summary = build_observation_evaluation_overlay(
        base_map, ray_support_mask
    )
    candidate = analyze_endpoint_side_envelopes(
        overlay,
        row_aisles,
        handoffs,
        resolution=float(resolution),
        radius_m=float(radius_m),
    )
    attach_endpoint_evidence_gaps(candidate)
    comparison = compare_endpoint_envelopes(baseline_envelope, candidate)
    return {
        "overlay_summary": overlay_summary,
        "candidate_envelope": candidate,
        "comparison": comparison,
        "policy": {
            "only_unknown_with_ray_support_is_free_in_evaluation_overlay": True,
            "occupied_is_never_overridden": True,
            "canonical_navigation_map_is_not_modified": True,
            "automatic_acceptance": False,
            "semantic_promotion": False,
        },
    }
