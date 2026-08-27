"""Evaluate generic support-count thresholds against frozen P1-D3 metrics.

The support grid may represent per-ray or unique-scan counts. This module never
chooses a threshold automatically and never edits the canonical navigation map.
"""

from __future__ import annotations

import numpy as np

from .ray_endpoint_evidence_ab import evaluate_ray_supported_endpoint_envelope


def evaluate_support_thresholds(
    base_map,
    support_count,
    row_aisles,
    handoffs,
    *,
    resolution,
    radius_m,
    baseline_envelope,
    min_support_values,
    support_basis,
):
    counts = np.asarray(support_count)
    base = np.asarray(base_map)
    if counts.shape != base.shape:
        raise ValueError("support_count shape must match base_map")
    if counts.ndim != 2:
        raise ValueError("support_count must be 2D")
    if np.any(counts < 0):
        raise ValueError("support_count must be non-negative")
    basis = str(support_basis)
    if basis not in {"ray", "scan"}:
        raise ValueError("support_basis must be 'ray' or 'scan'")

    thresholds = [int(value) for value in min_support_values]
    if not thresholds or any(value < 1 for value in thresholds):
        raise ValueError("min_support_values must contain integers >= 1")
    if thresholds != sorted(set(thresholds)):
        raise ValueError("min_support_values must be unique and increasing")

    results = []
    for threshold in thresholds:
        mask = counts >= threshold
        result = evaluate_ray_supported_endpoint_envelope(
            base,
            mask,
            row_aisles,
            handoffs,
            resolution=float(resolution),
            radius_m=float(radius_m),
            baseline_envelope=baseline_envelope,
        )
        results.append(
            {
                "min_support": threshold,
                "support_basis": basis,
                "supported_cell_count": int(np.count_nonzero(mask)),
                "overlay_summary": result["overlay_summary"],
                "comparison": result["comparison"],
                "policy": result["policy"],
            }
        )

    return {
        "schema_version": 1,
        "support_basis": basis,
        "thresholds": results,
        "automatic_threshold_selection": False,
        "navigation_map_modified": False,
        "semantic_promotion": False,
    }
