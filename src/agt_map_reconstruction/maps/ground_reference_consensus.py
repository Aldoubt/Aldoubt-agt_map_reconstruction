"""Build a conservative consensus ground-height reference from two local models.

The consensus is geometry-only. A cell is usable only when both local references
are finite, the nearest observed-ground support is within an explicit distance,
and the two models agree within an explicit height tolerance. Rejected cells
remain NaN and cannot contribute ray free-space support.
"""

from __future__ import annotations

import numpy as np


def _validate_threshold(name, value):
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return value


def build_ground_reference_consensus(
    reference_a,
    reference_b,
    nearest_support_distance_m,
    max_support_distance_m,
    max_model_disagreement_m,
):
    """Return midpoint reference only where support proximity and model agreement pass."""
    a = np.asarray(reference_a, dtype=np.float64)
    b = np.asarray(reference_b, dtype=np.float64)
    distance = np.asarray(nearest_support_distance_m, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or distance.ndim != 2:
        raise ValueError("reference and distance grids must be 2D")
    if a.shape != b.shape or a.shape != distance.shape:
        raise ValueError("reference and distance grids must have matching shapes")

    max_distance = _validate_threshold(
        "max_support_distance_m", max_support_distance_m
    )
    max_disagreement = _validate_threshold(
        "max_model_disagreement_m", max_model_disagreement_m
    )

    finite = np.isfinite(a) & np.isfinite(b) & np.isfinite(distance)
    disagreement = np.full(a.shape, np.nan, dtype=np.float64)
    disagreement[finite] = np.abs(a[finite] - b[finite])

    distance_ok = finite & (distance <= max_distance + 1e-12)
    disagreement_ok = finite & (disagreement <= max_disagreement + 1e-12)
    confidence = distance_ok & disagreement_ok

    consensus = np.full(a.shape, np.nan, dtype=np.float32)
    consensus[confidence] = (0.5 * (a[confidence] + b[confidence])).astype(np.float32)

    finite_count = int(np.count_nonzero(finite))
    accepted = int(np.count_nonzero(confidence))
    rejected_distance = int(np.count_nonzero(finite & ~distance_ok))
    rejected_disagreement = int(np.count_nonzero(finite & distance_ok & ~disagreement_ok))

    return {
        "ground_reference": consensus,
        "confidence_mask": confidence,
        "model_disagreement_m": disagreement.astype(np.float32),
        "summary": {
            "schema_version": 1,
            "finite_input_cell_count": finite_count,
            "accepted_cell_count": accepted,
            "accepted_fraction_of_finite": (
                0.0 if finite_count == 0 else float(accepted / finite_count)
            ),
            "rejected_distance_cell_count": rejected_distance,
            "rejected_disagreement_cell_count": rejected_disagreement,
            "max_support_distance_m": max_distance,
            "max_model_disagreement_m": max_disagreement,
            "reference_policy": "midpoint_of_two_local_models_when_confident",
            "semantic_promotion": False,
        },
    }
