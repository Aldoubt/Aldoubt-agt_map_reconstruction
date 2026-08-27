"""Observation-sufficiency classification for conservative map interpretation.

This layer is diagnostic only. It never edits or promotes cells in the canonical
navigation map. Unknown cells are partitioned by two independent prerequisites:
trusted ground reference and trajectory-derived unique-scan support.
"""

from __future__ import annotations

import numpy as np

from .navigation_export import FREE_VALUE, OCCUPIED_VALUE, UNKNOWN_VALUE

LABEL_OCCUPIED = np.uint8(0)
LABEL_KNOWN_FREE = np.uint8(1)
LABEL_UNKNOWN_NO_GROUND_REFERENCE = np.uint8(2)
LABEL_UNKNOWN_NO_OBSERVATION = np.uint8(3)
LABEL_UNKNOWN_SINGLE_SCAN = np.uint8(4)
LABEL_UNKNOWN_REPEATED_SCAN = np.uint8(5)

LABEL_NAMES = {
    int(LABEL_OCCUPIED): "occupied",
    int(LABEL_KNOWN_FREE): "known_free",
    int(LABEL_UNKNOWN_NO_GROUND_REFERENCE): "unknown_no_ground_reference",
    int(LABEL_UNKNOWN_NO_OBSERVATION): "unknown_ground_reference_no_observation",
    int(LABEL_UNKNOWN_SINGLE_SCAN): "unknown_single_scan_support",
    int(LABEL_UNKNOWN_REPEATED_SCAN): "unknown_repeated_scan_support",
}


def build_observation_sufficiency_labels(
    base_map,
    ground_reference,
    scan_support_count,
    *,
    min_repeated_scans=2,
):
    """Classify the frozen map without changing any navigation semantics."""
    base = np.asarray(base_map, dtype=np.uint8)
    ground = np.asarray(ground_reference, dtype=np.float64)
    support = np.asarray(scan_support_count)
    if base.ndim != 2:
        raise ValueError("base_map must be 2D")
    if ground.shape != base.shape or support.shape != base.shape:
        raise ValueError("base_map/ground_reference/scan_support_count shape mismatch")
    if int(min_repeated_scans) < 2:
        raise ValueError("min_repeated_scans must be >= 2")
    if not np.isin(base, [OCCUPIED_VALUE, UNKNOWN_VALUE, FREE_VALUE]).all():
        raise ValueError("base_map contains unsupported gray values")
    if np.issubdtype(support.dtype, np.signedinteger) and np.any(support < 0):
        raise ValueError("scan_support_count must be non-negative")

    labels = np.empty(base.shape, dtype=np.uint8)
    occupied = base == OCCUPIED_VALUE
    free = base == FREE_VALUE
    unknown = base == UNKNOWN_VALUE
    finite_ground = np.isfinite(ground)

    labels[occupied] = LABEL_OCCUPIED
    labels[free] = LABEL_KNOWN_FREE

    no_ground = unknown & ~finite_ground
    ground_ok = unknown & finite_ground
    no_observation = ground_ok & (support == 0)
    single_scan = ground_ok & (support == 1)
    repeated_scan = ground_ok & (support >= int(min_repeated_scans))
    intermediate = ground_ok & ~(no_observation | single_scan | repeated_scan)
    if np.any(intermediate):
        raise ValueError(
            "scan_support_count contains values not represented by the current classification"
        )

    labels[no_ground] = LABEL_UNKNOWN_NO_GROUND_REFERENCE
    labels[no_observation] = LABEL_UNKNOWN_NO_OBSERVATION
    labels[single_scan] = LABEL_UNKNOWN_SINGLE_SCAN
    labels[repeated_scan] = LABEL_UNKNOWN_REPEATED_SCAN
    return labels


def summarize_observation_sufficiency(labels, *, roi_mask=None):
    """Return deterministic counts/fractions for a full grid or supplied ROI."""
    arr = np.asarray(labels, dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError("labels must be 2D")
    if not np.isin(arr, list(LABEL_NAMES)).all():
        raise ValueError("labels contain unsupported values")

    if roi_mask is None:
        mask = np.ones(arr.shape, dtype=bool)
    else:
        mask = np.asarray(roi_mask, dtype=bool)
        if mask.shape != arr.shape:
            raise ValueError("roi_mask shape mismatch")

    total = int(np.count_nonzero(mask))
    counts = {}
    for value, name in LABEL_NAMES.items():
        count = int(np.count_nonzero(mask & (arr == value)))
        counts[name] = {
            "count": count,
            "fraction_of_roi": float(count / total) if total else 0.0,
        }

    unknown_values = (
        int(LABEL_UNKNOWN_NO_GROUND_REFERENCE),
        int(LABEL_UNKNOWN_NO_OBSERVATION),
        int(LABEL_UNKNOWN_SINGLE_SCAN),
        int(LABEL_UNKNOWN_REPEATED_SCAN),
    )
    unknown_mask = mask & np.isin(arr, unknown_values)
    unknown_total = int(np.count_nonzero(unknown_mask))
    for value in unknown_values:
        name = LABEL_NAMES[value]
        count = counts[name]["count"]
        counts[name]["fraction_of_unknown"] = (
            float(count / unknown_total) if unknown_total else 0.0
        )

    return {
        "roi_cell_count": total,
        "unknown_cell_count": unknown_total,
        "classes": counts,
        "navigation_map_modified": False,
        "automatic_semantic_promotion": False,
        "semantic_promotion": False,
    }
