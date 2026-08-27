"""Targeted-rescan requirement classification for frozen P1 endpoint ROIs.

This module converts the observation-sufficiency diagnosis into an acquisition
requirement layer. It does not edit the navigation map and it does not promote
UNKNOWN to FREE. Repeated-scan UNKNOWN cells are retained as evidence anchors;
only insufficient UNKNOWN classes are marked as requiring additional acquisition.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .observation_sufficiency import (
    LABEL_KNOWN_FREE,
    LABEL_OCCUPIED,
    LABEL_UNKNOWN_NO_GROUND_REFERENCE,
    LABEL_UNKNOWN_NO_OBSERVATION,
    LABEL_UNKNOWN_REPEATED_SCAN,
    LABEL_UNKNOWN_SINGLE_SCAN,
)

RESCAN_OUTSIDE_ENDPOINT = np.uint8(0)
RESCAN_OCCUPIED = np.uint8(1)
RESCAN_KNOWN_FREE = np.uint8(2)
RESCAN_REPEATED_SCAN_ANCHOR = np.uint8(3)
RESCAN_SINGLE_SCAN_REVISIT = np.uint8(4)
RESCAN_NO_OBSERVATION = np.uint8(5)
RESCAN_NO_GROUND_REFERENCE = np.uint8(6)

RESCAN_CLASS_NAMES = {
    int(RESCAN_OUTSIDE_ENDPOINT): "outside_endpoint_roi",
    int(RESCAN_OCCUPIED): "occupied",
    int(RESCAN_KNOWN_FREE): "known_free",
    int(RESCAN_REPEATED_SCAN_ANCHOR): "unknown_repeated_scan_anchor",
    int(RESCAN_SINGLE_SCAN_REVISIT): "rescan_single_scan_revisit",
    int(RESCAN_NO_OBSERVATION): "rescan_ground_known_no_observation",
    int(RESCAN_NO_GROUND_REFERENCE): "rescan_no_ground_reference",
}

_RESCAN_REQUIRED_VALUES = (
    int(RESCAN_SINGLE_SCAN_REVISIT),
    int(RESCAN_NO_OBSERVATION),
    int(RESCAN_NO_GROUND_REFERENCE),
)


def build_targeted_rescan_requirement(sufficiency_labels, endpoint_roi):
    """Map sufficiency classes to endpoint-specific acquisition requirements."""
    suff = np.asarray(sufficiency_labels, dtype=np.uint8)
    roi = np.asarray(endpoint_roi, dtype=bool)
    if suff.ndim != 2 or roi.shape != suff.shape:
        raise ValueError("sufficiency_labels/endpoint_roi shape mismatch")

    out = np.full(suff.shape, RESCAN_OUTSIDE_ENDPOINT, dtype=np.uint8)
    mapping = {
        int(LABEL_OCCUPIED): RESCAN_OCCUPIED,
        int(LABEL_KNOWN_FREE): RESCAN_KNOWN_FREE,
        int(LABEL_UNKNOWN_REPEATED_SCAN): RESCAN_REPEATED_SCAN_ANCHOR,
        int(LABEL_UNKNOWN_SINGLE_SCAN): RESCAN_SINGLE_SCAN_REVISIT,
        int(LABEL_UNKNOWN_NO_OBSERVATION): RESCAN_NO_OBSERVATION,
        int(LABEL_UNKNOWN_NO_GROUND_REFERENCE): RESCAN_NO_GROUND_REFERENCE,
    }
    supported_values = set(mapping)
    present = set(int(v) for v in np.unique(suff[roi]))
    unsupported = sorted(present - supported_values)
    if unsupported:
        raise ValueError(f"unsupported observation-sufficiency labels in ROI: {unsupported}")
    for source_value, target_value in mapping.items():
        out[roi & (suff == source_value)] = target_value
    return out


def _component_summary(mask):
    labels, count = ndimage.label(np.asarray(mask, dtype=bool))
    if int(count) == 0:
        return {
            "component_count": 0,
            "largest_component_cell_count": 0,
        }
    sizes = np.bincount(labels.reshape(-1))[1:]
    return {
        "component_count": int(count),
        "largest_component_cell_count": int(np.max(sizes)) if sizes.size else 0,
    }


def summarize_targeted_rescan_requirement(requirement_labels, *, roi_mask=None):
    """Summarize acquisition requirements without automatically selecting targets."""
    arr = np.asarray(requirement_labels, dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError("requirement_labels must be 2D")
    if not np.isin(arr, list(RESCAN_CLASS_NAMES)).all():
        raise ValueError("requirement_labels contain unsupported values")

    if roi_mask is None:
        roi = arr != RESCAN_OUTSIDE_ENDPOINT
    else:
        roi = np.asarray(roi_mask, dtype=bool)
        if roi.shape != arr.shape:
            raise ValueError("roi_mask shape mismatch")

    total = int(np.count_nonzero(roi))
    classes = {}
    for value, name in RESCAN_CLASS_NAMES.items():
        count = int(np.count_nonzero(roi & (arr == value)))
        classes[name] = {
            "count": count,
            "fraction_of_roi": float(count / total) if total else 0.0,
        }

    required = roi & np.isin(arr, _RESCAN_REQUIRED_VALUES)
    anchor = roi & (arr == RESCAN_REPEATED_SCAN_ANCHOR)
    return {
        "roi_cell_count": total,
        "rescan_required_cell_count": int(np.count_nonzero(required)),
        "rescan_required_fraction_of_roi": (
            float(np.count_nonzero(required) / total) if total else 0.0
        ),
        "repeated_scan_anchor_cell_count": int(np.count_nonzero(anchor)),
        "classes": classes,
        "rescan_required_components": _component_summary(required),
        "automatic_target_selection": False,
        "navigation_map_modified": False,
        "semantic_promotion": False,
    }
