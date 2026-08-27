"""Classify recovered row-aligned free bands into aisles and wide-area candidates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RowBandClassification:
    row_aisles: list[dict]
    open_area_candidates: list[dict]
    width_outlier_threshold_m: float | None
    iqr_factor: float


def robust_upper_width_threshold(widths, iqr_factor=1.5, min_samples=4):
    values = np.asarray(widths, dtype=float).reshape(-1)
    if float(iqr_factor) < 0.0:
        raise ValueError("iqr_factor must be >= 0")
    if int(min_samples) < 1:
        raise ValueError("min_samples must be >= 1")
    if values.size < int(min_samples):
        return None
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("widths must be finite and non-negative")
    q1, q3 = np.quantile(values, [0.25, 0.75])
    return float(q3 + float(iqr_factor) * (q3 - q1))


def classify_row_bands(bands, iqr_factor=1.5):
    """Separate ordinary row aisles from unusually wide row-aligned bands.

    Wide bands are deliberately named ``wide_open_area_candidate`` rather than
    ``headland`` because width alone cannot establish headland topology.
    """
    raw = [dict(item) for item in bands]
    widths = [float(item.get("width_m", 0.0)) for item in raw]
    threshold = robust_upper_width_threshold(widths, iqr_factor=iqr_factor)

    row_aisles = []
    open_candidates = []
    for item in raw:
        source_id = int(
            item.get("aisle_id", len(row_aisles) + len(open_candidates) + 1)
        )
        source_label = str(item.get("label", f"A{source_id:02d}"))
        item["source_band_id"] = source_id
        item["source_band_label"] = source_label
        width = float(item.get("width_m", 0.0))
        is_wide = threshold is not None and width > threshold + 1e-12

        if is_wide:
            item.pop("aisle_id", None)
            item["region_id"] = len(open_candidates) + 1
            item["label"] = f"O{len(open_candidates) + 1:02d}"
            item["region_class"] = "wide_open_area_candidate"
            open_candidates.append(item)
        else:
            item["aisle_id"] = len(row_aisles) + 1
            item["label"] = f"A{len(row_aisles) + 1:02d}"
            item["region_class"] = "row_aisle"
            row_aisles.append(item)

    return RowBandClassification(
        row_aisles=row_aisles,
        open_area_candidates=open_candidates,
        width_outlier_threshold_m=threshold,
        iqr_factor=float(iqr_factor),
    )
