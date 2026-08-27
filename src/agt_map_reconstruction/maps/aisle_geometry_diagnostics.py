"""Diagnose aisle-width limits and clearance-connectivity anomalies."""

from __future__ import annotations

import numpy as np


def robust_width_outlier_threshold(widths):
    values = np.asarray(widths, dtype=float).reshape(-1)
    if values.size < 4:
        return None
    q1, q3 = np.quantile(values, [0.25, 0.75])
    return float(q3 + 1.5 * (q3 - q1))


def diagnose_aisle_geometry(validation, feasible_margin_cells=0.5):
    """Separate width-floor failures from connectivity anomalies.

    The conservative feasible radius subtracts half a grid cell from the
    recovered half width. This prevents equality at a raster boundary from
    being over-interpreted as a connectivity defect.
    """
    aisles = list(validation.get("aisles", []))
    tests = validation.get("clearance_tests", {})
    if not aisles or not tests:
        raise ValueError("validation must contain aisles and clearance_tests")

    resolution = float(validation["resolution_m"])
    radii = sorted(float(key) for key in tests)
    min_radius = radii[0]
    widths = [float(item.get("width_m", 0.0)) for item in aisles]
    wide_threshold = robust_width_outlier_threshold(widths)
    half_cell_margin = float(feasible_margin_cells) * resolution

    rows = []
    for aisle in aisles:
        label = str(aisle.get("label", f"A{int(aisle['aisle_id']):02d}"))
        width = float(aisle.get("width_m", 0.0))
        half_width = 0.5 * width
        passes = {
            float(key): bool(value)
            for key, value in aisle.get("clearance_pass", {}).items()
        }
        passing = [radius for radius in radii if passes.get(radius, False)]
        failing = [radius for radius in radii if not passes.get(radius, False)]
        feasible_limit = max(0.0, half_width - half_cell_margin)
        unexpected = [
            radius
            for radius in failing
            if radius <= feasible_limit + 1e-12
        ]

        if passes.get(min_radius, False):
            minimum_mode = "pass"
        elif min_radius <= feasible_limit + 1e-12:
            minimum_mode = "connectivity_limited"
        else:
            minimum_mode = "width_limited"

        rows.append({
            "aisle_id": int(aisle["aisle_id"]),
            "label": label,
            "width_m": width,
            "length_m": float(aisle.get("length_m", 0.0)),
            "theoretical_half_width_m": half_width,
            "conservative_width_radius_limit_m": feasible_limit,
            "max_passing_radius_m": max(passing) if passing else None,
            "minimum_clearance_mode": minimum_mode,
            "unexpected_failed_radii_m": unexpected,
            "first_unexpected_failed_radius_m": (
                unexpected[0] if unexpected else None
            ),
            "wide_width_outlier": bool(
                wide_threshold is not None
                and width > wide_threshold + 1e-12
            ),
        })

    summary = {
        "minimum_clearance_radius_m": min_radius,
        "minimum_failures": [
            row["label"]
            for row in rows
            if row["minimum_clearance_mode"] != "pass"
        ],
        "minimum_width_limited": [
            row["label"]
            for row in rows
            if row["minimum_clearance_mode"] == "width_limited"
        ],
        "minimum_connectivity_limited": [
            row["label"]
            for row in rows
            if row["minimum_clearance_mode"] == "connectivity_limited"
        ],
        "unexpected_connectivity_failures": [
            row["label"]
            for row in rows
            if row["unexpected_failed_radii_m"]
        ],
        "wide_width_outliers": [
            row["label"] for row in rows if row["wide_width_outlier"]
        ],
        "wide_width_outlier_threshold_m": wide_threshold,
    }
    return {
        "schema_version": 1,
        "policy": {
            "feasible_margin_cells": float(feasible_margin_cells),
            "wide_width_outlier_method": "Q3 + 1.5 * IQR",
        },
        "summary": summary,
        "aisles": rows,
    }
