"""Lattice-aware D3.1 structural endpoint recovery.

The row-lattice layer may contain geometry-only inferred aisle slots.  Those
slots are allowed to define where adjacent ridge evidence should be searched,
but they never contribute occupancy/free evidence by themselves.  Structural
support still comes exclusively from the frozen navigation map HARD cells via
the existing inter-aisle ridge profiler.
"""

from __future__ import annotations

import numpy as np

from .structural_endpoint_boundary import fit_structural_endpoint_boundaries
from .structural_ridge_endpoint import (
    build_inter_aisle_ridge_profiles,
    detect_ridge_terminations,
    pair_aisle_structural_endpoints,
)


def _unit(vector):
    value = np.asarray(vector, dtype=np.float64).reshape(-1)
    if value.size != 2:
        raise ValueError("direction must contain exactly two values")
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("direction must be non-zero")
    return value / norm


def lattice_slots_to_rows(lattice_payload):
    """Convert lattice slots into row-like geometry for ridge search only.

    The returned ``region_class == row_aisle`` is an internal geometry contract
    required by ``build_inter_aisle_ridge_profiles``.  It must not be confused
    with semantic/navigation promotion; each row records ``geometry_only`` and
    ``navigation_free_promoted=False`` explicitly.
    """
    payload = dict(lattice_payload)
    if str(payload.get("status", "")) != "ok":
        raise ValueError("row lattice payload must have status=ok")
    slots = list(payload.get("slots") or [])
    if len(slots) < 2:
        raise ValueError("row lattice requires at least two slots")

    rows = []
    for slot in sorted(slots, key=lambda item: int(item["lattice_index"])):
        polygon = np.asarray(slot.get("polygon_xy"), dtype=np.float64)
        centerline = np.asarray(slot.get("centerline_xy"), dtype=np.float64)
        if polygon.ndim != 2 or polygon.shape[1] != 2 or polygon.shape[0] < 4:
            raise ValueError("lattice slot polygon_xy must be Nx2")
        if centerline.shape != (2, 2):
            raise ValueError("lattice slot centerline_xy must be 2x2")
        if bool(slot.get("navigation_free_promoted", False)):
            raise ValueError("lattice slot must not be navigation-free promoted")

        source_labels = list(slot.get("source_band_labels") or [])
        source_label = str(slot.get("source_band_label", ""))
        if source_label and source_label not in source_labels:
            source_labels.append(source_label)

        rows.append(
            {
                "label": str(slot["slot_id"]),
                "region_class": "row_aisle",
                "polygon_xy": polygon.tolist(),
                "centerline_xy": centerline.tolist(),
                "lattice_index": int(slot["lattice_index"]),
                "geometry_source": str(slot.get("source", "")),
                "evidence_strength": str(slot.get("evidence_strength", "")),
                "source_band_labels": source_labels,
                "geometry_only": True,
                "navigation_free_promoted": False,
                "semantic_promotion": False,
            }
        )
    return rows


def build_lattice_structural_endpoint_bundle(
    base_map,
    lattice_payload,
    *,
    resolution_m,
    bin_size_m,
    min_support_fraction,
    min_persistence_m,
    max_internal_gap_m,
    max_side_endpoint_disagreement_m,
    residual_floor_m,
    mad_scale,
    min_inlier_count,
    max_fit_rmse_m,
):
    """Recover structural endpoints using lattice slots only as search geometry."""
    rows = lattice_slots_to_rows(lattice_payload)
    axis = _unit(lattice_payload.get("row_axis_direction"))
    cross = np.asarray(lattice_payload.get("cross_row_direction"), dtype=np.float64)
    if cross.shape != (2,):
        cross = np.array([-axis[1], axis[0]], dtype=np.float64)
    else:
        cross = _unit(cross)
    if abs(float(axis @ cross)) > 1e-6:
        cross = np.array([-axis[1], axis[0]], dtype=np.float64)

    profiles = build_inter_aisle_ridge_profiles(
        base_map,
        rows,
        resolution_m=float(resolution_m),
        bin_size_m=float(bin_size_m),
        row_axis=axis,
    )
    terminations = [
        detect_ridge_terminations(
            profile,
            min_support_fraction=float(min_support_fraction),
            min_persistence_m=float(min_persistence_m),
            max_internal_gap_m=float(max_internal_gap_m),
        )
        for profile in profiles
    ]
    paired = pair_aisle_structural_endpoints(
        rows,
        terminations,
        row_axis=axis,
        max_side_endpoint_disagreement_m=float(max_side_endpoint_disagreement_m),
    )

    provenance = {row["label"]: row for row in rows}
    paired_out = []
    for record in paired:
        item = dict(record)
        row = provenance[item["label"]]
        item["lattice_index"] = int(row["lattice_index"])
        item["geometry_source"] = row["geometry_source"]
        item["evidence_strength"] = row["evidence_strength"]
        item["source_band_labels"] = list(row["source_band_labels"])
        paired_out.append(item)

    robust = fit_structural_endpoint_boundaries(
        paired_out,
        row_axis=axis,
        cross_axis=cross,
        resolution_m=float(resolution_m),
        residual_floor_m=float(residual_floor_m),
        mad_scale=float(mad_scale),
        min_inlier_count=int(min_inlier_count),
        max_fit_rmse_m=float(max_fit_rmse_m),
    )

    observed_count = sum(
        1
        for row in rows
        if row["geometry_source"] in {"observed_row_aisle", "observed_split_group"}
    )
    inferred_count = sum(
        1 for row in rows if row["geometry_source"] == "lattice_inferred_wide_band"
    )
    return {
        "schema_version": 1,
        "method": "lattice_geometry_plus_inter_slot_hard_evidence",
        "row_axis_direction": axis.tolist(),
        "cross_row_direction": cross.tolist(),
        "resolution_m": float(resolution_m),
        "lattice_slot_count": len(rows),
        "observed_slot_count": observed_count,
        "inferred_slot_count": inferred_count,
        "ridge_profile_count": len(profiles),
        "lattice_rows": rows,
        "ridge_profiles": profiles,
        "ridge_terminations": terminations,
        "paired_endpoints": paired_out,
        "robust_boundary": robust,
        "parameters": {
            "bin_size_m": float(bin_size_m),
            "min_support_fraction": float(min_support_fraction),
            "min_persistence_m": float(min_persistence_m),
            "max_internal_gap_m": float(max_internal_gap_m),
            "max_side_endpoint_disagreement_m": float(max_side_endpoint_disagreement_m),
            "residual_floor_m": float(residual_floor_m),
            "mad_scale": float(mad_scale),
            "min_inlier_count": int(min_inlier_count),
            "max_fit_rmse_m": float(max_fit_rmse_m),
        },
        "policy": {
            "inferred_slot_role": "ridge_search_geometry_only",
            "inferred_slot_supplies_structural_evidence": False,
            "inferred_slot_promoted_to_navigation_free": False,
            "generic_outer_wall_used_as_ridge": False,
            "unknown_counted_as_structural": False,
            "automatic_parameter_selection": False,
            "automatic_acceptance": False,
            "navigation_map_modified": False,
            "semantic_promotion": False,
        },
    }
