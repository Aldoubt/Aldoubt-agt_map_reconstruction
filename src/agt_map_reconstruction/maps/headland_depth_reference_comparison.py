"""Reference-only comparison between finite headland depth and unbounded diagnostics."""

from __future__ import annotations


def _aggregate_finite_bands(bands):
    records = list(bands or [])
    unknown = sum(int(item.get("unknown_cell_count", 0)) for item in records)
    trusted = sum(int(item.get("trusted_ground_unknown_cell_count", 0)) for item in records)
    if records:
        depth_min = min(float(item.get("depth_min_m")) for item in records)
        depth_max = max(float(item.get("depth_max_m")) for item in records)
    else:
        depth_min = None
        depth_max = None
    return {
        "depth_min_m": depth_min,
        "depth_max_m": depth_max,
        "unknown_cell_count": unknown,
        "trusted_ground_unknown_cell_count": trusted,
        "ground_reference_ceiling_fraction_of_unknown": (
            None if unknown <= 0 else float(trusted / unknown)
        ),
    }


def _historical_side(unbounded, side):
    stats = dict(((unbounded.get(side) or {}).get("conservative_outward") or {}))
    return {
        "unknown_cell_count": int(stats.get("unknown_cell_count", 0)),
        "trusted_ground_unknown_cell_count": int(
            stats.get("trusted_ground_unknown_cell_count", 0)
        ),
        "ground_reference_ceiling_fraction_of_unknown": stats.get(
            "ground_reference_ceiling_fraction_of_unknown"
        ),
        "domain_label": "historical_unbounded_outward_diagnostic",
    }


def compare_headland_depth_with_unbounded(
    headland_depth_evidence,
    unbounded_evidence,
):
    """Summarize both domains without treating them as equivalent peer methods."""
    finite = dict(headland_depth_evidence)
    unbounded = dict(unbounded_evidence)
    sides = {}
    for side in ("entry", "exit"):
        finite_bands = list(((finite.get(side) or {}).get("bands") or []))
        sides[side] = {
            "historical_unbounded": _historical_side(unbounded, side),
            "finite_depth_aggregate": _aggregate_finite_bands(finite_bands),
            "finite_depth_bands": finite_bands,
        }

    return {
        "schema_version": 1,
        "method": "finite_headland_depth_vs_unbounded_reference",
        "finite_depth_metadata": {
            "resolution_m": finite.get("resolution_m"),
            "uncertainty_quantile": finite.get("uncertainty_quantile"),
            "depth_edges_m": list(finite.get("depth_edges_m") or []),
            "max_outward_depth_m": finite.get("max_outward_depth_m"),
            "unresolved_ridge_ids": list(finite.get("unresolved_ridge_ids") or []),
            "sources": dict(finite.get("sources") or {}),
        },
        "entry": sides["entry"],
        "exit": sides["exit"],
        "spatial_domains_equivalent": False,
        "historical_unbounded_metrics_used_for_acceptance": False,
        "finite_depth_profile_is_primary": True,
        "policy": {
            "fraction_difference_reported_as_improvement": False,
            "historical_unbounded_is_reference_only": True,
            "automatic_acceptance": False,
            "navigation_map_modified": False,
            "semantic_promotion": False,
        },
    }
