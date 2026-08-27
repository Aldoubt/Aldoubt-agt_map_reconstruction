"""Summarize the evidence gap between strict and unknown-relaxed endpoint envelopes."""

from __future__ import annotations


def _best(policy):
    if not isinstance(policy, dict):
        return None
    return policy.get("best_component")


def summarize_endpoint_evidence_gap(side_result):
    """Return continuous evidence-gap metrics without semantic promotion.

    The strict envelope is supported only by confirmed free cells. The relaxed
    envelope allows unknown while still respecting hard occupied cells. A large
    relaxed gain accompanied by an unknown fraction near one means geometry is
    plausible but observational support is missing; it is not a HEADLAND pass.
    """
    strict = _best(side_result.get("strict"))
    relaxed = _best(side_result.get("relaxed_unknown_allowed"))
    if strict is None or relaxed is None:
        return {
            "available": False,
            "coverage_gain": None,
            "endpoint_distance_reduction_m": None,
            "outward_depth_gain_m": None,
            "relaxed_unknown_fraction": None,
            "relaxed_observed_fraction": None,
            "semantic_promotion": False,
        }

    strict_coverage = float(strict["cross_row_coverage_fraction"])
    relaxed_coverage = float(relaxed["cross_row_coverage_fraction"])
    strict_endpoint = strict.get("endpoint_distance_median_m")
    relaxed_endpoint = relaxed.get("endpoint_distance_median_m")
    strict_depth = float(strict["max_outward_depth_m"])
    relaxed_depth = float(relaxed["max_outward_depth_m"])
    unknown_fraction = float(relaxed["unknown_cell_fraction"])

    return {
        "available": True,
        "coverage_gain": relaxed_coverage - strict_coverage,
        "endpoint_distance_reduction_m": (
            None
            if strict_endpoint is None or relaxed_endpoint is None
            else float(strict_endpoint) - float(relaxed_endpoint)
        ),
        "outward_depth_gain_m": relaxed_depth - strict_depth,
        "relaxed_unknown_fraction": unknown_fraction,
        "relaxed_observed_fraction": 1.0 - unknown_fraction,
        "semantic_promotion": False,
    }


def attach_endpoint_evidence_gaps(result):
    """Attach evidence-gap summaries to an endpoint-envelope result in place."""
    sides = result.get("sides", {})
    for side_name in ("entry", "exit"):
        side = sides.get(side_name)
        if isinstance(side, dict):
            side["evidence_gap"] = summarize_endpoint_evidence_gap(side)
    result["evidence_gap_policy"] = {
        "meaning": (
            "strict-to-relaxed change quantifies missing observation support; "
            "relaxed connectivity is diagnostic only"
        ),
        "semantic_promotion": False,
    }
    return result
