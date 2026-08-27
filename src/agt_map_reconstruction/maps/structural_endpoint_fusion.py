"""Evidence-level fusion for P1-D3.1 structural ridge endpoints.

PGM HARD evidence remains authoritative wherever it already supports a ridge
termination. Targeted 3D evidence may fill only PGM-unsupported ridge slots, and
only when the 3D audit marks the ridge as endpoint-eligible. Local-only 3D
structure is preserved as provenance but never promoted to an endpoint.

The fused bundle is still diagnostic geometry. It does not modify the
navigation map, promote semantic free space, or let inferred lattice geometry
supply structural evidence by itself.
"""

from __future__ import annotations

from copy import deepcopy

from .structural_ridge_endpoint import pair_aisle_structural_endpoints


def _three_d_lookup(audit_payload):
    return {
        str(item.get("ridge_id", "")): item
        for item in (audit_payload.get("ridge_audits") or [])
        if str(item.get("ridge_id", ""))
    }


def _three_d_as_termination(source, audit, resolution_m):
    return {
        "schema_version": 3,
        "ridge_id": str(source["ridge_id"]),
        "left_aisle_label": str(source["left_aisle_label"]),
        "right_aisle_label": str(source["right_aisle_label"]),
        "resolution_m": float(resolution_m),
        "status": "ok",
        "entry_u_cells": float(audit["entry_u_cells"]),
        "exit_u_cells": float(audit["exit_u_cells"]),
        "entry_grid_xy": list(audit["entry_grid_xy"]),
        "exit_grid_xy": list(audit["exit_grid_xy"]),
        "evidence_source": "height_3d",
        "source_pgm_status": str(source.get("status", "")),
        "three_d_status": str(audit.get("status", "")),
        "structural_span_fraction": audit.get("structural_span_fraction"),
        "three_d_evidence_summary": deepcopy(audit.get("evidence_summary") or {}),
        "policy": {
            "pgm_was_unsupported": True,
            "endpoint_support_requires_3d_span_gate": True,
            "geometry_only_lattice_supplies_structural_evidence": False,
            "navigation_map_modified": False,
            "semantic_promotion": False,
        },
    }


def fuse_structural_endpoint_evidence(structural_bundle, three_d_audit):
    """Fuse PGM and targeted 3D ridge endpoint evidence without semantic promotion."""
    bundle = deepcopy(dict(structural_bundle))
    resolution = float(bundle.get("resolution_m", 0.0))
    if resolution <= 0.0:
        raise ValueError("structural bundle resolution_m must be > 0")
    rows = list(bundle.get("lattice_rows") or [])
    ridges = list(bundle.get("ridge_terminations") or [])
    if not rows or not ridges:
        raise ValueError("structural bundle must contain lattice_rows and ridge_terminations")

    audit_by_id = _three_d_lookup(dict(three_d_audit))
    fused = []
    pgm_supported = 0
    three_d_supported = 0
    local_3d_only = 0

    for source in ridges:
        ridge_id = str(source.get("ridge_id", ""))
        if not ridge_id:
            raise ValueError("ridge termination missing ridge_id")

        if source.get("status") == "ok":
            item = deepcopy(source)
            item["evidence_source"] = "pgm_hard"
            item["source_pgm_status"] = "ok"
            item["three_d_status"] = None
            item["local_3d_structure_observed"] = False
            fused.append(item)
            pgm_supported += 1
            continue

        audit = audit_by_id.get(ridge_id)
        if audit is not None and audit.get("status") == "ok_3d_structural_support":
            required = ("entry_u_cells", "exit_u_cells", "entry_grid_xy", "exit_grid_xy")
            if any(audit.get(key) is None for key in required):
                raise ValueError(f"3D audit {ridge_id} is endpoint-eligible but missing endpoint geometry")
            item = _three_d_as_termination(source, audit, resolution)
            item["local_3d_structure_observed"] = True
            fused.append(item)
            three_d_supported += 1
            continue

        item = deepcopy(source)
        item["evidence_source"] = "unresolved"
        item["source_pgm_status"] = str(source.get("status", ""))
        item["three_d_status"] = None if audit is None else str(audit.get("status", ""))
        local_observed = bool(
            audit is not None
            and audit.get("status") == "insufficient_longitudinal_structural_span"
            and (audit.get("evidence_summary") or {}).get("supported_bin_count", 0) > 0
        )
        item["local_3d_structure_observed"] = local_observed
        if audit is not None:
            item["structural_span_fraction"] = audit.get("structural_span_fraction")
            item["three_d_evidence_summary"] = deepcopy(audit.get("evidence_summary") or {})
        if local_observed:
            local_3d_only += 1
        fused.append(item)

    parameters = dict(bundle.get("parameters") or {})
    if "max_side_endpoint_disagreement_m" not in parameters:
        raise ValueError("structural bundle missing max_side_endpoint_disagreement_m")
    paired = pair_aisle_structural_endpoints(
        rows,
        fused,
        row_axis=bundle.get("row_axis_direction"),
        max_side_endpoint_disagreement_m=float(parameters["max_side_endpoint_disagreement_m"]),
    )

    row_provenance = {str(row["label"]): row for row in rows}
    paired_out = []
    for record in paired:
        item = deepcopy(record)
        row = row_provenance.get(str(item.get("label", "")), {})
        for key in ("lattice_index", "geometry_source", "evidence_strength", "source_band_labels"):
            if key in row:
                item[key] = deepcopy(row[key])
        paired_out.append(item)

    unresolved = sum(1 for item in fused if item.get("status") != "ok")
    bundle["schema_version"] = max(2, int(bundle.get("schema_version", 1)))
    bundle["method"] = "pgm_plus_3d_structural_ridge_fusion"
    bundle["ridge_terminations"] = fused
    bundle["paired_endpoints"] = paired_out
    bundle.pop("robust_boundary", None)
    bundle["fusion_summary"] = {
        "pgm_supported_ridge_count": pgm_supported,
        "three_d_supported_ridge_count": three_d_supported,
        "local_3d_only_ridge_count": local_3d_only,
        "unresolved_ridge_count": unresolved,
    }
    bundle["fusion_policy"] = {
        "pgm_supported_ridge_has_priority": True,
        "three_d_fills_only_pgm_unsupported_ridges": True,
        "three_d_endpoint_requires_ok_status": True,
        "geometry_only_lattice_supplies_structural_evidence": False,
        "local_3d_structure_promoted_to_endpoint_support": False,
        "ridge_outliers_deleted": False,
        "automatic_acceptance": False,
        "navigation_map_modified": False,
        "semantic_promotion": False,
    }
    # Alias under policy for stable callers that inspect the top-level policy.
    bundle["policy"] = {**dict(bundle.get("policy") or {}), **bundle["fusion_policy"]}
    return bundle
