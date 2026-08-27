"""Persist evidence-derived semantic geometry and a ready-to-load Nav2 map."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .aisle_reconstruction import recover_aisle_rectangles, write_aisle_bundle
from .ground_evidence import EvidenceClass
from .navigation_export import rasterize_aisles, write_navigation_bundle
from .semantic_reconstruction import (
    LABEL_AISLE,
    LABEL_OBSTACLE_CANDIDATE,
    LABEL_OCCUPIED_CONFIRMED,
    corridor_seed_from_evidence,
    refine_occupied_evidence_with_aisle_prior,
    semantic_labels_from_evidence,
)


def _normalized_direction(row_direction):
    direction = np.asarray(row_direction, dtype=float).reshape(-1)
    if direction.size != 2:
        raise ValueError("row_direction must contain exactly two values")
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ValueError("row_direction must be non-zero")
    return direction / norm


def write_semantic_navigation_assets(
    evidence,
    metadata,
    row_direction,
    output_dir,
    min_longitudinal_support_ratio=0.50,
    min_width_m=0.30,
    min_length_m=2.0,
    include_interpolated=True,
    navigation_clearance_radii_m=(0.20, 0.25, 0.30, 0.35, 0.40, 0.50),
):
    """Write evidence-derived semantics plus a conservative Nav2 bundle.

    Four-state evidence remains authoritative and is persisted unchanged.
    Recovered aisle geometry may reinterpret only confirmed occupied evidence
    that conflicts with a longitudinal aisle as an advisory obstacle candidate.
    Unknown/interpolated cells are never promoted. Confirmed occupied evidence
    outside aisle geometry remains hard, and candidate conflicts stay visible in
    ``candidate_mask.npy`` even when the static base map uses the aisle prior.
    """
    evidence = np.asarray(evidence, dtype=np.uint8)
    expected_shape = (int(metadata.height), int(metadata.width))
    if evidence.shape != expected_shape:
        raise ValueError(
            f"evidence shape {evidence.shape} does not match metadata shape {expected_shape}"
        )

    direction = _normalized_direction(row_direction)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    raw_labels = semantic_labels_from_evidence(evidence)
    seed = corridor_seed_from_evidence(
        evidence, include_interpolated=include_interpolated
    )
    aisles = recover_aisle_rectangles(
        seed,
        direction,
        float(metadata.resolution),
        min_longitudinal_support_ratio=min_longitudinal_support_ratio,
        min_width_m=min_width_m,
        min_length_m=min_length_m,
    )
    aisle_prior = rasterize_aisles(aisles, evidence.shape)
    labels = refine_occupied_evidence_with_aisle_prior(raw_labels, aisle_prior)
    aisle_conflict_candidates = (
        (raw_labels == LABEL_OCCUPIED_CONFIRMED)
        & (labels == LABEL_OBSTACLE_CANDIDATE)
    )

    np.save(output / "evidence.npy", evidence)
    np.save(output / "semantic_labels.npy", labels)
    np.save(output / "corridor_seed.npy", seed.astype(np.uint8))
    aisle_payload = write_aisle_bundle(
        aisles, metadata, output / "aisle_rectangles.json"
    )

    evidence_counts = {
        "unknown": int(np.count_nonzero(evidence == EvidenceClass.UNKNOWN)),
        "free_confirmed": int(
            np.count_nonzero(evidence == EvidenceClass.FREE_CONFIRMED)
        ),
        "occupied_confirmed": int(
            np.count_nonzero(evidence == EvidenceClass.OCCUPIED_CONFIRMED)
        ),
        "ground_interpolated": int(
            np.count_nonzero(evidence == EvidenceClass.GROUND_INTERPOLATED)
        ),
    }
    manifest = {
        "schema_version": 1,
        "grid": metadata.to_dict(),
        "row_direction": [float(v) for v in direction],
        "aisle_count": len(aisles),
        "geometry_policy": {
            "include_interpolated": bool(include_interpolated),
            "min_longitudinal_support_ratio": float(
                min_longitudinal_support_ratio
            ),
            "min_width_m": float(min_width_m),
            "min_length_m": float(min_length_m),
            "promote_aisle_prior_to_static_free": False,
            "occupied_aisle_conflict_policy": "obstacle_candidate",
            "promote_candidates_in_aisles_to_static_free": True,
        },
        "evidence_counts": evidence_counts,
        "label_counts": {
            "free_confirmed": int(np.count_nonzero(labels == LABEL_AISLE)),
            "obstacle_candidate": int(
                np.count_nonzero(labels == LABEL_OBSTACLE_CANDIDATE)
            ),
            "occupied_confirmed": int(
                np.count_nonzero(labels == LABEL_OCCUPIED_CONFIRMED)
            ),
            "unknown_or_interpolated": int(np.count_nonzero(labels == 0)),
        },
        "aisle_conflict_candidate_count": int(aisle_conflict_candidates.sum()),
    }
    (output / "semantic_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    navigation = write_navigation_bundle(
        labels,
        aisles,
        output / "navigation",
        resolution=metadata.resolution,
        origin=(metadata.origin_x, metadata.origin_y, metadata.origin_yaw),
        clearance_radii_m=navigation_clearance_radii_m,
        promote_aisle_prior=False,
        promote_candidates_in_aisles=True,
    )
    return {
        "semantic_labels": labels,
        "corridor_seed": seed,
        "aisle_payload": aisle_payload,
        "manifest": manifest,
        "navigation": navigation,
    }
