"""Persist evidence-derived semantic geometry and a ready-to-load Nav2 map."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .aisle_reconstruction import recover_aisle_rectangles, write_aisle_bundle
from .ground_evidence import EvidenceClass
from .navigation_export import rasterize_aisles, write_navigation_bundle
from .row_band_classification import (
    classify_row_bands,
    write_row_band_classification_bundle,
)
from .semantic_reconstruction import (
    LABEL_AISLE,
    LABEL_OBSTACLE_CANDIDATE,
    LABEL_OCCUPIED_CONFIRMED,
    corridor_seed_from_evidence,
    refine_occupied_evidence_with_aisle_prior,
    semantic_labels_from_evidence,
)

AISLE_CONFLICT_POLICIES = ("hard", "candidate")


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
    occupied_aisle_conflict_policy="hard",
    wide_band_iqr_factor=1.50,
    navigation_clearance_radii_m=(0.20, 0.25, 0.30, 0.35, 0.40, 0.50),
):
    """Write evidence-derived semantics plus a Nav2 static-map bundle.

    Recovered row-aligned bands are first separated into ordinary ``row_aisle``
    regions and unusually wide ``wide_open_area_candidate`` regions using an
    upper-width IQR outlier rule. Width alone is intentionally not treated as
    proof of a headland; open-area candidates stay as a geometry-level class
    for later endpoint/topology validation.

    ``hard`` is the conservative obstacle default. ``candidate`` relaxes only
    confirmed occupied evidence that overlaps an accepted row aisle. Occupied
    evidence inside a wide open-area candidate is not relaxed by this policy.
    Unknown/interpolated evidence is never promoted by either policy.
    """
    if occupied_aisle_conflict_policy not in AISLE_CONFLICT_POLICIES:
        raise ValueError(
            "occupied_aisle_conflict_policy must be one of: "
            + ", ".join(AISLE_CONFLICT_POLICIES)
        )

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
    raw_row_bands = recover_aisle_rectangles(
        seed,
        direction,
        float(metadata.resolution),
        min_longitudinal_support_ratio=min_longitudinal_support_ratio,
        min_width_m=min_width_m,
        min_length_m=min_length_m,
    )
    row_band_classification = classify_row_bands(
        raw_row_bands,
        iqr_factor=wide_band_iqr_factor,
    )
    aisles = row_band_classification.row_aisles
    aisle_prior = rasterize_aisles(aisles, evidence.shape)

    if occupied_aisle_conflict_policy == "candidate":
        labels = refine_occupied_evidence_with_aisle_prior(raw_labels, aisle_prior)
        promote_candidates_in_aisles = True
    else:
        labels = raw_labels
        promote_candidates_in_aisles = False

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
    row_band_payload = write_row_band_classification_bundle(
        row_band_classification,
        metadata,
        output / "row_band_regions.json",
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
        "raw_row_band_count": len(raw_row_bands),
        "aisle_count": len(aisles),
        "open_area_candidate_count": len(
            row_band_classification.open_area_candidates
        ),
        "geometry_policy": {
            "include_interpolated": bool(include_interpolated),
            "min_longitudinal_support_ratio": float(
                min_longitudinal_support_ratio
            ),
            "min_width_m": float(min_width_m),
            "min_length_m": float(min_length_m),
            "wide_band_classification": "upper_width_outlier_q3_plus_iqr",
            "wide_band_iqr_factor": float(wide_band_iqr_factor),
            "wide_band_width_threshold_m": (
                None
                if row_band_classification.width_outlier_threshold_m is None
                else float(row_band_classification.width_outlier_threshold_m)
            ),
            "promote_aisle_prior_to_static_free": False,
            "occupied_aisle_conflict_policy": occupied_aisle_conflict_policy,
            "promote_candidates_in_aisles_to_static_free": bool(
                promote_candidates_in_aisles
            ),
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
        promote_candidates_in_aisles=promote_candidates_in_aisles,
    )
    return {
        "semantic_labels": labels,
        "corridor_seed": seed,
        "aisle_payload": aisle_payload,
        "row_band_payload": row_band_payload,
        "row_band_classification": row_band_classification,
        "manifest": manifest,
        "navigation": navigation,
    }
