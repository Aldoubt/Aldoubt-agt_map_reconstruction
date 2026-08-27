#!/usr/bin/env python3
"""Build a confidence-gated consensus ground reference from two local models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Combine two local ground references only where they agree and remain "
            "close to observed ground support. Rejected cells stay NaN."
        )
    )
    parser.add_argument("--reference-a", required=True, help="local ground reference directory")
    parser.add_argument("--reference-b", required=True, help="local ground reference directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-support-distance-m", type=float, required=True)
    parser.add_argument("--max-model-disagreement-m", type=float, required=True)
    return parser


def _load_reference_dir(path):
    directory = Path(path).expanduser().resolve()
    reference = np.load(directory / "ground_reference.npy", allow_pickle=False)
    distance = np.load(
        directory / "ground_reference_nearest_support_distance.npy",
        allow_pickle=False,
    )
    manifest = json.loads(
        (directory / "ground_reference_manifest.json").read_text(encoding="utf-8")
    )
    return directory, reference, distance, manifest


def main():
    args = build_parser().parse_args()

    from agt_map_reconstruction.maps.ground_reference_consensus import (
        build_ground_reference_consensus,
    )

    a_dir, a, a_distance, a_manifest = _load_reference_dir(args.reference_a)
    b_dir, b, b_distance, b_manifest = _load_reference_dir(args.reference_b)
    if a.shape != b.shape:
        raise ValueError("reference grids must have matching shapes")
    if a_distance.shape != a.shape or b_distance.shape != a.shape:
        raise ValueError("nearest-support distance grids must match reference shape")
    if not np.allclose(a_distance, b_distance, equal_nan=True, rtol=0.0, atol=1e-7):
        raise ValueError("reference directories contain inconsistent nearest-support distance grids")

    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    result = build_ground_reference_consensus(
        a,
        b,
        a_distance,
        max_support_distance_m=args.max_support_distance_m,
        max_model_disagreement_m=args.max_model_disagreement_m,
    )

    np.save(output / "ground_reference.npy", result["ground_reference"])
    np.save(
        output / "ground_reference_confidence_mask.npy",
        result["confidence_mask"].astype(np.uint8),
    )
    np.save(
        output / "ground_reference_model_disagreement.npy",
        result["model_disagreement_m"],
    )
    np.save(
        output / "ground_reference_nearest_support_distance.npy",
        a_distance.astype(np.float32),
    )

    manifest = {
        "schema_version": 1,
        "source_reference_a": str(a_dir),
        "source_reference_b": str(b_dir),
        "source_neighbor_count_a": a_manifest.get("model", {}).get("neighbor_count"),
        "source_neighbor_count_b": b_manifest.get("model", {}).get("neighbor_count"),
        "summary": result["summary"],
        "policy": {
            "automatic_model_selection": False,
            "ground_reference_is_semantic_evidence": False,
            "semantic_promotion": False,
        },
    }
    (output / "ground_reference_consensus_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    summary = result["summary"]
    print("output:", output)
    print("reference_a_neighbors:", manifest["source_neighbor_count_a"])
    print("reference_b_neighbors:", manifest["source_neighbor_count_b"])
    print("max_support_distance_m:", summary["max_support_distance_m"])
    print("max_model_disagreement_m:", summary["max_model_disagreement_m"])
    print("accepted_cells:", summary["accepted_cell_count"])
    print("accepted_fraction_of_finite:", f"{summary['accepted_fraction_of_finite']:.6f}")
    print("rejected_distance_cells:", summary["rejected_distance_cell_count"])
    print("rejected_disagreement_cells:", summary["rejected_disagreement_cell_count"])
    print("semantic_promotion: false")


if __name__ == "__main__":
    main()
