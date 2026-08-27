#!/usr/bin/env python3
"""Sweep confidence gates inside the frozen P1-D3 entry/exit endpoint ROIs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Report how much D3 endpoint unknown space remains eligible as support-distance "
            "and local-model-disagreement gates are varied. No threshold is selected automatically."
        )
    )
    parser.add_argument("--map", required=True)
    parser.add_argument("--endpoint-envelope", required=True)
    parser.add_argument("--reference-a", required=True)
    parser.add_argument("--reference-b", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-support-distance-m", type=float, nargs="+", required=True)
    parser.add_argument("--max-model-disagreement-m", type=float, nargs="+", required=True)
    return parser


def _read_grid_pgm(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"failed to read PGM: {path}")
    return np.flipud(image).astype(np.uint8, copy=False)


def _load_reference_dir(path):
    directory = Path(path).expanduser().resolve()
    reference = np.load(directory / "ground_reference.npy", allow_pickle=False)
    distance = np.load(
        directory / "ground_reference_nearest_support_distance.npy",
        allow_pickle=False,
    )
    return directory, reference, distance


def main():
    args = build_parser().parse_args()

    from agt_map_reconstruction.maps.endpoint_ground_reference_gate_sweep import (
        sweep_endpoint_ground_reference_gate,
    )
    from agt_map_reconstruction.maps.navigation_export import UNKNOWN_VALUE

    map_path = Path(args.map).expanduser().resolve()
    envelope_path = Path(args.endpoint_envelope).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    base = _read_grid_pgm(map_path)
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    a_dir, a, a_distance = _load_reference_dir(args.reference_a)
    b_dir, b, b_distance = _load_reference_dir(args.reference_b)

    if a.shape != base.shape or b.shape != base.shape:
        raise ValueError("reference grids must match map shape")
    if not np.allclose(a_distance, b_distance, equal_nan=True, rtol=0.0, atol=1e-7):
        raise ValueError("reference directories contain inconsistent nearest-support distance grids")

    disagreement = np.abs(a.astype(np.float64) - b.astype(np.float64))
    result = sweep_endpoint_ground_reference_gate(
        base == UNKNOWN_VALUE,
        envelope,
        a_distance,
        disagreement,
        max_support_distances_m=args.max_support_distance_m,
        max_model_disagreements_m=args.max_model_disagreement_m,
    )
    result.update({
        "source_map": str(map_path),
        "source_endpoint_envelope": str(envelope_path),
        "source_reference_a": str(a_dir),
        "source_reference_b": str(b_dir),
    })

    json_path = output / "endpoint_ground_reference_gate_sweep.json"
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    print("output:", output)
    for side in ("entry", "exit"):
        payload = result["sides"][side]
        print(f"{side}: unknown_cells={payload['unknown_cell_count']}")
        for item in payload["grid"]:
            print(
                f"{side}: max_distance_m={item['max_support_distance_m']:.3f} "
                f"max_disagreement_m={item['max_model_disagreement_m']:.3f} "
                f"accepted_unknown_fraction={item['accepted_unknown_fraction']:.6f}"
            )
    print("automatic_threshold_selection: false")
    print("semantic_promotion: false")


if __name__ == "__main__":
    main()
