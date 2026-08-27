#!/usr/bin/env python3
"""Audit local ground-reference confidence inside the frozen P1-D3 endpoint ROIs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compare KNN local ground references only inside the frozen P1-D3 "
            "entry/exit endpoint ROIs. No model is auto-selected and no semantic "
            "cell is promoted."
        )
    )
    parser.add_argument("--map", required=True, help="navigation_base_map.pgm")
    parser.add_argument(
        "--endpoint-envelope",
        required=True,
        help="frozen P1-D3 headland_endpoint_envelope.json",
    )
    parser.add_argument(
        "--reference-dir",
        action="append",
        required=True,
        help=(
            "local-ground-reference output directory; repeat for K=8/16/32/64"
        ),
    )
    parser.add_argument("--output", required=True)
    return parser


def _read_grid_pgm(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"failed to read PGM: {path}")
    return np.flipud(image).astype(np.uint8, copy=False)


def _fmt(value):
    if value is None:
        return "None"
    return f"{float(value):.6f}"


def _load_reference(directory, expected_shape):
    directory = Path(directory).expanduser().resolve()
    manifest_path = directory / "ground_reference_manifest.json"
    reference_path = directory / "ground_reference.npy"
    nearest_path = directory / "ground_reference_nearest_support_distance.npy"
    valid_path = directory / "ground_reference_valid_mask.npy"
    for path in (manifest_path, reference_path, nearest_path, valid_path):
        if not path.exists():
            raise FileNotFoundError(path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model = manifest.get("model")
    if not isinstance(model, dict) or model.get("model_type") != "knn_local_affine":
        raise ValueError(f"unsupported ground reference manifest: {manifest_path}")
    neighbors = int(model["neighbor_count"])
    name = f"k{neighbors}"

    reference = np.load(reference_path, allow_pickle=False)
    nearest = np.load(nearest_path, allow_pickle=False)
    valid = np.load(valid_path, allow_pickle=False).astype(bool)
    for label, array in (
        ("ground_reference", reference),
        ("nearest_support_distance", nearest),
        ("valid_mask", valid),
    ):
        if array.shape != expected_shape:
            raise ValueError(
                f"{directory} {label} shape {array.shape} != map shape {expected_shape}"
            )
    return name, {
        "reference": reference,
        "nearest_support_distance_m": nearest,
        "valid_mask": valid,
        "neighbor_count": neighbors,
        "cv_residual_rmse_m": model.get("cv_residual_rmse_m"),
        "cv_residual_p95_abs_m": model.get("cv_residual_p95_abs_m"),
        "source_directory": str(directory),
    }


def main():
    args = build_parser().parse_args()

    from agt_map_reconstruction.maps.endpoint_ground_reference_confidence import (
        audit_endpoint_ground_reference_confidence,
    )

    map_path = Path(args.map).expanduser().resolve()
    endpoint_path = Path(args.endpoint_envelope).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    base = _read_grid_pgm(map_path)
    endpoint = json.loads(endpoint_path.read_text(encoding="utf-8"))

    models = {}
    canonical_nearest = None
    source_directories = {}
    for directory in args.reference_dir:
        name, payload = _load_reference(directory, base.shape)
        if name in models:
            raise ValueError(f"duplicate reference model: {name}")
        models[name] = payload
        source_directories[name] = payload.pop("source_directory")
        if canonical_nearest is None:
            canonical_nearest = payload["nearest_support_distance_m"]

    if len(models) < 2:
        raise ValueError("at least two --reference-dir inputs are required")

    result = audit_endpoint_ground_reference_confidence(
        base,
        endpoint,
        models,
        canonical_nearest,
    )
    result.update({
        "source_map": str(map_path),
        "source_endpoint_envelope": str(endpoint_path),
        "source_reference_directories": source_directories,
    })

    json_path = output / "endpoint_ground_reference_confidence.json"
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    print("output:", output)
    print("models:", result["model_names"])
    for side_name in ("entry", "exit"):
        side = result["sides"][side_name]
        cross = side["cross_model_disagreement"]
        print(
            f"{side_name}: "
            f"unknown_cells={side['unknown_cell_count']} "
            f"support_distance_median_m={_fmt(side['nearest_support_distance_median_m'])} "
            f"support_distance_p95_m={_fmt(side['nearest_support_distance_p95_m'])} "
            f"cross_model_range_median_m={_fmt(cross['range_median_m'])} "
            f"cross_model_range_p95_m={_fmt(cross['range_p95_m'])} "
            f"common_valid_fraction={_fmt(cross['common_valid_unknown_fraction'])}"
        )
        pair = side["pairwise_abs_difference"].get("k8__k16")
        if pair is not None:
            print(
                f"{side_name} k8__k16: "
                f"median_m={_fmt(pair['median_m'])} "
                f"p95_m={_fmt(pair['p95_m'])} "
                f"max_m={_fmt(pair['max_m'])}"
            )
        for model_name in result["model_names"]:
            model = side["models"][model_name]
            print(
                f"{side_name} {model_name}: "
                f"valid_unknown_fraction={_fmt(model['valid_unknown_fraction'])} "
                f"cv_rmse_m={_fmt(model['cv_residual_rmse_m'])} "
                f"cv_p95_m={_fmt(model['cv_residual_p95_abs_m'])}"
            )
    print("automatic_model_selection: false")
    print("semantic_promotion: false")


if __name__ == "__main__":
    main()
