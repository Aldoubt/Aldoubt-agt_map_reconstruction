#!/usr/bin/env python3
"""Build explicit conservative headland navigation masks from frozen evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Build a trusted headland free-space mask from finite depth bands, "
            "K8/K16 ground-reference confidence, and scan/ray support. Thresholds "
            "are explicit inputs; no automatic selection is performed."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--map", help="navigation_base_map.pgm")
    source.add_argument("--map-npy", help="2D uint8 map array; mainly for tests")
    parser.add_argument("--depth-profile", required=True)
    parser.add_argument("--reference-a", required=True)
    parser.add_argument("--reference-b", required=True)
    parser.add_argument("--scan-support-count", required=True)
    parser.add_argument("--ray-support-count")
    parser.add_argument("--entry-max-depth-m", type=float, required=True)
    parser.add_argument("--exit-max-depth-m", type=float, required=True)
    parser.add_argument("--max-support-distance-m", type=float, required=True)
    parser.add_argument("--max-model-disagreement-m", type=float, required=True)
    parser.add_argument("--min-scan-support", type=int, default=1)
    parser.add_argument("--min-ray-support", type=int, default=0)
    parser.add_argument("--output", required=True)
    return parser


def _read_map(args):
    if args.map_npy:
        path = Path(args.map_npy).expanduser().resolve()
        array = np.load(path, allow_pickle=False).astype(np.uint8, copy=False)
        if array.ndim != 2:
            raise ValueError("--map-npy must contain a 2D array")
        return path, array
    path = Path(args.map).expanduser().resolve()
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"failed to read map: {path}")
    return path, np.flipud(image).astype(np.uint8, copy=False)


def _load_masks(payload, profile_path):
    files = dict(payload.get("mask_files") or {})
    if not files:
        raise ValueError("depth profile must contain mask_files")
    root = profile_path.parent
    return {
        key: np.load(root / filename, allow_pickle=False).astype(bool, copy=False)
        for key, filename in files.items()
    }


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

    from agt_map_reconstruction.maps.headland_navigation_gate import (
        build_headland_navigation_gate,
    )

    map_path, base = _read_map(args)
    profile_path = Path(args.depth_profile).expanduser().resolve()
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    masks = _load_masks(profile, profile_path)

    ref_a_dir, ref_a, distance_a = _load_reference_dir(args.reference_a)
    ref_b_dir, ref_b, distance_b = _load_reference_dir(args.reference_b)
    if ref_a.shape != base.shape or ref_b.shape != base.shape:
        raise ValueError("ground reference grids must match map shape")
    if distance_a.shape != base.shape or distance_b.shape != base.shape:
        raise ValueError("support-distance grids must match map shape")
    if not np.allclose(distance_a, distance_b, equal_nan=True, rtol=0.0, atol=1e-7):
        raise ValueError("reference directories contain inconsistent support distances")

    scans_path = Path(args.scan_support_count).expanduser().resolve()
    scans = np.load(scans_path, allow_pickle=False)
    ray_path = None
    rays = None
    if args.ray_support_count:
        ray_path = Path(args.ray_support_count).expanduser().resolve()
        rays = np.load(ray_path, allow_pickle=False)

    disagreement = np.abs(ref_a.astype(np.float64) - ref_b.astype(np.float64))
    result, trusted, uncertainty = build_headland_navigation_gate(
        base,
        profile,
        masks,
        distance_a,
        disagreement,
        scans,
        entry_max_depth_m=args.entry_max_depth_m,
        exit_max_depth_m=args.exit_max_depth_m,
        max_support_distance_m=args.max_support_distance_m,
        max_model_disagreement_m=args.max_model_disagreement_m,
        min_scan_support=args.min_scan_support,
        ray_support_count=rays,
        min_ray_support=args.min_ray_support,
    )

    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    np.save(output / "trusted_headland_free_mask.npy", trusted.astype(np.uint8))
    np.save(
        output / "headland_navigation_uncertainty_mask.npy",
        uncertainty.astype(np.uint8),
    )
    result["sources"] = {
        "map": str(map_path),
        "depth_profile": str(profile_path),
        "reference_a": str(ref_a_dir),
        "reference_b": str(ref_b_dir),
        "scan_support_count": str(scans_path),
        "ray_support_count": None if ray_path is None else str(ray_path),
    }
    result["outputs"] = {
        "trusted_free_mask": "trusted_headland_free_mask.npy",
        "uncertainty_mask": "headland_navigation_uncertainty_mask.npy",
    }
    (output / "headland_navigation_gate.json").write_text(
        json.dumps(result, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    print("output:", output)
    print("method:", result["method"])
    print("trusted_free_cells:", result["trusted_free_cell_count"])
    print("uncertainty_cells:", result["uncertainty_cell_count"])
    print("entry_max_depth_m:", result["entry"]["max_depth_m"])
    print("exit_max_depth_m:", result["exit"]["max_depth_m"])
    print("automatic_threshold_selection: false")
    print("navigation_map_modified: false")


if __name__ == "__main__":
    main()
