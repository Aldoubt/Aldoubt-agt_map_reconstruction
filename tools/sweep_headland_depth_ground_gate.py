#!/usr/bin/env python3
"""Sweep K8/K16 ground-confidence gates across finite headland depth bands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Reuse finite headland depth masks and frozen K8/K16 confidence grids "
            "to measure accepted UNKNOWN fraction by outward depth. No threshold "
            "is selected automatically."
        )
    )
    parser.add_argument("--map", required=True)
    parser.add_argument("--depth-profile", required=True)
    parser.add_argument("--reference-a", required=True)
    parser.add_argument("--reference-b", required=True)
    parser.add_argument("--max-support-distance-m", type=float, nargs="+", required=True)
    parser.add_argument("--max-model-disagreement-m", type=float, nargs="+", required=True)
    parser.add_argument("--output", required=True)
    return parser


def _read_pgm(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"failed to read PGM: {path}")
    return np.flipud(image).astype(np.uint8, copy=False)


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


def _plot_sweep(result, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, side in zip(axes, ("entry", "exit")):
        bands = result[side]["bands"]
        x = [float(item["depth_midpoint_m"]) for item in bands]
        if bands:
            gate_pairs = [
                (
                    float(item["max_support_distance_m"]),
                    float(item["max_model_disagreement_m"]),
                )
                for item in bands[0]["grid"]
            ]
            for max_distance, max_disagreement in gate_pairs:
                y = []
                for band in bands:
                    record = next(
                        item
                        for item in band["grid"]
                        if np.isclose(item["max_support_distance_m"], max_distance)
                        and np.isclose(item["max_model_disagreement_m"], max_disagreement)
                    )
                    y.append(float(record["accepted_unknown_fraction"]))
                ax.plot(
                    x,
                    y,
                    marker="o",
                    linewidth=1.0,
                    label=f"d≤{max_distance:g}m, Δ≤{max_disagreement:g}m",
                )
        ax.set_title(f"{side} ground-gate sensitivity")
        ax.set_xlabel("outward depth from structural uncertainty edge (m)")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("accepted UNKNOWN fraction")
    if result["exit"]["bands"]:
        axes[1].legend(loc="best", fontsize=6)
    elif result["entry"]["bands"]:
        axes[0].legend(loc="best", fontsize=6)
    fig.suptitle("Finite headland ground-reference gate sensitivity — no automatic selection")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    args = build_parser().parse_args()

    from agt_map_reconstruction.maps.headland_depth_ground_gate_sweep import (
        sweep_headland_depth_ground_gate,
    )
    from agt_map_reconstruction.maps.navigation_export import UNKNOWN_VALUE

    map_path = Path(args.map).expanduser().resolve()
    profile_path = Path(args.depth_profile).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    base = _read_pgm(map_path)
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    profile_sources = dict(payload.get("sources") or {})
    profile_map_text = profile_sources.get("map")
    if profile_map_text and Path(profile_map_text).expanduser().resolve() != map_path:
        raise ValueError("--map differs from depth profile sources.map")
    masks = _load_masks(payload, profile_path)
    a_dir, a, a_distance = _load_reference_dir(args.reference_a)
    b_dir, b, b_distance = _load_reference_dir(args.reference_b)
    if a.shape != base.shape or b.shape != base.shape:
        raise ValueError("ground reference grids must match map shape")
    if a_distance.shape != base.shape or b_distance.shape != base.shape:
        raise ValueError("support-distance grids must match map shape")
    if not np.allclose(a_distance, b_distance, equal_nan=True, rtol=0.0, atol=1e-7):
        raise ValueError("reference directories contain inconsistent nearest-support distance grids")

    disagreement = np.abs(a.astype(np.float64) - b.astype(np.float64))
    result = sweep_headland_depth_ground_gate(
        base == UNKNOWN_VALUE,
        payload,
        masks,
        a_distance,
        disagreement,
        max_support_distances_m=args.max_support_distance_m,
        max_model_disagreements_m=args.max_model_disagreement_m,
    )
    result["sources"] = {
        "map": str(map_path),
        "depth_profile": str(profile_path),
        "fused_structural_bundle": profile_sources.get("fused_structural_bundle"),
        "fused_uncertainty": profile_sources.get("fused_uncertainty"),
        "row_lattice_completion": profile_sources.get("row_lattice_completion"),
        "source_structural_bundle": profile_sources.get("source_structural_bundle"),
        "targeted_3d_audit": profile_sources.get("targeted_3d_audit"),
        "reference_a": str(a_dir),
        "reference_b": str(b_dir),
    }
    (output / "headland_depth_ground_gate_sweep.json").write_text(
        json.dumps(result, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    _plot_sweep(result, output / "headland_depth_ground_gate_sweep.png")

    print("output:", output)
    print("method:", result["method"])
    for side in ("entry", "exit"):
        for band in result[side]["bands"]:
            print(
                f"{side} {band['depth_min_m']:.3f}-{band['depth_max_m']:.3f}m: "
                f"unknown_cells={band['unknown_cell_count']}"
            )
            for item in band["grid"]:
                print(
                    f"  d={item['max_support_distance_m']:.3f} "
                    f"a={item['max_model_disagreement_m']:.3f} "
                    f"accepted={item['accepted_unknown_fraction']:.6f}"
                )
    print("automatic_threshold_selection: false")
    print("physical_site_boundary_required: false")
    print("structural_geometry_modified: false")
    print("navigation_map_modified: false")
    print("semantic_promotion: false")


if __name__ == "__main__":
    main()
