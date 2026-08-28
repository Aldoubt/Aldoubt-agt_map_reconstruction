#!/usr/bin/env python3
"""Evaluate frozen ground/scan/ray evidence by finite headland depth band."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Reuse frozen ground, unique-scan, and ray-support grids inside finite "
            "headland depth bands. No rosbag replay or ray regeneration occurs."
        )
    )
    parser.add_argument("--depth-profile", required=True)
    parser.add_argument("--ground-reference", required=True)
    parser.add_argument("--scan-support-count", required=True)
    parser.add_argument("--ray-support-count")
    parser.add_argument("--min-repeated-scans", type=int, default=2)
    parser.add_argument("--output", required=True)
    return parser


def _read_pgm(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"failed to read PGM: {path}")
    return np.flipud(image).astype(np.uint8, copy=False)


def _load_masks(payload, profile_path):
    root = profile_path.parent
    files = dict(payload.get("mask_files") or {})
    if not files:
        raise ValueError("depth profile must contain mask_files")
    return {
        key: np.load(root / filename, allow_pickle=False).astype(bool, copy=False)
        for key, filename in files.items()
    }


def _metric_values(bands, key):
    values = []
    for band in bands:
        value = band.get(key)
        values.append(np.nan if value is None else float(value))
    return values


def _plot_depth_response(result, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [
        ("ground_reference_ceiling_fraction_of_unknown", "trusted-ground ceiling"),
        ("scan_observed_fraction_of_unknown", "scan observed"),
        ("repeated_scan_fraction_of_unknown", "repeated scan"),
        ("ray_supported_fraction_of_unknown", "ray supported"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, side in zip(axes, ("entry", "exit")):
        bands = result[side]["bands"]
        x = [float(item["depth_midpoint_m"]) for item in bands]
        for key, label in metrics:
            ax.plot(x, _metric_values(bands, key), marker="o", label=label)
        ax.set_title(f"{side} headland evidence")
        ax.set_xlabel("outward depth from structural uncertainty edge (m)")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("fraction of UNKNOWN cells")
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle("Finite headland depth response — evaluation only")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    args = build_parser().parse_args()

    from agt_map_reconstruction.maps.headland_depth_evidence import (
        evaluate_headland_depth_evidence,
    )

    profile_path = Path(args.depth_profile).expanduser().resolve()
    ground_path = Path(args.ground_reference).expanduser().resolve()
    scan_path = Path(args.scan_support_count).expanduser().resolve()
    ray_path = (
        None
        if args.ray_support_count is None
        else Path(args.ray_support_count).expanduser().resolve()
    )
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    map_text = (payload.get("sources") or {}).get("map")
    if not map_text:
        raise ValueError("depth profile must preserve sources.map")
    map_path = Path(map_text).expanduser().resolve()
    base = _read_pgm(map_path)
    ground = np.load(ground_path, allow_pickle=False)
    scan = np.load(scan_path, allow_pickle=False)
    ray = None if ray_path is None else np.load(ray_path, allow_pickle=False)
    masks = _load_masks(payload, profile_path)

    result = evaluate_headland_depth_evidence(
        base,
        ground,
        scan,
        payload,
        masks,
        min_repeated_scans=int(args.min_repeated_scans),
        ray_support_count=ray,
    )
    result["sources"] = {
        "depth_profile": str(profile_path),
        "map": str(map_path),
        "ground_reference": str(ground_path),
        "scan_support_count": str(scan_path),
        "ray_support_count": None if ray_path is None else str(ray_path),
    }
    (output / "headland_depth_evidence.json").write_text(
        json.dumps(result, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    _plot_depth_response(result, output / "headland_depth_evidence.png")

    print("output:", output)
    print("method:", result["method"])
    print("min_repeated_scans:", result["min_repeated_scans"])
    for side in ("entry", "exit"):
        for band in result[side]["bands"]:
            print(
                f"{side} {band['depth_min_m']:.3f}-{band['depth_max_m']:.3f}m: "
                f"roi={band['roi_cell_count']} unknown={band['unknown_cell_count']} "
                f"ground_ceiling={band['ground_reference_ceiling_fraction_of_unknown']} "
                f"scan_observed={band['scan_observed_fraction_of_unknown']} "
                f"repeated_scan={band['repeated_scan_fraction_of_unknown']} "
                f"ray_supported={band['ray_supported_fraction_of_unknown']}"
            )
    print("frozen_evidence_reused: true")
    print("rosbag_replay_performed: false")
    print("ray_evidence_regenerated: false")
    print("physical_site_boundary_required: false")
    print("automatic_acceptance: false")
    print("navigation_map_modified: false")
    print("semantic_promotion: false")


if __name__ == "__main__":
    main()
