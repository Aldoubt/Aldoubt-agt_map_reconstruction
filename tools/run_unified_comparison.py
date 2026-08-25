#!/usr/bin/env python3
"""Compare tiled PMF, EXP003 evidence, and MK-mini envelope feasibility."""

import argparse
import json
from pathlib import Path

import numpy as np

from agt_map_reconstruction.algorithms.morphological_pmf import PMFConfig, progressive_morphological_filter
from agt_map_reconstruction.io.pcd_loader import load_pcd
from agt_map_reconstruction.maps.navigation_evaluation import evaluate_navigation_method, pmf_grids_to_evidence
from agt_map_reconstruction.maps.vehicle_envelope import VehicleEnvelopeConfig
from agt_map_reconstruction.visualization.compare import save_segmentation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcd", type=Path, required=True)
    parser.add_argument("--exp003-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--max-window-m", type=float, default=1.0)
    parser.add_argument("--safety-margin-m", type=float, default=0.0)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    envelope = VehicleEnvelopeConfig(resolution=args.resolution, safety_margin_m=args.safety_margin_m)
    points = load_pcd(args.pcd)
    pmf = progressive_morphological_filter(
        points,
        PMFConfig(resolution=args.resolution, tile_size=args.tile_size, max_window_m=args.max_window_m),
    )
    pmf_evidence = pmf_grids_to_evidence(pmf["ground_grid"], pmf["observed_grid"])
    exp_evidence = np.load(args.exp003_run / "evidence.npy")
    if exp_evidence.shape != pmf_evidence.shape:
        raise ValueError("EXP003 and PMF grids have different shapes; use the same resolution/input bounds")
    rows = []
    for name, evidence in (("pmf", pmf_evidence), ("exp003", exp_evidence)):
        metrics, layers = evaluate_navigation_method(name, evidence, envelope)
        rows.append(metrics)
        method_dir = args.output / name
        method_dir.mkdir(parents=True, exist_ok=True)
        if name == "pmf":
            save_segmentation(
                {"ground": pmf["ground"], "non_ground": pmf["non_ground"]},
                method_dir,
            )
        np.save(method_dir / "evidence.npy", evidence, allow_pickle=False)
        np.save(method_dir / "vehicle_free.npy", layers["vehicle_free"], allow_pickle=False)
        np.save(method_dir / "aisle_candidate.npy", layers["aisle_candidate"], allow_pickle=False)
    np.save(args.output / "pmf_ground_surface.npy", pmf["ground_surface"], allow_pickle=False)
    ground_height = np.full(pmf["ground_surface"].shape, np.nan, dtype=np.float32)
    indices = np.floor((pmf["ground"][:, :2] - pmf["origin_xy"]) / args.resolution).astype(int)
    valid = (
        (indices[:, 0] >= 0) & (indices[:, 1] >= 0)
        & (indices[:, 0] < ground_height.shape[1])
        & (indices[:, 1] < ground_height.shape[0])
    )
    flat = ground_height.ravel()
    flat_indices = indices[valid, 1] * ground_height.shape[1] + indices[valid, 0]
    values = pmf["ground"][valid, 2].astype(np.float32)
    flat[flat_indices] = np.fmax(flat[flat_indices], values)
    np.save(args.output / "pmf_ground_height.npy", ground_height, allow_pickle=False)
    (args.output / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
