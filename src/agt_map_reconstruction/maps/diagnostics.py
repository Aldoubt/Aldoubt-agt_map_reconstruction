"""Small, numeric diagnostics for existing EXP003 raster artifacts."""

import json

import numpy as np
from scipy import ndimage

from .ground_evidence import EvidenceClass


def percentile_summary(array):
    """Return robust percentiles for finite values in a raster."""
    values = np.asarray(array, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {name: None for name in ("p1", "p5", "p50", "p95", "p99")}
    percentiles = np.percentile(finite, [1, 5, 50, 95, 99])
    return dict(zip(("p1", "p5", "p50", "p95", "p99"), map(float, percentiles)))


def evidence_counts(evidence):
    values = np.asarray(evidence)
    return {label.name.lower(): int((values == label).sum()) for label in EvidenceClass}


def _largest_component(mask):
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        return 0
    return int(np.bincount(labels.ravel())[1:].max())


def scan_inflation(evidence, resolution, radii_m):
    """Measure free-space loss and largest free component for each radius."""
    if resolution <= 0 or not np.isfinite(resolution):
        raise ValueError("resolution must be finite and positive")
    values = np.asarray(evidence)
    occupied = values == EvidenceClass.OCCUPIED_CONFIRMED
    free = values == EvidenceClass.FREE_CONFIRMED
    rows = []
    for radius in radii_m:
        if radius < 0 or not np.isfinite(radius):
            raise ValueError("radii must be finite and non-negative")
        distance = ndimage.distance_transform_edt(~occupied, sampling=resolution)
        inflated = distance <= radius
        remaining = free & ~inflated
        rows.append({
            "radius_m": float(radius),
            "free_cells": int(remaining.sum()),
            "free_fraction_of_measured": float(remaining.sum() / max(1, free.sum())),
            "largest_free_component": _largest_component(remaining),
        })
    return rows


def summarize_run(run_dir, radii_m=(0, .05, .10, .15, .20, .25, .40)):
    """Load an immutable run and return JSON-serializable diagnostic data."""
    from pathlib import Path
    run_dir = Path(run_dir)
    low = np.load(run_dir / "low_height.npy")
    ground = np.load(run_dir / "ground_surface.npy")
    clearance = np.load(run_dir / "clearance.npy")
    evidence = np.load(run_dir / "evidence.npy")
    metadata = {}
    metadata_path = run_dir / "metadata.yaml"
    if metadata_path.exists():
        import yaml
        metadata = yaml.safe_load(metadata_path.read_text()) or {}
    resolution = float(metadata.get("grid_resolution_m", 1.0))
    finite_clearance = clearance[np.isfinite(clearance)]
    negative = finite_clearance < -0.05
    return {
        "run_dir": str(run_dir),
        "low_height": percentile_summary(low),
        "ground_surface": percentile_summary(ground),
        "clearance": percentile_summary(clearance),
        "negative_clearance_below_-0.05m": {
            "cells": int(negative.sum()),
            "fraction_of_finite": float(negative.sum() / max(1, finite_clearance.size)),
        },
        "evidence_counts": evidence_counts(evidence),
        "inflation_scan": scan_inflation(evidence, resolution, radii_m),
    }


def write_summary(summary, path):
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def save_discrete_previews(run_dir, output_dir):
    """Write categorical previews that do not merge unknown and occupied."""
    from pathlib import Path
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    run_dir, output_dir = Path(run_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence = np.load(run_dir / "evidence.npy")
    costmap = np.load(run_dir / "costmap.npy")
    evidence_view = np.where(costmap == 254, EvidenceClass.OCCUPIED_CONFIRMED, evidence)
    palette = ListedColormap(["black", "white", "#d1495b", "#f3a712"])
    for name, array in (("evidence_discrete", evidence), ("costmap_discrete", evidence_view)):
        figure, axis = plt.subplots(figsize=(8, 8))
        axis.imshow(array, origin="lower", cmap=palette, vmin=0, vmax=3, interpolation="nearest")
        axis.set_title(name.replace("_", " "))
        figure.savefig(output_dir / f"{name}.png", dpi=180, bbox_inches="tight")
        plt.close(figure)
