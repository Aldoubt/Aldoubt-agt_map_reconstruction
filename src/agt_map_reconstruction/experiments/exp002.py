"""EXP002 agricultural corridor recovery experiment."""

import csv
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from agt_map_reconstruction.maps.corridor import (
    extract_corridor,
    extract_parallel_corridors,
    filter_row_aligned_components,
    restrict_to_row_extent,
    skeletonize_corridor,
)
from agt_map_reconstruction.maps.row_direction import estimate_row_direction
from agt_map_reconstruction.visualization.grid import save_grid


@dataclass(frozen=True)
class Exp002Config:
    resolution: float = 0.05
    kernel_size: int = 5
    chunk_size: int = 1_000_000
    row_height_threshold: float = 0.20
    min_width_m: float = 0.60
    max_width_m: float = 2.00
    min_length_m: float = 3.00
    min_row_coverage: float = 0.70
    min_row_profile: float = 0.25
    max_longitudinal_gap_m: float = 0.05
    max_cross_row_gap_m: float = 0.05
    max_boundary_gap_m: float = 0.30
    direction_threshold: float = 0.85
    min_aspect_ratio: float = 3.0
    baseline_min_cells: int = 5
    component_min_cells: int = 20

    def __post_init__(self):
        if self.resolution <= 0:
            raise ValueError("resolution must be positive")
        if self.kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 < self.min_width_m <= self.max_width_m:
            raise ValueError("corridor widths must satisfy 0 < min <= max")
        if self.min_length_m <= 0:
            raise ValueError("min_length_m must be positive")
        for name in (
            "min_row_coverage",
            "min_row_profile",
            "direction_threshold",
        ):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if (
            self.max_longitudinal_gap_m < 0
            or self.max_cross_row_gap_m < 0
            or self.max_boundary_gap_m < 0
        ):
            raise ValueError("gap tolerances must be non-negative")
        if self.min_aspect_ratio < 1:
            raise ValueError("min_aspect_ratio must be at least 1")
        if self.baseline_min_cells <= 0 or self.component_min_cells <= 0:
            raise ValueError("component cell thresholds must be positive")


@dataclass
class Exp002StageResult:
    corridor: np.ndarray
    centerline: np.ndarray
    metrics: dict
    details: dict


@dataclass
class Exp002Result:
    height: np.ndarray
    relative_height: np.ndarray
    traversability: np.ndarray
    row_angle_rad: float
    row_direction: np.ndarray
    stages: dict
    config: Exp002Config
    origin_xy: tuple


def build_run_id(commit_sha, instant=None):
    instant = instant or datetime.now(timezone.utc)
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=timezone.utc)
    instant = instant.astimezone(timezone.utc)
    short_commit = (commit_sha or "nogit")[:7]
    return f"{instant.strftime('%Y%m%dT%H%M%SZ')}_{short_commit}"


def sha256_file(path, chunk_size=1024 * 1024):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_run_directory(output_root, run_id):
    output_root = Path(output_root)
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be a single non-empty path component")
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / run_id
    run_dir.mkdir(exist_ok=False)
    return run_dir


def _stage_result(corridor, details=None):
    corridor = np.asarray(corridor, dtype=bool)
    centerline = np.asarray(skeletonize_corridor(corridor), dtype=bool)
    metrics = {
        "corridor_cells": int(corridor.sum()),
        "centerline_cells": int(centerline.sum()),
    }
    return Exp002StageResult(corridor, centerline, metrics, details or {})


def run_exp002_from_maps(height, relative_height, traversability, config=None,
                         origin_xy=(0.0, 0.0)):
    config = config or Exp002Config()
    height = np.asarray(height)
    relative_height = np.asarray(relative_height)
    traversability = np.asarray(traversability)
    if not (height.shape == relative_height.shape == traversability.shape):
        raise ValueError("height, relative_height and traversability must match")

    angle, direction = estimate_row_direction(
        relative_height,
        structure_threshold=config.row_height_threshold,
        component_min_cells=config.component_min_cells,
    )
    baseline = extract_corridor(
        traversability,
        min_width=config.baseline_min_cells,
    )
    min_length_cells = max(2, int(np.ceil(config.min_length_m / config.resolution)))
    from scipy import ndimage

    row_structure = (
        np.isfinite(relative_height)
        & (relative_height >= config.row_height_threshold)
    ) | (traversability == 2)
    max_cross_row_gap_cells = max(
        0,
        int(np.ceil(config.max_cross_row_gap_m / config.resolution)),
    )
    if max_cross_row_gap_cells:
        row_structure = ndimage.binary_closing(
            row_structure,
            structure=np.ones(
                (2 * max_cross_row_gap_cells + 1, 1),
                dtype=bool,
            ),
        )
    row_separated = restrict_to_row_extent(
        baseline & ~row_structure,
        row_structure,
        direction,
    )
    row_aware = filter_row_aligned_components(
        row_separated,
        direction,
        min_cells=config.component_min_cells,
        min_length_cells=min_length_cells,
        min_aspect_ratio=config.min_aspect_ratio,
        direction_threshold=config.direction_threshold,
        max_longitudinal_gap_cells=max(
            0,
            int(np.ceil(config.max_longitudinal_gap_m / config.resolution)),
        ),
    )
    constrained, constrained_details = extract_parallel_corridors(
        relative_height,
        traversability,
        direction,
        resolution=config.resolution,
        row_height_threshold=config.row_height_threshold,
        min_width_m=config.min_width_m,
        max_width_m=config.max_width_m,
        min_length_m=config.min_length_m,
        min_row_coverage=config.min_row_coverage,
        min_row_profile=config.min_row_profile,
        max_boundary_gap_m=config.max_boundary_gap_m,
    )
    stages = {
        "A": _stage_result(baseline, {"method": "connected_free_space"}),
        "B": _stage_result(
            row_aware,
            {
                "method": "componentwise_row_aligned_pca",
                "max_cross_row_gap_m": config.max_cross_row_gap_m,
                "max_longitudinal_gap_m": config.max_longitudinal_gap_m,
            },
        ),
        "C": _stage_result(constrained, constrained_details),
    }
    return Exp002Result(
        height=height,
        relative_height=relative_height,
        traversability=traversability,
        row_angle_rad=angle,
        row_direction=direction,
        stages=stages,
        config=config,
        origin_xy=(float(origin_xy[0]), float(origin_xy[1])),
    )


def _save_centerline_csv(centerline, path, origin_xy, resolution):
    y, x = np.where(centerline)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x_cell", "y_cell", "x_m", "y_m"])
        for x_cell, y_cell in zip(x.tolist(), y.tolist()):
            writer.writerow([
                x_cell,
                y_cell,
                origin_xy[0] + (x_cell + 0.5) * resolution,
                origin_xy[1] + (y_cell + 0.5) * resolution,
            ])


def _save_comparison(result, path):
    import matplotlib.pyplot as plt

    stage_count = len(result.stages)
    figure, axes = plt.subplots(1, stage_count, figsize=(5 * stage_count, 5))
    axes = np.atleast_1d(axes)
    for axis, (name, stage) in zip(axes, result.stages.items()):
        axis.imshow(stage.corridor, origin="lower", cmap="gray")
        axis.set_title(f"EXP002-{name}")
        axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_exp002_results(result, run_dir, metadata):
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)

    save_grid(result.height, run_dir / "height.png", "height")
    save_grid(
        result.relative_height,
        run_dir / "relative_height.png",
        "relative height",
    )
    save_grid(
        result.traversability,
        run_dir / "traversability.png",
        "traversability",
    )
    _save_comparison(result, run_dir / "abc_comparison.png")

    for name, stage in result.stages.items():
        stage_dir = run_dir / name
        stage_dir.mkdir()
        save_grid(stage.corridor, stage_dir / "corridor.png", f"EXP002-{name}")
        save_grid(
            stage.centerline,
            stage_dir / "centerline.png",
            f"EXP002-{name} centerline",
        )
        _save_centerline_csv(
            stage.centerline,
            stage_dir / "centerline.csv",
            result.origin_xy,
            result.config.resolution,
        )

    metadata_payload = dict(metadata)
    metadata_payload.update({
        "row_angle_rad": float(result.row_angle_rad),
        "row_direction": [float(value) for value in result.row_direction],
        "config": result.config.__dict__,
        "grid_origin_xy_m": list(result.origin_xy),
        "grid_resolution_m": result.config.resolution,
        "grid_shape_yx": list(result.height.shape),
    })
    metrics_payload = {
        name: {**stage.metrics, **stage.details}
        for name, stage in result.stages.items()
    }
    with (run_dir / "metadata.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata_payload, handle, sort_keys=False)
    with (run_dir / "metrics.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(metrics_payload, handle, sort_keys=False)
