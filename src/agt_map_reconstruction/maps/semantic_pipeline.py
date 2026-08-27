"""End-to-end PCD evidence reconstruction into semantic/Nav2 assets."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np

from .grid_geometry import GridMetadata
from .ground_evidence import EvidenceClass
from .semantic_assets import write_semantic_navigation_assets
from .semantic_reconstruction import corridor_seed_from_evidence


def metadata_from_statistics(statistics):
    """Convert robust raster statistics metadata into the common grid contract."""
    height, width = np.asarray(statistics.low_height).shape
    origin = np.asarray(statistics.origin_xy, dtype=float).reshape(-1)
    if origin.size != 2:
        raise ValueError("statistics origin_xy must contain exactly two values")
    return GridMetadata(
        resolution=float(statistics.resolution),
        origin_x=float(origin[0]),
        origin_y=float(origin[1]),
        width=int(width),
        height=int(height),
    )


def _pca_direction(x_cells, y_cells):
    points = np.column_stack((x_cells.astype(float), y_cells.astype(float)))
    centered = points - points.mean(axis=0)
    covariance = centered.T @ centered / len(points)
    values, vectors = np.linalg.eigh(covariance)
    direction = vectors[:, int(np.argmax(values))]
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ValueError("cannot infer a non-zero row direction from evidence")
    return direction / norm


def _projection_banding_score(x_cells, y_cells, angle_deg, bin_size_cells=2.0):
    """Measure how strongly occupied evidence collapses into parallel row bands."""
    angle = math.radians(float(angle_deg))
    cross_row = -math.sin(angle) * x_cells + math.cos(angle) * y_cells
    bins = np.floor((cross_row - cross_row.min()) / float(bin_size_cells)).astype(
        np.int64
    )
    counts = np.bincount(bins)
    return float(np.sum(counts.astype(np.float64) ** 2) / len(cross_row))


def _resolve_row_axis_from_occupied_banding(evidence, pca_direction):
    occupied_y, occupied_x = np.nonzero(
        np.asarray(evidence) == EvidenceClass.OCCUPIED_CONFIRMED
    )
    if len(occupied_x) < 32:
        return pca_direction

    # Bound the angular scan cost on large maps while preserving a deterministic
    # spatial sample of the occupied evidence.
    max_samples = 100_000
    if len(occupied_x) > max_samples:
        stride = int(np.ceil(len(occupied_x) / max_samples))
        occupied_x = occupied_x[::stride]
        occupied_y = occupied_y[::stride]

    base_angle = math.degrees(
        math.atan2(float(pca_direction[1]), float(pca_direction[0]))
    ) % 180.0
    centers = (base_angle, (base_angle + 90.0) % 180.0)
    offsets = np.arange(-15.0, 15.0 + 0.25, 0.5)

    best_angle = None
    best_score = -np.inf
    seen = set()
    for center in centers:
        for offset in offsets:
            angle = round((center + float(offset)) % 180.0, 6)
            if angle in seen:
                continue
            seen.add(angle)
            score = _projection_banding_score(
                occupied_x,
                occupied_y,
                angle,
                bin_size_cells=2.0,
            )
            if score > best_score:
                best_score = score
                best_angle = angle

    angle = math.radians(float(best_angle))
    return np.asarray([math.cos(angle), math.sin(angle)], dtype=float)


def infer_row_direction_from_evidence(evidence, include_interpolated=True):
    """Estimate crop-row direction while resolving the 90-degree PCA ambiguity.

    Confirmed/interpolated free support provides the scene's two dominant
    orthogonal axes. Confirmed occupied structure then selects the axis that
    produces the strongest parallel cross-row banding. This avoids treating
    the cross-row extent of many aisles as the crop-row direction.
    """
    seed = corridor_seed_from_evidence(
        evidence, include_interpolated=include_interpolated
    )
    yy, xx = np.nonzero(seed)
    if len(xx) < 2:
        raise ValueError("cannot infer row direction from fewer than two supported cells")

    pca_direction = _pca_direction(xx, yy)
    return _resolve_row_axis_from_occupied_banding(evidence, pca_direction)


def build_semantic_assets_from_points(
    points,
    output_dir,
    resolution=0.05,
    chunk_size=1_000_000,
    low_quantile=0.10,
    histogram_bins=64,
    ground_config=None,
    row_direction=None,
    min_longitudinal_support_ratio=0.50,
    min_width_m=0.30,
    min_length_m=2.0,
    include_interpolated=True,
    use_q90_for_obstacles=False,
    navigation_clearance_radii_m=(0.20, 0.25, 0.30, 0.35, 0.40, 0.50),
):
    """Build robust evidence, semantic geometry, and a Nav2 static-map bundle.

    The conservative default matches the original EXP003 policy: obstacle
    classification uses the low-envelope height statistic. ``q90_height`` is
    only used when ``use_q90_for_obstacles`` is explicitly enabled for a
    diagnostic comparison. In dense vegetation, q90 commonly represents
    canopy/leaf returns above otherwise traversable ground and is therefore too
    aggressive as a default static-obstacle statistic.
    """
    from .elevation_statistics import points_to_elevation_statistics
    from .ground_evidence import (
        GroundEvidenceConfig,
        build_ground_evidence_details,
    )

    statistics = points_to_elevation_statistics(
        points,
        resolution=resolution,
        chunk_size=chunk_size,
        low_quantile=low_quantile,
        histogram_bins=histogram_bins,
    )
    if ground_config is None:
        ground_config = GroundEvidenceConfig(resolution=float(resolution))
    if abs(float(ground_config.resolution) - float(resolution)) > 1e-12:
        raise ValueError("ground_config resolution must match raster resolution")

    q90_height = statistics.q90_height if use_q90_for_obstacles else None
    evidence_details = build_ground_evidence_details(
        statistics.low_height,
        statistics.point_count,
        ground_config,
        q90_height=q90_height,
    )
    metadata = metadata_from_statistics(statistics)
    if row_direction is None:
        direction = infer_row_direction_from_evidence(
            evidence_details.evidence,
            include_interpolated=include_interpolated,
        )
        row_direction_source = "evidence_pca_occupied_banding"
    else:
        direction = np.asarray(row_direction, dtype=float)
        row_direction_source = "explicit"

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "low_height.npy", statistics.low_height)
    np.save(output / "q90_height.npy", statistics.q90_height)
    np.save(output / "point_count.npy", statistics.point_count)
    np.save(output / "ground_surface.npy", evidence_details.ground_surface)
    np.save(output / "clearance.npy", evidence_details.clearance)

    bundle = write_semantic_navigation_assets(
        evidence=evidence_details.evidence,
        metadata=metadata,
        row_direction=direction,
        output_dir=output,
        min_longitudinal_support_ratio=min_longitudinal_support_ratio,
        min_width_m=min_width_m,
        min_length_m=min_length_m,
        include_interpolated=include_interpolated,
        navigation_clearance_radii_m=navigation_clearance_radii_m,
    )
    pipeline_manifest = {
        "schema_version": 1,
        "grid": metadata.to_dict(),
        "rasterization": {
            "chunk_size": int(chunk_size),
            "low_quantile": float(low_quantile),
            "histogram_bins": int(histogram_bins),
        },
        "ground_evidence_config": asdict(ground_config),
        "obstacle_height_source": "q90_height" if use_q90_for_obstacles else "low_height",
        "use_q90_for_obstacles": bool(use_q90_for_obstacles),
        "row_direction_source": row_direction_source,
        "row_direction": bundle["manifest"]["row_direction"],
    }
    (output / "pipeline_manifest.json").write_text(
        json.dumps(pipeline_manifest, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return {
        "statistics": statistics,
        "evidence_details": evidence_details,
        "metadata": metadata,
        "bundle": bundle,
        "pipeline_manifest": pipeline_manifest,
    }
