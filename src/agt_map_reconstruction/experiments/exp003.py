"""Reproducible EXP003 ground-evidence experiment orchestration."""

from collections.abc import Mapping
import ctypes
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import errno
import hashlib
from numbers import Integral, Real
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import yaml

from agt_map_reconstruction.maps.elevation_statistics import (
    points_to_elevation_statistics,
)
from agt_map_reconstruction.maps.ground_evidence import (
    EvidenceClass,
    GroundEvidenceConfig,
    build_ground_evidence_details,
    build_navigation_costmap,
)


@dataclass(frozen=True)
class Exp003Config:
    """All rasterization, ground-model, and navigation parameters."""

    resolution: float = 0.05
    chunk_size: int = 1_000_000
    low_quantile: float = 0.10
    histogram_bins: int = 64
    min_points_per_cell: int = 3
    min_ground_support_cells: int = 2
    ground_window_m: float = 0.50
    ground_percentile: float = 20.0
    ground_seed_percentile: float = 10.0
    max_ground_step_m: float = 0.20
    max_interpolation_gap_m: float = 0.25
    obstacle_height_m: float = 0.15
    obstacle_inflation_radius_m: float = 0.25
    interpolated_ground_cost: int = 64
    use_q90_for_obstacles: bool = False

    def __post_init__(self):
        integer_parameters = (
            "chunk_size",
            "histogram_bins",
            "min_points_per_cell",
            "min_ground_support_cells",
            "interpolated_ground_cost",
        )
        real_parameters = (
            "resolution",
            "low_quantile",
            "ground_window_m",
            "ground_percentile",
            "ground_seed_percentile",
            "max_ground_step_m",
            "max_interpolation_gap_m",
            "obstacle_height_m",
            "obstacle_inflation_radius_m",
        )
        for name in integer_parameters + real_parameters:
            if isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must not be boolean")
        if not isinstance(self.use_q90_for_obstacles, bool):
            raise ValueError("use_q90_for_obstacles must be boolean")
        if (
            not isinstance(self.resolution, Real)
            or not np.isfinite(self.resolution)
            or self.resolution <= 0
        ):
            raise ValueError("resolution must be finite and positive")
        for name in (
            "chunk_size",
            "histogram_bins",
            "min_points_per_cell",
            "min_ground_support_cells",
        ):
            value = getattr(self, name)
            if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not isinstance(self.low_quantile, Real)
            or not np.isfinite(self.low_quantile)
            or not 0.0 <= self.low_quantile <= 1.0
        ):
            raise ValueError("low_quantile must be finite and between 0 and 1")
        for name in ("ground_percentile", "ground_seed_percentile"):
            value = getattr(self, name)
            if (
                not isinstance(value, Real)
                or not np.isfinite(value)
                or not 0.0 <= value <= 100.0
            ):
                raise ValueError(f"{name} must be finite and between 0 and 100")
        for name in (
            "ground_window_m",
            "max_ground_step_m",
            "max_interpolation_gap_m",
            "obstacle_height_m",
            "obstacle_inflation_radius_m",
        ):
            value = getattr(self, name)
            if not isinstance(value, Real) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.ground_window_m == 0:
            raise ValueError("ground_window_m must be positive")
        if (
            not isinstance(self.interpolated_ground_cost, Integral)
            or isinstance(self.interpolated_ground_cost, bool)
            or not 1 <= self.interpolated_ground_cost <= 253
        ):
            raise ValueError("interpolated_ground_cost must be between 1 and 253")
        for name in integer_parameters:
            object.__setattr__(self, name, int(getattr(self, name)))
        for name in real_parameters:
            object.__setattr__(self, name, float(getattr(self, name)))

    def ground_evidence_config(self):
        """Return the Task 2 configuration represented by this run config."""
        return GroundEvidenceConfig(
            resolution=float(self.resolution),
            min_points_per_cell=int(self.min_points_per_cell),
            min_ground_support_cells=int(self.min_ground_support_cells),
            ground_window_m=float(self.ground_window_m),
            ground_percentile=float(self.ground_percentile),
            ground_seed_percentile=float(self.ground_seed_percentile),
            max_ground_step_m=float(self.max_ground_step_m),
            max_interpolation_gap_m=float(self.max_interpolation_gap_m),
            obstacle_height_m=float(self.obstacle_height_m),
            obstacle_inflation_radius_m=float(self.obstacle_inflation_radius_m),
            interpolated_ground_cost=int(self.interpolated_ground_cost),
        )


@dataclass(frozen=True)
class Exp003Result:
    """Authoritative numeric products and traceability for one EXP003 run."""

    low_height: np.ndarray
    q10_height: np.ndarray
    q50_height: np.ndarray
    q90_height: np.ndarray
    ground_surface: np.ndarray
    clearance: np.ndarray
    point_count: np.ndarray
    ground_model_support: np.ndarray
    evidence: np.ndarray
    costmap: np.ndarray
    origin_xy: np.ndarray
    resolution: float
    input_points: int
    finite_input_points: int
    config: Exp003Config


def run_exp003(points, config):
    """Run EXP003 on XYZ points while preserving the input PCD coordinate frame."""
    if not isinstance(config, Exp003Config):
        raise TypeError("config must be an Exp003Config")
    points = np.asarray(points)
    statistics = points_to_elevation_statistics(
        points,
        resolution=config.resolution,
        chunk_size=config.chunk_size,
        low_quantile=config.low_quantile,
        histogram_bins=config.histogram_bins,
    )
    evidence_config = config.ground_evidence_config()
    evidence_result = build_ground_evidence_details(
        statistics.low_height,
        statistics.point_count,
        evidence_config,
        q90_height=statistics.q90_height if config.use_q90_for_obstacles else None,
    )
    costmap = build_navigation_costmap(evidence_result.evidence, evidence_config)
    finite_input_points = int(np.isfinite(points[:, :3]).all(axis=1).sum())
    return Exp003Result(
        low_height=statistics.low_height,
        q10_height=statistics.q10_height,
        q50_height=statistics.q50_height,
        q90_height=statistics.q90_height,
        ground_surface=evidence_result.ground_surface,
        clearance=evidence_result.clearance,
        point_count=statistics.point_count,
        ground_model_support=evidence_result.ground_model_support,
        evidence=evidence_result.evidence,
        costmap=costmap,
        origin_xy=statistics.origin_xy,
        resolution=statistics.resolution,
        input_points=int(len(points)),
        finite_input_points=finite_input_points,
        config=config,
    )


def _result_metrics(result):
    occupied = result.evidence == EvidenceClass.OCCUPIED_CONFIRMED
    return {
        "input_points": result.input_points,
        "finite_input_points": result.finite_input_points,
        "grid_cells": int(result.evidence.size),
        "measured_cells": int((result.point_count >= result.config.min_points_per_cell).sum()),
        "free_confirmed_cells": int(
            (result.evidence == EvidenceClass.FREE_CONFIRMED).sum()
        ),
        "occupied_confirmed_cells": int(occupied.sum()),
        "ground_interpolated_cells": int(
            (result.evidence == EvidenceClass.GROUND_INTERPOLATED).sum()
        ),
        "unknown_cells": int((result.evidence == EvidenceClass.UNKNOWN).sum()),
        "inflated_cells": int(((result.costmap == 254) & ~occupied).sum()),
    }


def _save_preview(array, path, title):
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 8))
    image = axis.imshow(array, origin="lower")
    axis.set_title(title)
    figure.colorbar(image, ax=axis)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _metadata_payload(result, metadata):
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    required = {
        "created_at_utc",
        "repository",
        "git_commit",
        "git_dirty",
        "input_pcd",
        "input_size_bytes",
    }
    missing = sorted(required - metadata.keys())
    if missing:
        raise ValueError(f"required metadata fields are missing: {', '.join(missing)}")
    if "experiment" in metadata and metadata["experiment"] != "EXP003":
        raise ValueError("experiment metadata must be EXP003")
    if "schema_version" in metadata and metadata["schema_version"] != 1:
        raise ValueError("schema_version metadata must be 1")
    for name in ("repository", "git_commit", "input_pcd"):
        if not isinstance(metadata[name], str) or not metadata[name]:
            raise ValueError(f"{name} must be a non-empty string")
    if not Path(metadata["input_pcd"]).is_absolute():
        raise ValueError("input_pcd must be an absolute path")
    commit = metadata["git_commit"]
    if len(commit) != 40 or any(
        character not in "0123456789abcdefABCDEF" for character in commit
    ):
        raise ValueError("git_commit must be a full 40-character hexadecimal SHA")
    if not isinstance(metadata["git_dirty"], bool):
        raise ValueError("git_dirty must be a boolean")
    if (
        not isinstance(metadata["input_size_bytes"], Integral)
        or isinstance(metadata["input_size_bytes"], bool)
        or metadata["input_size_bytes"] < 0
    ):
        raise ValueError("input_size_bytes must be a non-negative integer")
    created_at = metadata["created_at_utc"]
    if not isinstance(created_at, str):
        raise ValueError("created_at_utc must be a UTC ISO-8601 string")
    try:
        parsed_created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("created_at_utc must be a UTC ISO-8601 string") from error
    if parsed_created_at.utcoffset() != timedelta(0):
        raise ValueError("created_at_utc must include a UTC offset")
    if "input_sha256" in metadata:
        digest = metadata["input_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in digest)
        ):
            raise ValueError("input_sha256 must be a 64-character hexadecimal digest")

    payload = dict(metadata)
    payload["experiment"] = "EXP003"
    payload["schema_version"] = 1
    payload["input_size_bytes"] = int(payload["input_size_bytes"])
    payload.update(
        {
            "input_points": result.input_points,
            "finite_input_points": result.finite_input_points,
            "config": asdict(result.config),
            "grid_origin_xy_m": [float(value) for value in result.origin_xy],
            "grid_resolution_m": result.resolution,
            "grid_shape_yx": list(result.low_height.shape),
        }
    )
    return payload


def _publish_directory_no_clobber(staging_dir, run_dir):
    if os.name == "posix":
        renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
        if renameat2 is not None:
            renameat2.argtypes = (
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            )
            renameat2.restype = ctypes.c_int
            result = renameat2(
                -100,
                os.fsencode(staging_dir),
                -100,
                os.fsencode(run_dir),
                1,
            )
            if result == 0:
                return
            error_number = ctypes.get_errno()
            if error_number not in (errno.ENOSYS, errno.EINVAL):
                raise OSError(error_number, os.strerror(error_number), run_dir)
        raise RuntimeError(
            "atomic no-clobber publication is unavailable on this POSIX platform"
        )
    if run_dir.exists():
        raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), run_dir)
    os.rename(staging_dir, run_dir)


def write_exp003_results(result, run_dir, metadata):
    """Create an immutable run directory with numeric arrays and previews."""
    if not isinstance(result, Exp003Result):
        raise TypeError("result must be an Exp003Result")
    metadata_payload = _metadata_payload(result, metadata)
    metadata_yaml = yaml.safe_dump(metadata_payload, sort_keys=False)
    metrics_yaml = yaml.safe_dump(_result_metrics(result), sort_keys=False)
    run_dir = Path(run_dir)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    if run_dir.exists():
        raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), run_dir)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{run_dir.name}.",
            suffix=".tmp",
            dir=run_dir.parent,
        )
    )

    try:
        arrays = {
            "low_height": result.low_height,
            "ground_surface": result.ground_surface,
            "clearance": result.clearance,
            "point_count": result.point_count,
            "evidence": result.evidence,
            "costmap": result.costmap,
        }
        for name, array in arrays.items():
            np.save(staging_dir / f"{name}.npy", array, allow_pickle=False)
            if name != "point_count":
                _save_preview(
                    array,
                    staging_dir / f"{name}.png",
                    name.replace("_", " "),
                )

        (staging_dir / "metadata.yaml").write_text(metadata_yaml, encoding="utf-8")
        (staging_dir / "metrics.yaml").write_text(metrics_yaml, encoding="utf-8")
        _publish_directory_no_clobber(staging_dir, run_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


def sha256_file(path, chunk_size=1024 * 1024):
    """Hash a file without reading the complete PCD into memory."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
