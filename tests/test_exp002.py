from datetime import datetime, timezone
from pathlib import Path

import csv

import numpy as np
import pytest

from agt_map_reconstruction.experiments.exp002 import (
    Exp002Config,
    build_run_id,
    create_run_directory,
    sha256_file,
    run_exp002_from_maps,
    write_exp002_results,
)
from agt_map_reconstruction.maps.corridor import (
    extract_parallel_corridors,
    filter_row_aligned_components,
)
from agt_map_reconstruction.maps.grid_map import points_to_height_grid
from agt_map_reconstruction.maps.row_direction import estimate_row_direction


def _synthetic_greenhouse():
    relative_height = np.zeros((60, 120), dtype=float)
    relative_height[10:15, 10:110] = 0.45
    relative_height[35:40, 10:110] = 0.45
    relative_height[10:15, 55:60] = 0.0
    relative_height[35:40, 75:80] = 0.0

    traversability = np.ones_like(relative_height, dtype=np.uint8)
    traversability[relative_height > 0.15] = 2
    return relative_height, traversability


def test_height_grid_keeps_minimum_z_per_cell():
    points = np.array(
        [
            [0.01, 0.01, 0.30],
            [0.02, 0.02, 0.10],
            [0.11, 0.01, 0.25],
            [0.01, 0.11, 0.40],
        ],
        dtype=np.float32,
    )

    grid = points_to_height_grid(points, resolution=0.10, chunk_size=2)

    assert grid.shape == (2, 2)
    assert grid[0, 0] == pytest.approx(0.10)
    assert grid[0, 1] == pytest.approx(0.25)
    assert grid[1, 0] == pytest.approx(0.40)
    assert np.isnan(grid[1, 1])


def test_height_grid_does_not_collapse_float32_resolution_boundaries():
    x = (np.arange(120, dtype=np.float64) * 0.05).astype(np.float32)
    points = np.column_stack((x, np.zeros_like(x), np.ones_like(x)))

    grid = points_to_height_grid(points, resolution=0.05, chunk_size=17)

    assert grid.shape == (1, 120)
    assert np.isfinite(grid).all()


def test_height_grid_ignores_nonfinite_xyz_without_poisoning_cells():
    points = np.array(
        [
            [0.0, 0.0, 0.3],
            [0.0, 0.0, np.nan],
            [0.1, 0.0, 0.2],
            [np.nan, 0.0, 0.1],
            [np.inf, 0.0, 0.1],
        ],
        dtype=np.float32,
    )

    grid = points_to_height_grid(points, resolution=0.1, chunk_size=2)

    assert grid.shape == (1, 2)
    np.testing.assert_allclose(grid, [[0.3, 0.2]])


def test_componentwise_row_direction_and_filter_reject_cross_row_blob():
    structure = np.zeros((90, 130), dtype=float)
    structure[10:14, 10:115] = 0.5
    structure[35:39, 10:115] = 0.5
    structure[45:80, 118:122] = 0.5

    angle, direction = estimate_row_direction(
        structure,
        structure_threshold=0.2,
        component_min_cells=20,
    )

    assert abs(np.dot(direction, np.array([1.0, 0.0]))) > 0.98
    assert abs(np.sin(angle)) < 0.2

    candidates = np.zeros_like(structure, dtype=bool)
    candidates[10:15, 10:110] = True
    candidates[10:15, 30] = False
    candidates[10:15, 60] = False
    candidates[10:15, 90] = False
    candidates[25:75, 20:24] = True
    candidates[70:76, 80:86] = True

    filtered = filter_row_aligned_components(
        candidates,
        row_direction=direction,
        min_cells=20,
        min_length_cells=40,
        min_aspect_ratio=4.0,
        direction_threshold=0.9,
        max_longitudinal_gap_cells=1,
    )

    assert filtered[12, 50]
    assert not filtered[50, 22]
    assert not filtered[72, 82]


def test_parallel_corridor_extraction_uses_width_length_and_continuity():
    relative_height, traversability = _synthetic_greenhouse()

    corridor, details = extract_parallel_corridors(
        relative_height,
        traversability,
        row_direction=np.array([1.0, 0.0]),
        resolution=0.05,
        row_height_threshold=0.20,
        min_width_m=0.90,
        max_width_m=1.20,
        min_length_m=4.0,
        min_row_coverage=0.80,
    )

    assert corridor[20:33, 15:105].mean() > 0.90
    assert not corridor[2:8].any()
    assert not corridor[45:55].any()
    assert details["accepted_corridors"] == 1
    assert details["widths_m"][0] == pytest.approx(1.0, abs=0.1)


def test_sparse_row_profile_survives_grid_sampling():
    relative_height = np.zeros((60, 119), dtype=float)
    relative_height[10:12, 10:95] = 0.45
    relative_height[33:36, 10:95] = 0.45
    traversability = np.ones_like(relative_height, dtype=np.uint8)
    traversability[relative_height > 0.20] = 2

    corridor, details = extract_parallel_corridors(
        relative_height,
        traversability,
        row_direction=np.array([1.0, 0.0]),
        resolution=0.05,
        row_height_threshold=0.20,
        min_width_m=0.90,
        max_width_m=1.20,
        min_length_m=3.0,
        min_row_coverage=0.80,
        min_row_profile=0.25,
    )

    assert corridor.any()
    assert details["accepted_corridors"] == 1


def test_large_boundary_gap_is_not_filled_as_continuous_corridor():
    relative_height, traversability = _synthetic_greenhouse()
    relative_height[10:15, 45:75] = 0.0
    relative_height[35:40, 45:75] = 0.0
    traversability[:] = 1
    traversability[relative_height > 0.20] = 2

    corridor, _ = extract_parallel_corridors(
        relative_height,
        traversability,
        row_direction=np.array([1.0, 0.0]),
        resolution=0.05,
        row_height_threshold=0.20,
        min_width_m=0.90,
        max_width_m=1.20,
        min_length_m=1.0,
        min_row_coverage=0.70,
        max_boundary_gap_m=0.30,
    )

    assert corridor.any()
    assert not corridor[:, 45:75].any()


@pytest.mark.parametrize("rotation_deg", [-30.0, -15.0, 15.0, 30.0])
def test_row_frame_uses_image_coordinate_rotation_sign(rotation_deg):
    from scipy import ndimage

    relative_height, traversability = _synthetic_greenhouse()
    rotated_height = ndimage.rotate(
        relative_height,
        rotation_deg,
        reshape=False,
        order=0,
        mode="constant",
        cval=np.nan,
    )
    rotated_traversability = ndimage.rotate(
        traversability,
        rotation_deg,
        reshape=False,
        order=0,
        mode="constant",
        cval=0,
    )
    _, direction = estimate_row_direction(
        rotated_height,
        structure_threshold=0.20,
    )

    corridor, details = extract_parallel_corridors(
        rotated_height,
        rotated_traversability,
        row_direction=direction,
        resolution=0.05,
        row_height_threshold=0.20,
        min_width_m=0.85,
        max_width_m=1.25,
        min_length_m=3.0,
        min_row_coverage=0.70,
    )

    assert corridor.any()
    assert details["accepted_corridors"] == 1


def test_stage_b_closes_small_cross_row_breaks_before_component_pca():
    relative_height = np.zeros((60, 120), dtype=float)
    relative_height[10, 10:110] = 0.45
    relative_height[12, 10:110] = 0.45
    relative_height[35, 10:110] = 0.45
    relative_height[37, 10:110] = 0.45
    traversability = np.ones_like(relative_height, dtype=np.uint8)
    traversability[relative_height > 0.20] = 2
    config = Exp002Config(
        resolution=0.05,
        row_height_threshold=0.20,
        min_length_m=4.0,
        min_aspect_ratio=3.0,
        max_cross_row_gap_m=0.05,
    )

    result = run_exp002_from_maps(
        relative_height + 1.0,
        relative_height,
        traversability,
        config,
    )

    assert result.stages["B"].metrics["corridor_cells"] > 0
    assert result.stages["B"].metrics["corridor_cells"] < result.stages["A"].metrics["corridor_cells"]


def test_exp002_writes_immutable_abc_run_artifacts(tmp_path):
    relative_height, traversability = _synthetic_greenhouse()
    height = relative_height + 1.0
    config = Exp002Config(
        resolution=0.05,
        row_height_threshold=0.20,
        min_width_m=0.90,
        max_width_m=1.20,
        min_length_m=4.0,
        min_row_coverage=0.80,
    )

    result = run_exp002_from_maps(
        height,
        relative_height,
        traversability,
        config,
        origin_xy=(10.0, -2.0),
    )

    assert set(result.stages) == {"A", "B", "C"}
    assert result.stages["A"].metrics["corridor_cells"] > 0
    assert result.stages["C"].metrics["corridor_cells"] > 0
    assert result.stages["C"].metrics["corridor_cells"] < result.stages["A"].metrics["corridor_cells"]
    assert result.stages["C"].metrics["centerline_cells"] < result.stages["C"].metrics["corridor_cells"]
    assert np.all(
        result.stages["C"].centerline <= result.stages["C"].corridor
    )

    run_dir = create_run_directory(tmp_path, "fixed-run")
    write_exp002_results(
        result,
        run_dir,
        metadata={"experiment": "EXP002", "input_points": 1234},
    )

    expected = {
        "metadata.yaml",
        "metrics.yaml",
        "height.png",
        "relative_height.png",
        "traversability.png",
        "abc_comparison.png",
        "A/corridor.png",
        "A/centerline.png",
        "A/centerline.csv",
        "B/corridor.png",
        "B/centerline.png",
        "B/centerline.csv",
        "C/corridor.png",
        "C/centerline.png",
        "C/centerline.csv",
    }
    actual = {
        str(path.relative_to(run_dir))
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    assert expected <= actual

    with (run_dir / "A/centerline.csv").open(newline="") as handle:
        first = next(csv.DictReader(handle))
    assert set(first) == {"x_cell", "y_cell", "x_m", "y_m"}
    assert float(first["x_m"]) == pytest.approx(
        10.0 + (int(first["x_cell"]) + 0.5) * 0.05
    )
    assert float(first["y_m"]) == pytest.approx(
        -2.0 + (int(first["y_cell"]) + 0.5) * 0.05
    )

    with pytest.raises(FileExistsError):
        create_run_directory(tmp_path, "fixed-run")


def test_traceability_helpers_use_utc_commit_and_streaming_hash(tmp_path):
    instant = datetime(2026, 8, 24, 12, 34, 56, tzinfo=timezone.utc)
    assert build_run_id("2b519c8f3d5e", instant) == "20260824T123456Z_2b519c8"

    payload = tmp_path / "input.pcd"
    payload.write_bytes(b"EXP002\nprocessed.pcd\n")
    assert sha256_file(payload, chunk_size=4) == (
        "f3c099c3c4a4816251bfc65ea45e1ef4"
        "3f5e14a90b0115e92184cb9894658692"
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"resolution": 0.0},
        {"min_width_m": 2.0, "max_width_m": 1.0},
        {"min_row_coverage": 1.1},
        {"chunk_size": 0},
    ],
)
def test_exp002_config_rejects_invalid_geometry_and_runtime_values(overrides):
    with pytest.raises(ValueError):
        Exp002Config(**overrides)
