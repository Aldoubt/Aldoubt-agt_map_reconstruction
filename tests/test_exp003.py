from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest
import yaml

import agt_map_reconstruction.experiments.exp003 as exp003_module
from agt_map_reconstruction.experiments.exp003 import (
    Exp003Config,
    run_exp003,
    write_exp003_results,
)

from agt_map_reconstruction.maps.elevation_statistics import (
    points_to_elevation_statistics,
)
from agt_map_reconstruction.maps.ground_evidence import (
    EvidenceClass,
    GroundEvidenceConfig,
    build_ground_evidence,
    build_ground_evidence_details,
    build_navigation_costmap,
)


def test_elevation_statistics_uses_lowest_finite_xy_as_grid_origin():
    points = np.array(
        [
            [10.20, -1.80, 1.0],
            [10.02, -1.98, 2.0],
            [10.11, -1.89, 3.0],
        ],
        dtype=np.float64,
    )

    statistics = points_to_elevation_statistics(
        points,
        resolution=0.10,
        chunk_size=2,
        low_quantile=0.10,
        histogram_bins=32,
    )

    np.testing.assert_allclose(statistics.origin_xy, [10.02, -1.98])
    assert statistics.low_height.shape == (2, 2)


def test_elevation_statistics_counts_only_finite_xyz_samples():
    points = np.array(
        [
            [0.01, 0.01, 0.30],
            [0.02, 0.02, 0.10],
            [0.11, 0.01, 0.25],
            [0.01, 0.11, np.nan],
            [np.nan, 0.11, 0.40],
            [0.11, np.inf, 0.50],
        ],
        dtype=np.float64,
    )

    statistics = points_to_elevation_statistics(
        points,
        resolution=0.10,
        chunk_size=2,
        low_quantile=0.10,
        histogram_bins=32,
    )

    np.testing.assert_array_equal(statistics.point_count, [[2, 1]])
    np.testing.assert_allclose(statistics.minimum_height, [[0.10, 0.25]])
    np.testing.assert_allclose(statistics.maximum_height, [[0.30, 0.25]])
    assert np.isfinite(statistics.low_height).all()


def test_elevation_statistics_rejects_a_single_low_outlier_from_low_height():
    points = np.array(
        [[0.01, 0.01, -5.0]] + [[0.01, 0.01, 0.25]] * 31,
        dtype=np.float64,
    )

    statistics = points_to_elevation_statistics(
        points,
        resolution=0.10,
        chunk_size=7,
        low_quantile=0.10,
        histogram_bins=64,
    )

    assert statistics.minimum_height[0, 0] == pytest.approx(-5.0)
    assert statistics.low_height[0, 0] > 0.0
    assert statistics.low_height[0, 0] < 0.25


def test_elevation_statistics_rejects_low_outlier_despite_distant_global_high():
    points = np.array(
        [[0.01, 0.01, -5.0]]
        + [[0.01, 0.01, 0.25]] * 31
        + [[10.01, 0.01, 1_000_000.0]],
        dtype=np.float64,
    )

    statistics = points_to_elevation_statistics(
        points,
        resolution=0.10,
        chunk_size=7,
        low_quantile=0.10,
        histogram_bins=64,
    )

    assert statistics.low_height[0, 0] > 0.0


def test_elevation_statistics_marks_unobserved_in_grid_cells_with_nan_heights():
    points = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.2, 0.2, 2.0],
        ],
        dtype=np.float64,
    )

    statistics = points_to_elevation_statistics(
        points,
        resolution=0.10,
        chunk_size=1,
        low_quantile=0.10,
        histogram_bins=32,
    )

    assert statistics.point_count[1, 1] == 0
    assert np.isnan(statistics.low_height[1, 1])
    assert np.isnan(statistics.minimum_height[1, 1])
    assert np.isnan(statistics.maximum_height[1, 1])


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("resolution", np.nan),
        ("resolution", np.inf),
        ("chunk_size", 0.5),
        ("low_quantile", np.nan),
        ("low_quantile", np.inf),
    ],
)
def test_elevation_statistics_rejects_invalid_numeric_parameters(parameter, value):
    arguments = {
        "resolution": 0.10,
        "chunk_size": 1,
        "low_quantile": 0.10,
        "histogram_bins": 32,
    }
    arguments[parameter] = value

    with pytest.raises(ValueError):
        points_to_elevation_statistics(np.array([[0.0, 0.0, 0.0]]), **arguments)


def test_elevation_statistics_is_invariant_to_chunk_size():
    points = np.array(
        [
            [0.01, 0.01, 0.10],
            [0.01, 0.01, 0.20],
            [0.11, 0.01, 0.30],
            [0.11, 0.01, 0.40],
            [0.01, 0.11, 0.50],
            [0.01, 0.11, 0.60],
            [0.11, 0.11, 0.70],
        ],
        dtype=np.float64,
    )

    small_chunks = points_to_elevation_statistics(
        points,
        resolution=0.10,
        chunk_size=1,
        low_quantile=0.25,
        histogram_bins=32,
    )
    large_chunks = points_to_elevation_statistics(
        points,
        resolution=0.10,
        chunk_size=100,
        low_quantile=0.25,
        histogram_bins=32,
    )

    np.testing.assert_allclose(small_chunks.origin_xy, large_chunks.origin_xy)
    np.testing.assert_array_equal(small_chunks.point_count, large_chunks.point_count)
    np.testing.assert_allclose(
        small_chunks.minimum_height,
        large_chunks.minimum_height,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        small_chunks.maximum_height,
        large_chunks.maximum_height,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        small_chunks.low_height,
        large_chunks.low_height,
        equal_nan=True,
    )


def test_ground_evidence_distinguishes_measured_ground_and_elevated_obstacles():
    low_height = np.zeros((3, 3), dtype=np.float64)
    low_height[1, 1] = 1.0
    point_count = np.full((3, 3), 3, dtype=np.int64)

    evidence = build_ground_evidence(
        low_height,
        point_count,
        GroundEvidenceConfig(
            resolution=1.0,
            min_points_per_cell=3,
            ground_window_m=3.0,
            ground_percentile=20.0,
            obstacle_height_m=0.3,
            max_interpolation_gap_m=0.0,
        ),
    )

    assert evidence.dtype == np.uint8
    assert evidence[1, 1] == EvidenceClass.OCCUPIED_CONFIRMED
    assert evidence[0, 0] == EvidenceClass.FREE_CONFIRMED


def test_ground_evidence_preserves_low_density_cells_as_unknown_without_gap_support():
    low_height = np.zeros((3, 3), dtype=np.float64)
    point_count = np.full((3, 3), 3, dtype=np.int64)
    point_count[1, 1] = 2

    evidence = build_ground_evidence(
        low_height,
        point_count,
        GroundEvidenceConfig(
            resolution=1.0,
            min_points_per_cell=3,
            max_interpolation_gap_m=0.0,
        ),
    )

    assert evidence[1, 1] == EvidenceClass.UNKNOWN


def test_ground_evidence_labels_a_bounded_hole_as_interpolated_ground():
    low_height = np.zeros((5, 5), dtype=np.float64)
    point_count = np.full((5, 5), 3, dtype=np.int64)
    low_height[2, 2] = np.nan
    point_count[2, 2] = 0

    evidence = build_ground_evidence(
        low_height,
        point_count,
        GroundEvidenceConfig(
            resolution=1.0,
            min_points_per_cell=3,
            max_interpolation_gap_m=1.1,
        ),
    )

    assert evidence[2, 2] == EvidenceClass.GROUND_INTERPOLATED


def test_ground_evidence_rejects_a_large_hole_as_one_unknown_component():
    low_height = np.zeros((7, 7), dtype=np.float64)
    point_count = np.full((7, 7), 3, dtype=np.int64)
    low_height[1:6, 1:6] = np.nan
    point_count[1:6, 1:6] = 0

    evidence = build_ground_evidence(
        low_height,
        point_count,
        GroundEvidenceConfig(
            resolution=1.0,
            min_points_per_cell=3,
            max_interpolation_gap_m=2.1,
        ),
    )

    np.testing.assert_array_equal(
        evidence[1:6, 1:6],
        np.full((5, 5), EvidenceClass.UNKNOWN, dtype=np.uint8),
    )


def test_ground_evidence_details_linearly_interpolates_a_sloped_bounded_gap():
    low_height = np.tile(np.arange(7, dtype=np.float64), (7, 1))
    point_count = np.full((7, 7), 3, dtype=np.int64)
    low_height[2:5, 2:5] = np.nan
    point_count[2:5, 2:5] = 0

    details = build_ground_evidence_details(
        low_height,
        point_count,
        GroundEvidenceConfig(
            resolution=1.0,
            min_points_per_cell=3,
            ground_window_m=0.5,
            ground_percentile=50.0,
            max_ground_step_m=1.1,
            max_interpolation_gap_m=3.0,
            obstacle_height_m=10.0,
        ),
    )

    np.testing.assert_array_equal(
        details.evidence[2:5, 2:5],
        np.full((3, 3), EvidenceClass.GROUND_INTERPOLATED, dtype=np.uint8),
    )
    np.testing.assert_allclose(details.ground_surface[3, 2:5], [2.0, 3.0, 4.0])


def test_ground_evidence_does_not_interpolate_a_sparse_corner_region():
    low_height = np.full((9, 9), np.nan, dtype=np.float64)
    point_count = np.zeros((9, 9), dtype=np.int64)
    for row, column, height in (
        (1, 1, 0.0),
        (1, 7, 0.0),
        (7, 1, 0.0),
        (7, 7, 60.0),
    ):
        low_height[row, column] = height
        point_count[row, column] = 3

    details = build_ground_evidence_details(
        low_height,
        point_count,
        GroundEvidenceConfig(
            resolution=1.0,
            min_points_per_cell=3,
            ground_window_m=0.5,
            max_interpolation_gap_m=1.1,
            obstacle_height_m=100.0,
        ),
    )

    assert not np.any(details.evidence == EvidenceClass.GROUND_INTERPOLATED)


def test_ground_evidence_preserves_an_edge_open_gap_as_unknown():
    low_height = np.zeros((7, 7), dtype=np.float64)
    point_count = np.full((7, 7), 3, dtype=np.int64)
    low_height[0:4, 3] = np.nan
    point_count[0:4, 3] = 0

    details = build_ground_evidence_details(
        low_height,
        point_count,
        GroundEvidenceConfig(
            resolution=1.0,
            min_points_per_cell=3,
            max_interpolation_gap_m=1.1,
        ),
    )

    np.testing.assert_array_equal(
        details.evidence[0:4, 3],
        np.full(4, EvidenceClass.UNKNOWN, dtype=np.uint8),
    )


def test_ground_evidence_rejects_an_entire_oversized_hole():
    low_height = np.zeros((9, 9), dtype=np.float64)
    point_count = np.full((9, 9), 3, dtype=np.int64)
    low_height[1:8, 1:8] = np.nan
    point_count[1:8, 1:8] = 0

    details = build_ground_evidence_details(
        low_height,
        point_count,
        GroundEvidenceConfig(
            resolution=1.0,
            min_points_per_cell=3,
            max_interpolation_gap_m=1.1,
        ),
    )

    np.testing.assert_array_equal(
        details.evidence[1:8, 1:8],
        np.full((7, 7), EvidenceClass.UNKNOWN, dtype=np.uint8),
    )


def test_ground_evidence_does_not_self_confirm_an_isolated_measurement():
    low_height = np.full((5, 5), np.nan, dtype=np.float64)
    point_count = np.zeros((5, 5), dtype=np.int64)
    low_height[2, 2] = 5.0
    point_count[2, 2] = 3

    details = build_ground_evidence_details(
        low_height,
        point_count,
        GroundEvidenceConfig(
            resolution=1.0,
            min_points_per_cell=3,
            min_ground_support_cells=3,
            ground_window_m=2.0,
        ),
    )

    assert not details.ground_model_support[2, 2]
    assert details.evidence[2, 2] == EvidenceClass.UNKNOWN
    assert np.isnan(details.ground_surface[2, 2])


def test_ground_evidence_rejects_a_disconnected_elevated_patch():
    low_height = np.full((9, 9), np.nan, dtype=np.float64)
    point_count = np.zeros((9, 9), dtype=np.int64)
    low_height[0:3, 3:6] = 5.0
    point_count[0:3, 3:6] = 3
    low_height[6:9, :] = 0.0
    point_count[6:9, :] = 3

    details = build_ground_evidence_details(
        low_height,
        point_count,
        GroundEvidenceConfig(
            resolution=1.0,
            min_points_per_cell=3,
            min_ground_support_cells=2,
            ground_window_m=2.0,
            ground_seed_percentile=20.0,
            max_ground_step_m=0.3,
        ),
    )

    np.testing.assert_array_equal(
        details.ground_model_support[0:3, 3:6],
        np.zeros((3, 3), dtype=bool),
    )
    np.testing.assert_array_equal(
        details.evidence[0:3, 3:6],
        np.full((3, 3), EvidenceClass.UNKNOWN, dtype=np.uint8),
    )


def test_ground_evidence_rejects_interior_of_connected_elevated_plateau():
    low_height = np.zeros((11, 11), dtype=np.float64)
    point_count = np.full((11, 11), 3, dtype=np.int64)
    low_height[3:8, 3:8] = 5.0

    details = build_ground_evidence_details(
        low_height,
        point_count,
        GroundEvidenceConfig(
            resolution=1.0,
            min_points_per_cell=3,
            min_ground_support_cells=2,
            ground_window_m=2.0,
            ground_seed_percentile=20.0,
            max_ground_step_m=0.3,
            obstacle_height_m=0.3,
            max_interpolation_gap_m=0.0,
        ),
    )

    assert not details.ground_model_support[5, 5]
    assert details.evidence[5, 5] == EvidenceClass.UNKNOWN
    assert details.evidence[3, 5] == EvidenceClass.OCCUPIED_CONFIRMED


def test_ground_evidence_support_propagates_across_a_gradual_slope():
    low_height = np.tile(np.arange(11, dtype=np.float64) * 0.1, (7, 1))
    point_count = np.full(low_height.shape, 3, dtype=np.int64)

    details = build_ground_evidence_details(
        low_height,
        point_count,
        GroundEvidenceConfig(
            resolution=1.0,
            min_points_per_cell=3,
            min_ground_support_cells=2,
            ground_window_m=2.0,
            ground_seed_percentile=20.0,
            max_ground_step_m=0.15,
            obstacle_height_m=0.3,
            max_interpolation_gap_m=0.0,
        ),
    )

    assert details.ground_model_support.all()
    np.testing.assert_array_equal(
        details.evidence,
        np.full(low_height.shape, EvidenceClass.FREE_CONFIRMED, dtype=np.uint8),
    )


def test_navigation_costmap_uses_metric_euclidean_obstacle_inflation():
    evidence = np.full((5, 5), EvidenceClass.FREE_CONFIRMED, dtype=np.uint8)
    evidence[2, 2] = EvidenceClass.OCCUPIED_CONFIRMED
    evidence[0, 0] = EvidenceClass.UNKNOWN
    evidence[4, 4] = EvidenceClass.GROUND_INTERPOLATED

    costmap = build_navigation_costmap(
        evidence,
        GroundEvidenceConfig(
            resolution=1.0,
            obstacle_inflation_radius_m=1.5,
            interpolated_ground_cost=17,
        ),
    )

    assert costmap.dtype == np.uint8
    assert costmap[2, 2] == 254
    assert costmap[1, 1] == 254
    assert costmap[0, 2] == 0
    assert costmap[0, 0] == 255
    assert costmap[4, 4] == 17


def _synthetic_exp003_points():
    points = []
    for y in range(3):
        for x in range(3):
            height = 1.0 if (x, y) == (1, 1) else 0.0
            points.extend(
                [
                    [10.01 + x, -1.99 + y, height],
                    [10.11 + x, -1.89 + y, height],
                    [10.21 + x, -1.79 + y, height],
                ]
            )
    return np.asarray(points, dtype=np.float64)


def _exp003_metadata(**overrides):
    metadata = {
        "created_at_utc": "2026-08-24T12:34:56+00:00",
        "repository": "Aldoubt/Aldoubt-agt_map_reconstruction",
        "git_commit": "0123456789abcdef0123456789abcdef01234567",
        "git_dirty": False,
        "input_pcd": "/data/processed.pcd",
        "input_size_bytes": 123456,
    }
    metadata.update(overrides)
    return metadata


def _exp003_cli_namespace(name):
    script = Path(__file__).parents[1] / "tools" / "run_ground_evidence_test.py"
    namespace = {"__name__": name, "__file__": str(script)}
    exec(compile(script.read_bytes(), script, "exec"), namespace)
    return namespace


@pytest.mark.parametrize(
    "overrides",
    [
        {"resolution": 0.0},
        {"resolution": np.inf},
        {"chunk_size": 0},
        {"low_quantile": 1.1},
        {"histogram_bins": 0},
        {"min_points_per_cell": 0},
        {"min_ground_support_cells": 0},
        {"ground_window_m": 0.0},
        {"ground_percentile": 101.0},
        {"ground_seed_percentile": 101.0},
        {"max_ground_step_m": -0.1},
        {"max_interpolation_gap_m": -0.1},
        {"obstacle_height_m": -0.1},
        {"obstacle_inflation_radius_m": -0.1},
        {"interpolated_ground_cost": 254},
    ],
)
def test_exp003_config_rejects_invalid_algorithm_parameters(overrides):
    with pytest.raises(ValueError):
        Exp003Config(**overrides)


@pytest.mark.parametrize(
    "parameter",
    [
        "resolution",
        "chunk_size",
        "low_quantile",
        "histogram_bins",
        "min_points_per_cell",
        "min_ground_support_cells",
        "ground_window_m",
        "ground_percentile",
        "ground_seed_percentile",
        "max_ground_step_m",
        "max_interpolation_gap_m",
        "obstacle_height_m",
        "obstacle_inflation_radius_m",
        "interpolated_ground_cost",
    ],
)
def test_exp003_config_rejects_boolean_numeric_parameters(parameter):
    with pytest.raises(ValueError):
        Exp003Config(**{parameter: True})


def test_exp003_config_normalizes_numpy_scalars_for_a_complete_write(tmp_path):
    config = Exp003Config(
        resolution=np.float64(1.0),
        chunk_size=np.int64(5),
        low_quantile=np.float64(0.1),
        histogram_bins=np.int64(16),
        min_points_per_cell=np.int64(3),
        min_ground_support_cells=np.int64(2),
        ground_window_m=np.float64(2.0),
        ground_percentile=np.float64(20.0),
        ground_seed_percentile=np.float64(10.0),
        max_ground_step_m=np.float64(0.2),
        max_interpolation_gap_m=np.float64(0.0),
        obstacle_height_m=np.float64(0.3),
        obstacle_inflation_radius_m=np.float64(0.0),
        interpolated_ground_cost=np.int64(64),
    )
    result = run_exp003(_synthetic_exp003_points(), config)
    run_dir = tmp_path / "numpy-scalars"

    write_exp003_results(
        result,
        run_dir,
        _exp003_metadata(input_size_bytes=np.int64(123456)),
    )

    saved = yaml.safe_load((run_dir / "metadata.yaml").read_text())
    assert type(config.resolution) is float
    assert type(config.chunk_size) is int
    assert type(config.min_ground_support_cells) is int
    assert type(saved["config"]["resolution"]) is float
    assert type(saved["config"]["chunk_size"]) is int
    assert saved["config"]["min_ground_support_cells"] == 2


def test_run_exp003_preserves_pcd_frame_origin_and_builds_numeric_products():
    points = _synthetic_exp003_points()
    config = Exp003Config(
        resolution=1.0,
        chunk_size=5,
        low_quantile=0.1,
        histogram_bins=16,
        min_points_per_cell=3,
        ground_window_m=2.0,
        ground_percentile=20.0,
        max_interpolation_gap_m=0.0,
        obstacle_height_m=0.3,
        obstacle_inflation_radius_m=0.0,
    )

    result = run_exp003(points, config)

    np.testing.assert_allclose(result.origin_xy, [10.01, -1.99])
    assert result.resolution == 1.0
    assert result.low_height.shape == (3, 3)
    assert result.ground_surface.shape == (3, 3)
    assert result.clearance.shape == (3, 3)
    assert result.point_count.shape == (3, 3)
    assert result.evidence.shape == (3, 3)
    assert result.costmap.shape == (3, 3)
    assert result.ground_model_support.shape == (3, 3)
    assert result.evidence[1, 1] == EvidenceClass.OCCUPIED_CONFIRMED
    assert result.clearance[1, 1] == pytest.approx(1.0)


def test_run_exp003_fills_ground_surface_for_every_bounded_interpolated_cell():
    points = []
    for y in range(5):
        for x in range(5):
            if 1 <= x <= 3 and 1 <= y <= 3:
                continue
            points.extend([[x + 0.01, y + 0.01, 0.0]] * 3)
    result = run_exp003(
        np.asarray(points, dtype=np.float64),
        Exp003Config(
            resolution=1.0,
            min_points_per_cell=3,
            ground_window_m=0.5,
            max_interpolation_gap_m=2.1,
            obstacle_inflation_radius_m=0.0,
        ),
    )

    interpolated = result.evidence == EvidenceClass.GROUND_INTERPOLATED
    assert interpolated[2, 2]
    assert np.isfinite(result.ground_surface[interpolated]).all()
    np.testing.assert_allclose(result.ground_surface[interpolated], 0.0)


def test_run_exp003_exports_the_same_continuous_sloped_gap_surface():
    points = []
    for y in range(7):
        for x in range(7):
            if 2 <= x <= 4 and 2 <= y <= 4:
                continue
            points.extend([[x + 0.01, y + 0.01, float(x)]] * 3)

    result = run_exp003(
        np.asarray(points, dtype=np.float64),
        Exp003Config(
            resolution=1.0,
            min_points_per_cell=3,
            ground_window_m=0.5,
            ground_percentile=50.0,
            max_ground_step_m=1.1,
            max_interpolation_gap_m=3.0,
            obstacle_height_m=10.0,
            obstacle_inflation_radius_m=0.0,
        ),
    )

    np.testing.assert_allclose(result.ground_surface[3, 2:5], [2.0, 3.0, 4.0])


def test_write_exp003_results_creates_immutable_authoritative_artifacts(tmp_path):
    points = _synthetic_exp003_points()
    config = Exp003Config(
        resolution=1.0,
        chunk_size=5,
        min_points_per_cell=3,
        ground_window_m=2.0,
        max_interpolation_gap_m=0.0,
        obstacle_height_m=0.3,
        obstacle_inflation_radius_m=0.0,
    )
    result = run_exp003(points, config)
    run_dir = tmp_path / "fixed-run"

    write_exp003_results(
        result,
        run_dir,
        metadata=_exp003_metadata(),
    )

    expected = {
        "metadata.yaml",
        "metrics.yaml",
        "low_height.npy",
        "ground_surface.npy",
        "clearance.npy",
        "point_count.npy",
        "evidence.npy",
        "costmap.npy",
        "low_height.png",
        "ground_surface.png",
        "clearance.png",
        "evidence.png",
        "costmap.png",
    }
    assert {path.name for path in run_dir.iterdir()} == expected

    for name in (
        "low_height",
        "ground_surface",
        "clearance",
        "point_count",
        "evidence",
        "costmap",
    ):
        np.testing.assert_array_equal(
            np.load(run_dir / f"{name}.npy"),
            getattr(result, name),
        )

    metadata = yaml.safe_load((run_dir / "metadata.yaml").read_text())
    assert metadata["experiment"] == "EXP003"
    assert metadata["schema_version"] == 1
    assert metadata["created_at_utc"] == "2026-08-24T12:34:56+00:00"
    assert metadata["repository"] == "Aldoubt/Aldoubt-agt_map_reconstruction"
    assert metadata["git_commit"] == "0123456789abcdef0123456789abcdef01234567"
    assert metadata["git_dirty"] is False
    assert metadata["input_pcd"] == "/data/processed.pcd"
    assert metadata["input_size_bytes"] == 123456
    assert metadata["input_points"] == 27
    assert metadata["grid_origin_xy_m"] == pytest.approx([10.01, -1.99])
    assert metadata["grid_resolution_m"] == 1.0
    assert metadata["grid_shape_yx"] == [3, 3]
    assert metadata["config"]["obstacle_height_m"] == 0.3
    assert metadata["config"]["min_ground_support_cells"] == 2

    metrics = yaml.safe_load((run_dir / "metrics.yaml").read_text())
    assert metrics == {
        "input_points": 27,
        "finite_input_points": 27,
        "grid_cells": 9,
        "measured_cells": 9,
        "free_confirmed_cells": 8,
        "occupied_confirmed_cells": 1,
        "ground_interpolated_cells": 0,
        "unknown_cells": 0,
        "inflated_cells": 0,
    }

    metadata_before_duplicate = (run_dir / "metadata.yaml").read_bytes()
    with pytest.raises(FileExistsError):
        write_exp003_results(result, run_dir, metadata=_exp003_metadata())
    assert (run_dir / "metadata.yaml").read_bytes() == metadata_before_duplicate
    assert {path.name for path in tmp_path.iterdir()} == {"fixed-run"}


def test_write_exp003_results_cleans_staging_after_an_artifact_failure(
    tmp_path,
    monkeypatch,
):
    result = run_exp003(_synthetic_exp003_points(), Exp003Config(resolution=1.0))
    run_dir = tmp_path / "failed-run"

    def fail_preview(*args, **kwargs):
        raise RuntimeError("injected preview failure")

    monkeypatch.setattr(exp003_module, "_save_preview", fail_preview)

    with pytest.raises(RuntimeError, match="injected preview failure"):
        write_exp003_results(result, run_dir, _exp003_metadata())

    assert not run_dir.exists()
    assert list(tmp_path.iterdir()) == []


def test_atomic_publish_fails_closed_without_no_replace_support(
    tmp_path,
    monkeypatch,
):
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    (staging_dir / "artifact.npy").write_bytes(b"complete")
    run_dir = tmp_path / "final"
    monkeypatch.setattr(
        exp003_module.ctypes,
        "CDLL",
        lambda *args, **kwargs: object(),
    )

    with pytest.raises(RuntimeError, match="atomic no-clobber publication"):
        exp003_module._publish_directory_no_clobber(staging_dir, run_dir)

    assert staging_dir.is_dir()
    assert not run_dir.exists()


def test_write_exp003_results_rejects_incomplete_provenance_before_creation(tmp_path):
    result = run_exp003(_synthetic_exp003_points(), Exp003Config(resolution=1.0))
    run_dir = tmp_path / "incomplete"

    with pytest.raises(ValueError, match="required metadata"):
        write_exp003_results(result, run_dir, metadata={})

    assert not run_dir.exists()


@pytest.mark.parametrize(
    "identity",
    [
        {"experiment": "EXP002"},
        {"schema_version": 99},
    ],
)
def test_write_exp003_results_rejects_contradictory_identity(tmp_path, identity):
    result = run_exp003(_synthetic_exp003_points(), Exp003Config(resolution=1.0))
    run_dir = tmp_path / "contradictory"

    with pytest.raises(ValueError):
        write_exp003_results(result, run_dir, _exp003_metadata(**identity))

    assert not run_dir.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("created_at_utc", "2026-08-24T12:34:56+01:00"),
        ("repository", ""),
        ("git_commit", ""),
        ("git_commit", "0123456789abcdef"),
        ("git_commit", "g" * 40),
        ("git_dirty", "false"),
        ("input_pcd", ""),
        ("input_pcd", "relative/processed.pcd"),
        ("input_size_bytes", -1),
        ("input_size_bytes", True),
        ("input_sha256", "not-a-sha256"),
    ],
)
def test_write_exp003_results_rejects_invalid_provenance_values(
    tmp_path,
    field,
    value,
):
    result = run_exp003(_synthetic_exp003_points(), Exp003Config(resolution=1.0))
    run_dir = tmp_path / f"invalid-{field}"

    with pytest.raises(ValueError):
        write_exp003_results(
            result,
            run_dir,
            _exp003_metadata(**{field: value}),
        )

    assert not run_dir.exists()


def test_write_exp003_results_rejects_non_mapping_metadata(tmp_path):
    result = run_exp003(_synthetic_exp003_points(), Exp003Config(resolution=1.0))

    with pytest.raises(TypeError, match="mapping"):
        write_exp003_results(result, tmp_path / "invalid-metadata", metadata=None)


def test_exp003_cli_module_import_does_not_require_open3d():
    namespace = _exp003_cli_namespace("exp003_cli_import_test")

    assert callable(namespace["main"])


def test_exp003_cli_accepts_all_algorithm_parameters():
    namespace = _exp003_cli_namespace("exp003_cli_argument_test")

    args = namespace["_parse_args"](
        [
            "--pcd", "input.pcd",
            "--output", "runs",
            "--run-id", "fixed-run",
            "--hash-pcd",
            "--resolution", "0.1",
            "--chunk-size", "99",
            "--low-quantile", "0.2",
            "--histogram-bins", "32",
            "--min-points-per-cell", "4",
            "--min-ground-support-cells", "5",
            "--ground-window-m", "0.6",
            "--ground-percentile", "25",
            "--ground-seed-percentile", "15",
            "--max-ground-step-m", "0.12",
            "--max-interpolation-gap-m", "0.3",
            "--obstacle-height-m", "0.2",
            "--obstacle-inflation-radius-m", "0.4",
            "--interpolated-ground-cost", "72",
        ]
    )

    assert args.pcd == Path("input.pcd")
    assert args.output == Path("runs")
    assert args.run_id == "fixed-run"
    assert args.hash_pcd is True
    assert args.resolution == 0.1
    assert args.chunk_size == 99
    assert args.low_quantile == 0.2
    assert args.histogram_bins == 32
    assert args.min_points_per_cell == 4
    assert args.min_ground_support_cells == 5
    assert args.ground_window_m == 0.6
    assert args.ground_percentile == 25.0
    assert args.ground_seed_percentile == 15.0
    assert args.max_ground_step_m == 0.12
    assert args.max_interpolation_gap_m == 0.3
    assert args.obstacle_height_m == 0.2
    assert args.obstacle_inflation_radius_m == 0.4
    assert args.interpolated_ground_cost == 72


@pytest.mark.parametrize("run_id", ["../outside", ".", ".."])
def test_exp003_cli_rejects_a_run_id_that_escapes_the_output_root(run_id):
    namespace = _exp003_cli_namespace("exp003_cli_run_id_test")

    with pytest.raises(SystemExit):
        namespace["_parse_args"](
            ["--pcd", "input.pcd", "--run-id", run_id]
        )


def test_exp003_git_provenance_is_anchored_when_called_outside_the_repository(
    tmp_path,
    monkeypatch,
):
    namespace = _exp003_cli_namespace("exp003_cli_git_cwd_test")
    repository_root = Path(__file__).parents[1]
    expected_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.chdir(tmp_path)

    provenance = namespace["_git_provenance"]()

    assert provenance["git_commit"] == expected_commit
    assert type(provenance["git_dirty"]) is bool


def test_exp003_git_command_failure_is_not_reported_as_clean():
    namespace = _exp003_cli_namespace("exp003_cli_git_failure_test")

    with pytest.raises(RuntimeError, match="Git command failed"):
        namespace["_git_output"]("not-a-git-subcommand")


def test_exp003_cli_rejects_a_replaced_pcd_before_publication(
    tmp_path,
    monkeypatch,
):
    namespace = _exp003_cli_namespace("exp003_cli_pcd_replacement_test")
    pcd_path = tmp_path / "processed.pcd"
    pcd_path.write_bytes(b"original PCD snapshot")
    replacement = tmp_path / "replacement.pcd"
    replacement.write_bytes(b"replacement PCD with different identity and size")
    output_root = tmp_path / "results"

    fake_loader = types.ModuleType("agt_map_reconstruction.io.pcd_loader")

    def load_and_replace(path):
        replacement.replace(path)
        return _synthetic_exp003_points()

    fake_loader.load_pcd = load_and_replace
    monkeypatch.setitem(
        sys.modules,
        "agt_map_reconstruction.io.pcd_loader",
        fake_loader,
    )

    with pytest.raises(RuntimeError, match="input PCD changed"):
        namespace["main"](
            [
                "--pcd", str(pcd_path),
                "--output", str(output_root),
                "--run-id", "replaced-input",
                "--hash-pcd",
            ]
        )

    assert not output_root.exists()
