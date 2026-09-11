from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agt_map_reconstruction.preprocessing.cleaning import (
    CleaningConfig,
    RadiusOutlierConfig,
    clean_points,
    load_cleaning_config,
    radius_outlier_filter,
    voxel_filter,
)


def test_voxel_filter_keeps_one_point_per_voxel():
    points = np.array(
        [
            [0.01, 0.01, 0.01],
            [0.02, 0.02, 0.02],
            [0.21, 0.00, 0.00],
        ],
        dtype=np.float32,
    )

    kept, removed = voxel_filter(points, 0.10)

    assert kept.shape == (2, 3)
    assert removed.shape == (1, 3)
    np.testing.assert_allclose(kept[0], points[0])
    np.testing.assert_allclose(removed[0], points[1])


def test_radius_filter_removes_isolated_point():
    dense = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.02, 0.00, 0.00],
            [0.00, 0.02, 0.00],
        ],
        dtype=np.float32,
    )
    points = np.vstack([dense, np.array([[5.0, 5.0, 0.0]], dtype=np.float32)])

    kept, removed = radius_outlier_filter(points, radius=0.05, min_neighbors=2)

    assert len(kept) == 3
    assert len(removed) == 1
    np.testing.assert_allclose(removed[0], [5.0, 5.0, 0.0])


def test_clean_points_reports_stage_counts():
    points = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.01, 0.01, 0.00],
            [0.20, 0.00, 0.00],
            [5.00, 5.00, 0.00],
        ],
        dtype=np.float32,
    )
    config = CleaningConfig(
        voxel_size=0.05,
        radius_outlier=RadiusOutlierConfig(
            enabled=False,
            radius=0.30,
            min_neighbors=8,
        ),
    )

    result = clean_points(points, config)

    assert result.metadata["input_points"] == 4
    assert result.metadata["output_points"] == 3
    assert result.metadata["removed_points"] == 1
    assert result.metadata["dynamic_ghost_cleaner"]["implemented"] is False


def test_load_migrated_config(tmp_path):
    config_path = tmp_path / "clean.yaml"
    config_path.write_text(
        """
preprocessing:
  voxel_size: 0.12
  radius_outlier:
    enabled: false
    radius: 0.4
    min_neighbors: 5
  cluster_filter:
    enabled: false
    tolerance: 0.3
    min_cluster_size: 10
    max_cluster_size: null
""",
        encoding="utf-8",
    )

    config = load_cleaning_config(config_path)

    assert config.voxel_size == 0.12
    assert config.radius_outlier.enabled is False
    assert config.radius_outlier.min_neighbors == 5
    assert config.cluster_filter.max_cluster_size is None
