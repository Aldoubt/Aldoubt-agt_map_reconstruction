import numpy as np
import pytest

from agt_map_reconstruction.algorithms.morphological_pmf import (
    PMFConfig,
    progressive_morphological_filter,
)


def _terrain_with_raised_strip():
    points = []
    for y in range(20):
        for x in range(40):
            z = 0.0 if not 14 <= y < 17 else 0.8
            points.append([x * 0.1, y * 0.1, z])
    return np.asarray(points, dtype=np.float64)


def test_pmf_rejects_a_raised_strip_and_keeps_flat_ground():
    result = progressive_morphological_filter(
        _terrain_with_raised_strip(),
        PMFConfig(resolution=0.1, tile_size=16, max_window_m=1.0),
    )
    assert len(result["ground"]) > 500
    assert len(result["non_ground"]) > 0
    assert result["ground_surface"].shape == (20, 40)


def test_pmf_tile_size_does_not_change_classification_at_tile_boundary():
    points = _terrain_with_raised_strip()
    small = progressive_morphological_filter(
        points, PMFConfig(resolution=0.1, tile_size=10, max_window_m=0.4)
    )
    large = progressive_morphological_filter(
        points, PMFConfig(resolution=0.1, tile_size=40, max_window_m=0.4)
    )
    assert len(small["ground"]) == len(large["ground"])
    assert len(small["non_ground"]) == len(large["non_ground"])


@pytest.mark.parametrize("field", ["resolution", "chunk_size", "tile_size", "max_window_m"])
def test_pmf_rejects_invalid_configuration(field):
    values = {field: 0}
    with pytest.raises(ValueError):
        PMFConfig(**values)
