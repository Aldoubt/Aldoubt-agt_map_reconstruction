import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agt_map_reconstruction.maps.navigation_export import (
    FREE_VALUE,
    UNKNOWN_VALUE,
    build_navigation_layers,
)


def test_uncertainty_vetoes_only_new_trusted_promotion_and_preserves_baseline_free():
    semantic = np.zeros((5, 8), dtype=np.uint8)
    semantic[2, 2] = 1  # already-confirmed baseline free

    trusted = np.zeros_like(semantic, dtype=bool)
    trusted[2, 3] = True  # new trusted promotion
    trusted[2, 4] = True  # trusted, but explicitly uncertain

    uncertainty = np.zeros_like(semantic, dtype=bool)
    uncertainty[2, 2] = True  # overlaps existing baseline free
    uncertainty[2, 4] = True  # must veto the new promotion

    layers = build_navigation_layers(
        semantic,
        [],
        promote_aisle_prior=False,
        trusted_free_mask=trusted,
        uncertainty_mask=uncertainty,
    )

    assert layers.base_map[2, 2] == FREE_VALUE
    assert layers.base_map[2, 3] == FREE_VALUE
    assert layers.base_map[2, 4] == UNKNOWN_VALUE


def test_overlay_reports_baseline_free_separately_from_uncertain_new_promotion(tmp_path):
    from agt_map_reconstruction.maps.navigation_export import write_navigation_bundle

    semantic = np.zeros((5, 8), dtype=np.uint8)
    semantic[2, 2] = 1

    trusted = np.zeros_like(semantic, dtype=bool)
    trusted[2, 3] = True
    trusted[2, 4] = True

    uncertainty = np.zeros_like(semantic, dtype=bool)
    uncertainty[2, 2] = True
    uncertainty[2, 4] = True

    result = write_navigation_bundle(
        semantic_labels=semantic,
        aisle_rectangles=[],
        output_dir=tmp_path,
        resolution=0.05,
        promote_aisle_prior=False,
        trusted_free_mask=trusted,
        uncertainty_mask=uncertainty,
        clearance_radii_m=(0.10,),
    )

    validation = result["validation"]
    assert validation["uncertainty_baseline_free_overlap_cell_count"] == 1
    assert validation["uncertainty_nonbaseline_exported_as_free_cell_count"] == 0
    assert validation["trusted_free_blocked_by_uncertainty_cell_count"] == 1
    assert validation["conservative_uncertainty_semantics_valid"] is True
