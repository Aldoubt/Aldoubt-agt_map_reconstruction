import json

import numpy as np

from agt_map_reconstruction.maps.grid_geometry import GridMetadata
from agt_map_reconstruction.maps.ground_evidence import EvidenceClass
from agt_map_reconstruction.maps.navigation_export import OCCUPIED_VALUE
from agt_map_reconstruction.maps.semantic_assets import write_semantic_navigation_assets
from agt_map_reconstruction.maps.semantic_reconstruction import LABEL_OCCUPIED_CONFIRMED


def _five_row_bands_with_one_wide_region():
    evidence = np.zeros((100, 120), dtype=np.uint8)
    evidence[5:9, 5:115] = EvidenceClass.FREE_CONFIRMED
    evidence[15:20, 5:115] = EvidenceClass.FREE_CONFIRMED
    evidence[27:33, 5:115] = EvidenceClass.FREE_CONFIRMED
    evidence[40:47, 5:115] = EvidenceClass.FREE_CONFIRMED
    evidence[60:85, 5:115] = EvidenceClass.FREE_CONFIRMED
    evidence[70, 60] = EvidenceClass.OCCUPIED_CONFIRMED
    return evidence


def test_semantic_bundle_excludes_wide_band_from_row_aisle_navigation_prior(tmp_path):
    evidence = _five_row_bands_with_one_wide_region()
    metadata = GridMetadata(
        resolution=0.10,
        origin_x=0.0,
        origin_y=0.0,
        width=evidence.shape[1],
        height=evidence.shape[0],
    )

    result = write_semantic_navigation_assets(
        evidence=evidence,
        metadata=metadata,
        row_direction=np.array([1.0, 0.0]),
        output_dir=tmp_path,
        min_longitudinal_support_ratio=0.50,
        min_width_m=0.30,
        min_length_m=5.0,
        occupied_aisle_conflict_policy="candidate",
        navigation_clearance_radii_m=(0.10,),
    )

    manifest = result["manifest"]
    assert manifest["raw_row_band_count"] == 5
    assert manifest["aisle_count"] == 4
    assert manifest["open_area_candidate_count"] == 1
    assert np.isclose(
        manifest["geometry_policy"]["wide_band_width_threshold_m"],
        1.0,
    )
    assert manifest["geometry_policy"]["wide_band_classification"] == (
        "upper_width_outlier_q3_plus_iqr"
    )

    region_payload = json.loads((tmp_path / "row_band_regions.json").read_text())
    assert region_payload["classification"]["raw_row_band_count"] == 5
    assert region_payload["classification"]["row_aisle_count"] == 4
    assert region_payload["classification"]["open_area_candidate_count"] == 1
    assert region_payload["regions"][-1]["region_class"] == (
        "wide_open_area_candidate"
    )

    # The occupied return lies inside the wide open-area candidate, not inside
    # one of the four accepted row aisles. It must therefore remain hard even
    # when aisle-conflict relaxation is explicitly enabled.
    labels = np.load(tmp_path / "semantic_labels.npy")
    assert labels[70, 60] == LABEL_OCCUPIED_CONFIRMED
    assert result["navigation"]["layers"].base_map[70, 60] == OCCUPIED_VALUE
    assert result["manifest"]["aisle_conflict_candidate_count"] == 0
