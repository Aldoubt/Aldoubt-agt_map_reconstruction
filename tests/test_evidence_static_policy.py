import numpy as np

from agt_map_reconstruction.maps.grid_geometry import GridMetadata
from agt_map_reconstruction.maps.ground_evidence import EvidenceClass
from agt_map_reconstruction.maps.navigation_export import UNKNOWN_VALUE
from agt_map_reconstruction.maps.semantic_assets import write_semantic_navigation_assets


def test_evidence_bundle_does_not_promote_interpolated_aisle_cells_to_static_free(tmp_path):
    evidence = np.zeros((12, 40), dtype=np.uint8)
    evidence[3:7, 3:37] = EvidenceClass.FREE_CONFIRMED
    evidence[4, 20] = EvidenceClass.GROUND_INTERPOLATED
    metadata = GridMetadata(
        resolution=0.10,
        origin_x=0.0,
        origin_y=0.0,
        width=40,
        height=12,
    )

    result = write_semantic_navigation_assets(
        evidence=evidence,
        metadata=metadata,
        row_direction=np.array([1.0, 0.0]),
        output_dir=tmp_path,
        min_longitudinal_support_ratio=0.50,
        min_width_m=0.30,
        min_length_m=2.0,
        navigation_clearance_radii_m=(0.10,),
    )

    assert result["navigation"]["layers"].base_map[4, 20] == UNKNOWN_VALUE
    assert (
        result["manifest"]["geometry_policy"]["promote_aisle_prior_to_static_free"]
        is False
    )
