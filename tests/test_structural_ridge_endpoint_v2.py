import numpy as np

from agt_map_reconstruction.maps.structural_ridge_endpoint import (
    build_inter_aisle_ridge_profiles,
    detect_ridge_terminations,
    pair_aisle_structural_endpoints,
)


def _aisle(label, y0, y1, x0=2.0, x1=77.0):
    yc = 0.5 * (y0 + y1)
    return {
        "label": label,
        "region_class": "row_aisle",
        "polygon_xy": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        "centerline_xy": [[x0, yc], [x1, yc]],
    }


def test_inter_aisle_ridges_ignore_short_wall_and_recover_row_ends():
    free, hard = 254, 0
    base = np.full((40, 80), free, dtype=np.uint8)
    rows = [_aisle("A01", 4, 8), _aisle("A02", 14, 18), _aisle("A03", 24, 28)]

    # Short greenhouse wall-like evidence at the geometric boundary. It must not
    # become a row termination because it is not longitudinally persistent.
    base[:, 2:4] = hard

    # Two actual inter-aisle ridges with slightly different structural ends.
    base[9:14, 10:61] = hard
    base[19:24, 12:59] = hard

    # Fragment the first ridge internally. Endpoint detection must remain tied
    # to the boundary-anchored sustained row body, not choose a local longest run.
    base[9:14, 34:36] = free

    profiles = build_inter_aisle_ridge_profiles(
        base,
        rows,
        resolution_m=0.10,
        bin_size_m=0.10,
        row_axis=[1.0, 0.0],
    )
    assert len(profiles) == 2
    assert profiles[0]["left_aisle_label"] == "A01"
    assert profiles[0]["right_aisle_label"] == "A02"
    assert profiles[1]["left_aisle_label"] == "A02"
    assert profiles[1]["right_aisle_label"] == "A03"

    detected = [
        detect_ridge_terminations(
            profile,
            min_support_fraction=0.50,
            min_persistence_m=0.80,
            max_internal_gap_m=0.30,
        )
        for profile in profiles
    ]

    assert all(item["status"] == "ok" for item in detected)
    assert np.isclose(detected[0]["entry_u_cells"], 10.0, atol=1.0)
    assert np.isclose(detected[0]["exit_u_cells"], 61.0, atol=1.0)
    assert np.isclose(detected[1]["entry_u_cells"], 12.0, atol=1.0)
    assert np.isclose(detected[1]["exit_u_cells"], 59.0, atol=1.0)

    paired = pair_aisle_structural_endpoints(
        rows,
        detected,
        row_axis=[1.0, 0.0],
        max_side_endpoint_disagreement_m=0.50,
    )
    by_label = {item["label"]: item for item in paired}
    middle = by_label["A02"]
    assert middle["entry"]["status"] == "ok_bilateral"
    assert middle["exit"]["status"] == "ok_bilateral"
    assert np.isclose(middle["entry"]["structural_u_cells"], 11.0, atol=1.0)
    assert np.isclose(middle["exit"]["structural_u_cells"], 60.0, atol=1.0)


def test_outer_aisle_with_only_one_inter_aisle_ridge_stays_ambiguous():
    free, hard = 254, 0
    base = np.full((30, 60), free, dtype=np.uint8)
    rows = [_aisle("A01", 4, 8, 2, 57), _aisle("A02", 14, 18, 2, 57)]
    base[9:14, 10:50] = hard

    profiles = build_inter_aisle_ridge_profiles(
        base,
        rows,
        resolution_m=0.10,
        bin_size_m=0.10,
        row_axis=[1.0, 0.0],
    )
    detected = [
        detect_ridge_terminations(
            profile,
            min_support_fraction=0.50,
            min_persistence_m=0.80,
            max_internal_gap_m=0.20,
        )
        for profile in profiles
    ]
    paired = pair_aisle_structural_endpoints(
        rows,
        detected,
        row_axis=[1.0, 0.0],
        max_side_endpoint_disagreement_m=0.50,
    )
    assert paired[0]["entry"]["status"] == "ambiguous_single_side"
    assert paired[1]["exit"]["status"] == "ambiguous_single_side"
