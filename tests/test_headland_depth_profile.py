import copy

import numpy as np
import pytest

from agt_map_reconstruction.maps.headland_depth_profile import (
    build_headland_depth_profile,
)


def _fused_bundle():
    return {
        "row_axis_direction": [1.0, 0.0],
        "cross_row_direction": [0.0, 1.0],
        "resolution_m": 0.5,
        "lattice_rows": [
            {
                "label": "L1",
                "polygon_xy": [[0, 2], [29, 2], [29, 3], [0, 3]],
                "source_centerline_xy": [[0, 2.5], [29, 2.5]],
            },
            {
                "label": "L2",
                "polygon_xy": [[0, 6], [29, 6], [29, 7], [0, 7]],
                "source_centerline_xy": [[0, 6.5], [29, 6.5]],
            },
            {
                "label": "L3",
                "polygon_xy": [[0, 10], [29, 10], [29, 11], [0, 11]],
                "source_centerline_xy": [[0, 10.5], [29, 10.5]],
            },
        ],
        "ridge_profiles": [
            {"ridge_id": "R12", "ridge_cross_span_cells": [3.0, 6.0]},
            {"ridge_id": "R23", "ridge_cross_span_cells": [7.0, 10.0]},
        ],
        "ridge_terminations": [
            {"ridge_id": "R12", "status": "ok", "evidence_source": "pgm_hard"},
            {
                "ridge_id": "R23",
                "status": "insufficient_structural_support",
                "evidence_source": "unresolved",
                "local_3d_structure_observed": True,
            },
        ],
    }


def _uncertainty():
    def side(intercept):
        return {
            "trend_status": "ok",
            "trend": {
                "slope_du_dv": 0.0,
                "intercept_u": float(intercept),
                "center_trend_only": True,
            },
            "abs_residual_m": {
                "p50": 0.5,
                "p90": 0.75,
                "p95": 1.0,
                "max": 1.5,
            },
        }

    return {
        "row_axis_direction": [1.0, 0.0],
        "cross_row_direction": [0.0, 1.0],
        "resolution_m": 0.5,
        "entry": side(10.0),
        "exit": side(20.0),
    }


def test_default_depth_bands_are_finite_opposite_and_exclude_unresolved():
    result, masks = build_headland_depth_profile(
        _fused_bundle(),
        _uncertainty(),
        grid_shape_yx=(14, 30),
        depth_edges_m=[0.0, 0.5, 1.0, 2.0, 4.0],
        uncertainty_quantile="p95",
    )

    assert result["depth_edges_m"] == [0.0, 0.5, 1.0, 2.0, 4.0]
    assert result["max_outward_depth_m"] == 4.0
    assert result["policy"]["physical_site_boundary_required"] is False
    assert result["policy"]["navigation_map_modified"] is False
    assert result["policy"]["semantic_promotion"] is False

    # resolution=0.5 m/cell, entry center=10 cells and p95=1 m=2 cells.
    # Entry depth zero is x=8. 0-0.5 m selects the first outward cell x=7.
    assert masks["entry_depth_0_0p5"][4, 7]
    assert not masks["entry_depth_0_0p5"][4, 8]
    # Exit center=20 cells: outer uncertainty edge is x=22; first outward cell x=23.
    assert masks["exit_depth_0_0p5"][4, 23]
    assert not masks["exit_depth_0_0p5"][4, 22]

    # The 2-4 m entry band is half-open: x=1 is 3.5 m outward, x=0 is exactly 4.0 m.
    assert masks["entry_depth_2_4"][4, 1]
    assert not masks["entry_depth_2_4"][4, 0]
    assert not np.any(masks["entry_depth_2_4"][:, 8:])
    # Exit 2-4 m is clipped by the finite map, never extended via UNKNOWN semantics.
    assert masks["exit_depth_2_4"][4, 27]

    assert not np.any(masks["entry_depth_0_0p5"] & masks["exit_depth_0_0p5"])
    assert not np.any(
        masks["entry_depth_0_0p5"] & masks["structurally_unresolved_cross"]
    )
    assert not np.any(
        masks["exit_depth_0_0p5"] & masks["structurally_unresolved_cross"]
    )

    # R23 occupies 7<=cross-v<=10. It must be excluded from every resolved band.
    assert masks["structurally_unresolved_cross"][8, 7]
    for name, mask in masks.items():
        if name == "structurally_unresolved_cross":
            continue
        assert not np.any(mask & masks["structurally_unresolved_cross"]), name


def test_depth_masks_are_pairwise_disjoint_and_record_band_metadata():
    result, masks = build_headland_depth_profile(
        _fused_bundle(),
        _uncertainty(),
        grid_shape_yx=(14, 30),
    )

    names = list(masks)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            assert not np.any(masks[first] & masks[second]), (first, second)

    entry_bands = result["entry"]["bands"]
    assert [item["mask_key"] for item in entry_bands] == [
        "entry_depth_0_0p5",
        "entry_depth_0p5_1",
        "entry_depth_1_2",
        "entry_depth_2_4",
    ]
    assert [(item["depth_min_m"], item["depth_max_m"]) for item in entry_bands] == [
        (0.0, 0.5),
        (0.5, 1.0),
        (1.0, 2.0),
        (2.0, 4.0),
    ]


def test_bad_depth_edges_are_rejected():
    for edges in ([0, 1, 0.5], [0, 0.5, 0.5], [0.5, 1.0], [0.0]):
        with pytest.raises(ValueError):
            build_headland_depth_profile(
                _fused_bundle(),
                _uncertainty(),
                grid_shape_yx=(14, 30),
                depth_edges_m=edges,
            )


def test_reversed_source_centerlines_do_not_change_normalized_frozen_geometry():
    fused = _fused_bundle()
    reversed_sources = copy.deepcopy(fused)
    for row in reversed_sources["lattice_rows"]:
        row["source_centerline_xy"] = list(reversed(row["source_centerline_xy"]))
        row["polygon_xy"] = list(reversed(row["polygon_xy"]))

    result_a, masks_a = build_headland_depth_profile(
        fused,
        _uncertainty(),
        grid_shape_yx=(14, 30),
    )
    result_b, masks_b = build_headland_depth_profile(
        reversed_sources,
        _uncertainty(),
        grid_shape_yx=(14, 30),
    )

    assert result_a["row_axis_direction"] == result_b["row_axis_direction"]
    assert result_a["cross_row_direction"] == result_b["cross_row_direction"]
    assert masks_a.keys() == masks_b.keys()
    for name in masks_a:
        assert np.array_equal(masks_a[name], masks_b[name]), name
