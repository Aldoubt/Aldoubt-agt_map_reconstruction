import numpy as np

from agt_map_reconstruction.maps.row_lattice_completion import complete_row_lattice


def _region(source_id, center_v, width_cells, *, region_class="row_aisle", u0=0.0, u1=100.0):
    v0 = float(center_v - 0.5 * width_cells)
    v1 = float(center_v + 0.5 * width_cells)
    label = f"A{source_id:02d}" if region_class == "row_aisle" else f"O{source_id:02d}"
    return {
        "source_band_id": int(source_id),
        "source_band_label": f"A{source_id:02d}",
        "label": label,
        "region_class": region_class,
        "polygon_xy": [[u0, v0], [u1, v0], [u1, v1], [u0, v1]],
        "centerline_xy": [[u0, center_v], [u1, center_v]],
        "width_m": float(width_cells * 0.10),
        "length_m": float((u1 - u0) * 0.10),
        "heading_rad": 0.0,
    }


def test_wide_band_is_split_into_inferred_lattice_slots_without_free_promotion():
    regions = [
        _region(1, 10.0, 4.0),
        _region(2, 20.0, 4.0),
        _region(3, 30.0, 4.0),
        _region(4, 40.0, 4.0),
        _region(5, 60.0, 30.0, region_class="wide_open_area_candidate"),
    ]

    result = complete_row_lattice(
        regions,
        resolution_m=0.10,
        row_axis=[1.0, 0.0],
        min_observed_slots=4,
    )

    assert result["status"] == "ok"
    assert np.isclose(result["nominal_pitch_cells"], 10.0)
    observed = [s for s in result["slots"] if s["source"] == "observed_row_aisle"]
    inferred = [s for s in result["slots"] if s["source"] == "lattice_inferred_wide_band"]
    assert [round(s["center_v_cells"], 6) for s in observed] == [10.0, 20.0, 30.0, 40.0]
    assert [round(s["center_v_cells"], 6) for s in inferred] == [50.0, 60.0, 70.0]
    assert all(s["evidence_strength"] == "weak_inferred" for s in inferred)
    assert all(s["navigation_free_promoted"] is False for s in result["slots"])
    assert result["policy"]["navigation_map_modified"] is False
    assert result["policy"]["semantic_promotion"] is False


def test_inference_is_refused_when_stable_observed_lattice_is_insufficient():
    regions = [
        _region(1, 10.0, 4.0),
        _region(2, 20.0, 4.0),
        _region(3, 35.0, 30.0, region_class="wide_open_area_candidate"),
    ]

    result = complete_row_lattice(
        regions,
        resolution_m=0.10,
        row_axis=[1.0, 0.0],
        min_observed_slots=4,
    )

    assert result["status"] == "insufficient_observed_lattice"
    assert result["nominal_pitch_cells"] is None
    assert all(s["source"] == "observed_row_aisle" for s in result["slots"])
    assert result["policy"]["automatic_acceptance"] is False


def test_inferred_slot_geometry_uses_nominal_aisle_width_and_parent_longitudinal_span():
    regions = [
        _region(1, 10.0, 4.0, u0=5.0, u1=95.0),
        _region(2, 20.0, 4.0, u0=5.0, u1=95.0),
        _region(3, 30.0, 4.0, u0=5.0, u1=95.0),
        _region(4, 40.0, 4.0, u0=5.0, u1=95.0),
        _region(5, 55.0, 20.0, region_class="wide_open_area_candidate", u0=7.0, u1=91.0),
    ]

    result = complete_row_lattice(
        regions,
        resolution_m=0.10,
        row_axis=[1.0, 0.0],
        min_observed_slots=4,
    )
    inferred = [s for s in result["slots"] if s["source"] == "lattice_inferred_wide_band"]
    assert len(inferred) == 2
    slot = inferred[0]
    polygon = np.asarray(slot["polygon_xy"], dtype=float)
    assert np.isclose(np.ptp(polygon[:, 0]), 84.0)
    assert np.isclose(np.ptp(polygon[:, 1]), 4.0)
    assert slot["parent_region_class"] == "wide_open_area_candidate"
