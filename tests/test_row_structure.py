import numpy as np

from agt_map_reconstruction.maps.row_structure import RowStructureConfig, analyze_row_structure


def test_row_structure_recovers_rows_and_vehicle_fitting_aisle():
    height = np.zeros((80, 160), dtype=float)
    height[15:19, 10:150] = 0.5
    height[45:49, 10:150] = 0.5
    result = analyze_row_structure(
        height,
        config=RowStructureConfig(
            resolution=0.05,
            min_row_length_m=3.0,
            min_aisle_width_m=0.6,
            vehicle_width_m=0.6,
        ),
    )
    assert len(result["rows"]) == 2
    assert result["aisles"][0]["vehicle_width_fits"] is True
    assert result["aisle_mask"].any()


def test_row_structure_repairs_only_short_longitudinal_gaps():
    height = np.zeros((40, 100), dtype=float)
    height[8:12, :100] = 0.5
    height[28:32, :100] = 0.5
    height[16:24, 40:44] = np.nan
    result = analyze_row_structure(
        height,
        config=RowStructureConfig(resolution=0.05, repair_gap_m=0.3),
    )
    assert result["aisle_mask"].any()


def test_row_structure_accepts_an_external_obstacle_layer_for_walls():
    height = np.zeros((40, 100), dtype=float)
    height[8:12] = 0.5
    height[28:32] = 0.5
    obstacles = np.zeros_like(height, dtype=bool)
    obstacles[14:28, 50:53] = True
    result = analyze_row_structure(
        height,
        config=RowStructureConfig(resolution=0.05, wall_min_length_m=0.5),
        obstacle_grid=obstacles,
    )
    assert result["walls"]
