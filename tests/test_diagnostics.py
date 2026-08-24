import json

import numpy as np

from agt_map_reconstruction.maps.diagnostics import (
    evidence_counts,
    percentile_summary,
    scan_inflation,
    summarize_run,
)
from agt_map_reconstruction.maps.ground_evidence import EvidenceClass


def test_percentile_summary_ignores_nan_and_reports_empty_rasters():
    summary = percentile_summary(np.array([[np.nan, 1.0, 3.0]]))
    assert summary["p50"] == 2.0
    assert percentile_summary(np.full((2, 2), np.nan))["p50"] is None


def test_evidence_counts_uses_stable_labels():
    evidence = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    assert evidence_counts(evidence) == {
        "unknown": 1,
        "free_confirmed": 1,
        "occupied_confirmed": 1,
        "ground_interpolated": 1,
    }


def test_scan_inflation_is_metric_and_reports_component_loss():
    evidence = np.full((5, 5), EvidenceClass.FREE_CONFIRMED, dtype=np.uint8)
    evidence[2, 2] = EvidenceClass.OCCUPIED_CONFIRMED
    rows = scan_inflation(evidence, resolution=1.0, radii_m=[0, 1.5])
    assert rows[0]["free_cells"] == 24
    assert rows[1]["free_cells"] == 16
    assert rows[1]["largest_free_component"] == 16


def test_summarize_run_reads_existing_arrays_without_pcd(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    np.save(run_dir / "low_height.npy", np.array([[1.0, np.nan]]))
    np.save(run_dir / "ground_surface.npy", np.array([[1.0, 1.0]]))
    np.save(run_dir / "clearance.npy", np.array([[0.0, -0.1]]))
    np.save(run_dir / "evidence.npy", np.array([[1, 0]], dtype=np.uint8))
    (run_dir / "metadata.yaml").write_text("grid_resolution_m: 0.5\n")
    summary = summarize_run(run_dir, radii_m=[0])
    assert summary["negative_clearance_below_-0.05m"]["cells"] == 1
    assert summary["inflation_scan"][0]["radius_m"] == 0.0
