import json

from tools.compare_ray_sampling_experiments import collect_run


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sweep(entry_unknown, exit_unknown):
    return {
        "thresholds": [
            {
                "min_support_rays": 1,
                "sides": {
                    "entry": {
                        "supported_unknown_cell_count": entry_unknown,
                        "supported_unknown_fraction_of_roi_unknown": 0.01,
                        "component_count": 2,
                        "largest_component_cell_count": 3,
                        "raw_supported_cross_row_span_fraction": 0.5,
                        "raw_supported_max_outward_depth_m": 0.7,
                        "new_strict_safe_cell_count": 1,
                    },
                    "exit": {
                        "supported_unknown_cell_count": exit_unknown,
                        "supported_unknown_fraction_of_roi_unknown": 0.02,
                        "component_count": 4,
                        "largest_component_cell_count": 5,
                        "raw_supported_cross_row_span_fraction": 0.6,
                        "raw_supported_max_outward_depth_m": 0.8,
                        "new_strict_safe_cell_count": 2,
                    },
                },
            }
        ]
    }


def test_collect_run_keeps_ray_and_scan_support_separate(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    _write(
        run / "streaming_observation_evidence_manifest.json",
        {
            "streaming": {"scan_stride": 10, "export_point_stride": 20},
            "summary": {
                "selected_scan_count": 623,
                "sampled_point_count": 445686,
                "pose_supported_ray_count": 432029,
                "pose_rejected_before_trajectory": 851,
                "pose_rejected_after_trajectory": 808,
                "pose_rejected_large_gap": 11998,
                "ray_supported_cell_count": 210439,
                "scan_supported_cell_count": 210439,
                "max_scan_support_count": 18,
            },
        },
    )
    _write(run / "ray_support_sweep.json", _sweep(627, 416))
    _write(run / "scan_support_sweep.json", _sweep(59, 161))

    result = collect_run("baseline", run)

    assert result["scan_stride"] == 10
    assert result["export_point_stride"] == 20
    assert result["max_scan_support_count"] == 18
    ray_entry = next(
        row for row in result["rows"]
        if row["support_basis"] == "ray" and row["side"] == "entry"
    )
    scan_entry = next(
        row for row in result["rows"]
        if row["support_basis"] == "scan" and row["side"] == "entry"
    )
    assert ray_entry["supported_unknown"] == 627
    assert scan_entry["supported_unknown"] == 59


def test_collect_run_requires_common_thresholds(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    _write(
        run / "streaming_observation_evidence_manifest.json",
        {
            "streaming": {"scan_stride": 10, "export_point_stride": 20},
            "summary": {
                "selected_scan_count": 1,
                "sampled_point_count": 1,
                "pose_supported_ray_count": 1,
                "pose_rejected_before_trajectory": 0,
                "pose_rejected_after_trajectory": 0,
                "pose_rejected_large_gap": 0,
            },
        },
    )
    _write(run / "ray_support_sweep.json", {"thresholds": [{"min_support_rays": 1, "sides": {}}]})
    _write(run / "scan_support_sweep.json", {"thresholds": [{"min_support_rays": 2, "sides": {}}]})

    try:
        collect_run("bad", run)
    except ValueError as exc:
        assert "no common" in str(exc)
    else:
        raise AssertionError("expected ValueError")
