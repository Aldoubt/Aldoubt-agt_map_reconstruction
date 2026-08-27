import numpy as np

from agt_map_reconstruction.maps.review_corrections import (
    apply_review_corrections,
    clip_aisle_to_scene,
)


def _rectangle():
    return {
        "aisle_id": 1,
        "label": "A01",
        "polygon_xy": [[2, 4], [18, 4], [18, 8], [2, 8]],
        "length_m": 1.6,
        "width_m": 0.4,
    }


def test_clip_aisle_trims_outside_end():
    scene = np.zeros((20, 20), dtype=bool)
    scene[:, 5:16] = True
    clipped, changed = clip_aisle_to_scene(_rectangle(), scene)
    assert changed is True
    points = np.asarray(clipped["polygon_xy"])
    assert points[:, 0].min() >= 5.0
    assert points[:, 0].max() <= 16.0


def test_review_promotes_nonhard_cells_but_keeps_pillar():
    labels = np.zeros((20, 20), dtype=np.uint8)
    labels[5, 10] = 6
    scene = np.ones_like(labels, dtype=bool)
    payload = {"rectangles": [_rectangle()]}
    review = {"aisles": {"A01": {"review_status": "pass", "reason": "debris"}}}
    corrected, _ = apply_review_corrections(labels, scene, payload, review)
    assert corrected[6, 10] == 1
    assert corrected[5, 10] == 6


def test_review_clips_ridges_without_dropping_them():
    labels = np.ones((20, 20), dtype=np.uint8)
    scene = np.zeros_like(labels, dtype=bool)
    scene[:, 5:16] = True
    ridge = {"label": "R01", "polygon_xy": [[1, 4], [19, 4], [19, 6], [1, 6]],
             "length_m": 1.8, "width_m": 0.2}
    payload = {"rectangles": [_rectangle()], "ridge_rectangles": [ridge]}
    corrected, updated = apply_review_corrections(labels, scene, payload,
                                                   {"aisles": {}})
    assert len(updated["ridge_rectangles"]) == 1
    points = np.asarray(updated["ridge_rectangles"][0]["polygon_xy"])
    assert points[:, 0].min() >= 5.0
    assert points[:, 0].max() <= 16.0
