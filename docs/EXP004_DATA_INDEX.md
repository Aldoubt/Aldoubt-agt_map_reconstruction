# EXP004 uploaded result index

This index lists the lightweight generated artifacts uploaded with the
`feat/exp004-reviewed-v3` branch. Large binary layers (`.pgm`, `.npy`, `.ply`)
remain local and can be regenerated from the commands in `docs/EXPERIMENTS.md`.

## Latest reviewed result

`results/EXP004/navigation-map-reviewed-v3/`

- `aisle_rectangles.json`: corrected aisle/ridge/wall geometry and map frame.
- `navigation_base_map.yaml`: resolution and map origin metadata.
- `validation.json`: static-map semantic and clearance validation.
- `review_corrections.json`: applied local manual corrections.

`results/EXP004/smooth-lateral-route-reviewed-v3/`

- `smooth_route_search.json` / `.csv`: strict route metrics for all 20 aisles.
- `smooth_route_overlay.png`: all-aisle route overview.
- `A01_`, `A03_`, `A06_`, `A11_`, `A17_`, `A18_...png`: critical aisle overlays.

`results/EXP004/pcd-route-review-reviewed-v3/`

- `pcd_semantic_overlay.json`: PCD overlay metadata.
- `aisle_rectangles_world.json`: world-coordinate semantic geometry.
- `pcd_semantic_overlay_top.png`: PCD/semantic top-view review.

## Baselines

- `robot-footprint-v1/aisle_footprint_validation.json` / `.csv`: strict
  centerline footprint baseline.
- `in-aisle-route-search-v1/aisle_offset_search.json` / `.csv`: constant
  lateral-offset baseline.
- `smooth-lateral-route-v1/smooth_route_search.json` / `.csv`: unreviewed B2
  comparison result.
