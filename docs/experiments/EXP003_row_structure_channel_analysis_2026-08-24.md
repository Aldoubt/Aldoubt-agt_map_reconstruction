# EXP003 Row Structure and Channel Analysis

Date: 2026-08-24

## Purpose

This PCD-only iteration recovers geometric structure before navigation
planning. It uses the MK-mini bare-body envelope only as an offline width
screen and does not require a real vehicle route.

## Inputs

```text
PCD: /home/yangxuan/Aldoubt-agt_map_reconstruction/date/processed.pcd
height grid: results/unified_navigation_comparison_20260824_v4/pmf_ground_height.npy
obstacle grid: results/unified_navigation_comparison_20260824_v4/exp003/evidence.npy
grid: 0.05 m/cell, 912 x 797
row angle: 15 degrees (manual first-pass alignment)
```

## Method

`agt_map_reconstruction.maps.row_structure` performs row-frame rotation,
smoothed transverse profile peak detection, row start/end and width estimates,
pairwise aisle extraction, bounded along-row gap repair, and external occupied
layer overlay for wall/support candidates. Unknown cells are not filled across
rows. MK-mini width screening uses `vehicle_width_m = 0.60`.

The vehicle screen is a width-only candidate test. It does not yet validate the
full 0.840 m swept footprint or the 1.5 m turning-radius connector.

## Command

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src .venv/bin/python \
  tools/analyze_row_structure.py \
  results/unified_navigation_comparison_20260824_v4/pmf_ground_height.npy \
  --obstacle-grid results/unified_navigation_comparison_20260824_v4/exp003/evidence.npy \
  --output results/unified_navigation_comparison_20260824_v4/row_structure_final \
  --row-angle-deg 15 --resolution 0.05 --vehicle-width-m 0.60 \
  --ridge-height-threshold-m 0.03 --max-row-width-m 0.80
```

## Output paths

```text
results/unified_navigation_comparison_20260824_v4/row_structure_final/structure.json
results/unified_navigation_comparison_20260824_v4/row_structure_final/structure_overlay.png
results/unified_navigation_comparison_20260824_v4/row_structure_final/ridge_mask.npy
results/unified_navigation_comparison_20260824_v4/row_structure_final/aisle_mask.npy
results/unified_navigation_comparison_20260824_v4/row_structure_final/wall_mask.npy
results/unified_navigation_comparison_20260824_v4/row_structure_final/filled_height.npy
```

## First-pass result

```text
accepted row candidates: 19
estimated row width mean: 0.324 m
estimated row width range: 0.100 .. 0.600 m
channel candidates: 17
channels meeting 0.600 m vehicle-width test: 17
wall/support candidates from occupied overlay: 51
```

These values are diagnostic, not accepted field measurements. The mean row
width is biased low because the current profile detects narrow high-return
bands rather than the full crop/soil envelope. Missed rows also produce
unrealistically wide pairwise channels. The overlay is useful for locating
failure regions, but it is not yet a route map.

## Interpretation

The PCD contains strong parallel structure. The lower vegetation region should
be repaired along the row axis rather than filled globally. Wall/support
detection needs a separate occupied or maximum-height layer; the PMF
ground-height raster alone cannot recover vertical supports reliably.

Next gate: stabilize row counting and spacing with explicit row-spacing bounds,
then apply the full MK-mini rectangle and minimum-turning-radius checks to each
channel.
