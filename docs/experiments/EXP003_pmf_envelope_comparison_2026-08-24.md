# EXP003 PMF and MK-mini Envelope Comparison

Date: 2026-08-24

## Input

The comparison uses the repository-fixed PCD:

```text
path: /home/yangxuan/Aldoubt-agt_map_reconstruction/date/processed.pcd
points: 85,912,613
resolution: 0.05 m/cell
grid_shape_yx: 912 x 797
```

The PCD is not copied into experiment output. Its identity is recorded by the
canonical EXP003 runs; the source file remains the single input for reruns.

## Methods

### Tiled PMF

`agt_map_reconstruction.algorithms.morphological_pmf` now implements a 2.5D
progressive morphological filter rather than the former global Z histogram
threshold. It uses a minimum-height raster, progressively larger grayscale
morphological openings, and tiles with a morphology halo.

Configuration:

```text
resolution: 0.05 m
tile_size: 256 cells
max_window: 1.00 m
initial_distance: 0.05 m
slope: 0.20
point_height_threshold: 0.15 m
```

### EXP003

The comparison reference is the same-grid run:

```text
results/EXP003/20260824T121728Z_60062e6/
```

This is the low-height EXP003 baseline. The later Q90 experiment is retained
separately because it was intentionally conservative and blocked nearly all
free space.

### MK-mini body envelope

The vehicle feasibility preview uses the frozen offline envelope v0:

```text
reference: geometric_center
length: 0.840 m
width: 0.600 m
payload: excluded
safety_margin: 0.00 m
```

The envelope is applied as a rectangular footprint in the row-aligned map
frame. Unknown cells remain non-traversable. The 1.5 m minimum turning radius
is not evaluated by this raster check and remains a separate route-connector
constraint.

## Command

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONNOUSERSITE=1 \
  .venv/bin/python tools/run_unified_comparison.py \
  --pcd date/processed.pcd \
  --exp003-run results/EXP003/20260824T121728Z_60062e6 \
  --output results/unified_navigation_comparison_20260824 \
  --resolution 0.05 \
  --tile-size 256 \
  --max-window-m 1.0 \
  --safety-margin-m 0.0
```

## Produced artifacts

```text
results/unified_navigation_comparison_20260824/metrics.json
results/unified_navigation_comparison_20260824/pmf/evidence.npy
results/unified_navigation_comparison_20260824/pmf/vehicle_free.npy
results/unified_navigation_comparison_20260824/pmf/aisle_candidate.npy
results/unified_navigation_comparison_20260824/pmf/ground.png
results/unified_navigation_comparison_20260824/pmf/non_ground.png
results/unified_navigation_comparison_20260824/pmf/overlay.png
results/unified_navigation_comparison_20260824/exp003/evidence.npy
results/unified_navigation_comparison_20260824/exp003/vehicle_free.npy
results/unified_navigation_comparison_20260824/exp003/aisle_candidate.npy
```

The authoritative source arrays and previews for the EXP003 reference remain
in its run directory above. Generated `results/` content is ignored by Git.

## Results

| method | measured | free | occupied | unknown | MK-mini safe | aisle candidate |
|---|---:|---:|---:|---:|---:|---:|
| tiled PMF | 472,026 | 46,740 | 425,286 | 254,838 | 1,921 | 1,889 |
| EXP003 baseline | 462,001 | 270,877 | 191,117 | 264,863 | 33,119 | 31,134 |

The PMF result is substantially more conservative on this PCD. It is a valid
algorithmic comparison, not evidence that PMF is better: no point-level ground
truth or manually labelled ROI exists yet. The next decision must use the
successful MK-mini route as positive evidence and evaluate false blocking,
minimum envelope clearance, and aisle continuity.

## Verification

```text
117 tests passed
```

The PMF implementation has synthetic tests for raised-feature rejection,
tile-size invariance, and parameter validation. The comparison output was
generated successfully on all 85,912,613 PCD points.
