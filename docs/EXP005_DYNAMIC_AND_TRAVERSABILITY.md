# EXP005 — Dynamic Cleaning and Traversability Benchmarks

## Goal

Turn sparse LIO map products into navigation assets without mixing three different problems:

1. self geometry and robot-mounted interference;
2. dynamic-object contamination during mapping;
3. terrain traversability for the physical robot.

The production navigation repository should consume validated assets only. Algorithm experiments stay here.

## Pipeline

```text
raw/keyframe LiDAR scans + optimized poses
        |
        +-- self-geometry filter
        |
        +-- dynamic-map cleaning
        |     |- baseline: temporal/map-frame consistency
        |     |- ERASOR / Removert adapters
        |
        +-- ground segmentation benchmark
        |     |- existing height threshold
        |     |- existing PMF-inspired
        |     |- Patchwork++
        |
        +-- 2.5D terrain grid
        |     |- ground elevation
        |     |- observation/support count
        |     |- slope
        |     |- local step height
        |     |- roughness
        |     |- obstacle height over local ground
        |
        +-- robot capability model
        |     |- footprint
        |     |- max accepted slope
        |     |- max accepted step
        |     |- roughness threshold
        |     |- clearance
        |
        +-- traversability cost
        |
        +-- Nav2 PGM/YAML + semantic/cost layers
```

## Dynamic-object policy

Dynamic people must not be handled by one permanent rear-sector mask.

### Mapping/static asset

People and vehicles are removed only from the map-accumulation branch. The LIO
front-end keeps its time-preserving sensor contract unless a filter is separately
validated for odometry.

Preferred offline benchmark order:

1. temporal/map-frame consistency baseline;
2. ERASOR;
3. Removert;
4. optional semantic/instance method only if geometry-only cleaning is insufficient.

The cleaner input should be per-frame scan + optimized `T_map_lidar` pose. This
allows the static map to be regenerated after PGO instead of cleaning a drifted
naive accumulation.

### Runtime navigation

People remain obstacles in the local perception/costmap branch. Dynamic local
obstacles should decay, not be permanently burned into the global PGM.

## Sparse FAST-LIO map policy

Do not densify a sparse global PCD by inventing geometry.

Preferred source order for navigation-map generation:

1. raw/keyframe scans reprojected with optimized poses;
2. registered keyframe clouds;
3. final global PCD only as a fallback.

Sparse cells are represented as unknown until enough support is present. Small
holes may be interpolated only in the elevation layer when neighboring ground
support is consistent; obstacle geometry is never hallucinated.

## 2.5D traversability layers

For grid cell `c`:

- `elevation`: robust median/quantile of ground Z;
- `support`: number of independent scans/keyframes observing the cell;
- `slope`: local plane/gradient inclination;
- `step`: maximum local height discontinuity within a footprint-scaled window;
- `roughness`: robust residual/MAD around local ground fit;
- `obstacle_height`: non-ground height relative to local ground;
- `unknown`: insufficient evidence.

A first cost can be:

```text
C = max(
  slope_cost,
  step_cost,
  roughness_cost,
  obstacle_cost,
  unknown_cost
)
```

Robot footprint convolution/dilation is applied after cell metrics so a free
center cell is not considered traversable when the tracks collide with a nearby
obstacle.

## Benchmarks

### B0 — Lightning-LM g2p5 baseline

Use the g2p5 idea as a flat/near-flat reference:

- dynamic/fixed floor estimate;
- min/max height band relative to ground;
- grid resolution;
- ROS-compatible PGM.

This is a baseline, not the target traversability algorithm.

### B1 — Patchwork++ + AGT traversability grid

Patchwork++ separates ground/non-ground. AGT computes elevation, slope, step,
roughness and obstacle-relative-to-ground layers.

### B2 — elevation_mapping + traversability_estimation reference

Use as an architectural/reference benchmark for variance-aware elevation and
slope/step/roughness filters. Do not make the production stack depend on it until
Humble build/runtime stability is verified.

### B3 — Nav2 local terrain layer

Evaluate a ground/non-ground evidence layer with temporal decay for local
costmap use. This is complementary to the offline global map.

## Acceptance metrics

Dynamic cleaning:

- static preservation rate;
- dynamic rejection rate;
- map density retained;
- localization fitness before/after cleaning.

Traversability:

- known-area coverage;
- false-free rate at real obstacles/steps;
- false-occupied rate on traversable slopes;
- footprint collision rate;
- route connectivity;
- map resolution/runtime/memory.

## Initial Bunker defaults for experiments

Do not freeze physical capability thresholds from nominal specifications alone.
Measure on the real chassis.

The repository may use the current footprint geometry as an initial envelope,
but slope/step/roughness thresholds remain experiment parameters until field
validation.

## Repository boundary

- `agt_pointcloud_pipeline`: self geometry, sensor-level diagnostics, runtime filtering.
- `Aldoubt-agt_map_reconstruction`: dynamic map cleaning, ground/traversability benchmarks, map assets.
- `agt_navigation_v3`: consume validated static map and online local-costmap outputs; no benchmark code.
