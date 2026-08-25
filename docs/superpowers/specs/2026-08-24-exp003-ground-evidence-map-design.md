# EXP003 Ground Evidence Map Design

## Purpose

Recover a conservative, explainable global navigation grid from the fixed
FAST-LIVO2 `processed.pcd` before investing in rosbag raycasting. EXP003 must
separate measured free ground, occupied cells, interpolated ground, and unknown
space instead of collapsing all valid cells into a binary traversability mask.

## Scope

- Keep EXP002 unchanged as the agricultural-corridor experiment.
- Use the authoritative PCD; snapshot its stable file identity and record its
  absolute path, size, point count, optional SHA-256, exact Git state, grid
  origin, resolution, and all parameters. Reject publication if the PCD changes
  after snapshotting.
- Build robust per-cell elevation statistics in bounded memory.
- Estimate a continuous local ground surface only across bounded gaps.
- Derive obstacle evidence from height above the estimated ground.
- Produce a conservative navigation costmap with configurable obstacle
  inflation.
- Record rosbag raycasting as a future EXP004 feasibility probe, not as an
  implemented capability or current baseline.

## Representation

The evidence grid uses stable integer labels:

- `0 UNKNOWN`: insufficient evidence or an unsupported gap.
- `1 FREE_CONFIRMED`: measured low surface consistent with the ground model.
- `2 OCCUPIED_CONFIRMED`: measured structure above the obstacle threshold.
- `3 GROUND_INTERPOLATED`: a bounded hole supported by surrounding ground;
  retained separately and not silently described as measured free space.

The navigation costmap uses `uint8`: confirmed free is `0`, interpolated
ground has a configurable nonzero penalty, inflated/occupied cells are `254`,
and unknown cells are `255`.

## Pipeline

1. Rasterize finite XYZ points in chunks. For each cell retain point count,
   minimum Z, and a robust low-height estimate derived from a fixed-bin lower
   histogram after the global grid bounds are known.
2. Reject cells below `min_points_per_cell`; use the low-height estimate as the
   measured surface.
3. Seed ground from the global `ground_seed_percentile` low-height envelope and
   propagate support through measured neighbors only when their height step is
   at most `max_ground_step_m`. Estimate ground with a NaN-aware low-percentile
   filter over propagated ground in a metric window. Exclude the target cell
   from its own model and require at least `min_ground_support_cells` distinct
   propagated-ground neighbors; otherwise the model is unsupported.
4. Close only small enclosed unsupported components. Reject components that
   touch the grid edge, exceed `max_interpolation_gap_m`, have a non-ground
   boundary, or lack local multi-side support. Every interpolation vertex must
   lie within the metric gap bound of the target cell.
5. Classify measured cells by height above independently supported ground and
   preserve unsupported cells as unknown. Interpolated cells receive their own
   label.
6. Inflate confirmed obstacles by the configured robot safety radius and emit
   the navigation costmap.

## Outputs

Every immutable `results/EXP003/<run_id>/` contains:

- `metadata.yaml`, `metrics.yaml`
- `low_height.npy`, `ground_surface.npy`, `clearance.npy`
- `point_count.npy`, `evidence.npy`, `costmap.npy`
- PNG previews for low height, ground surface, clearance, evidence, and costmap

NumPy arrays are authoritative numeric artifacts; PNGs are inspection aids.

## Validation

Synthetic tests must cover robust low-height behavior, sparse-noise rejection,
low-envelope seeded and step-constrained ground support, bounded
component-local interpolation, rejection
of sparse/open/oversized gaps, obstacle preservation, unknown preservation,
inflation, coordinate origin, stable PCD identity, immutable output
directories, and artifact metadata. A small synthetic end-to-end PCD smoke
run is required. The real 85M-point PCD run remains an operator-side acceptance
step and must not be claimed complete in the cloud workspace.

## Deferred EXP004

A later probe may replay a short rosbag segment using time-aligned raw LiDAR
frames and FAST-LIVO2 poses for raycasting and log-odds fusion. It will compare
the same local regions against EXP003 and will only become a baseline if it
demonstrates useful gains without pose-error obstacle carving.
