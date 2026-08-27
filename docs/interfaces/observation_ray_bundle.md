# Observation Ray Bundle v1

This interface is the frozen ROS-independent input contract for P1-E trajectory-aware observation evidence.

## Purpose

An aggregated PCD preserves return geometry but loses the sensor origin of each return. Therefore it cannot distinguish observed-free line of sight from unobserved or occluded space. P1-E restores that missing observation provenance from time-aligned LiDAR rays and poses.

This interface does not define a headland label and does not modify the static PGM.

## File

`observation_rays.npz`

Required arrays are `schema_version=1`, `frame_id`, `ray_origin_xyz_m (N,3)`, and `ray_endpoint_xyz_m (N,3)`. Optional arrays are `timestamp_s (N,)` and `scan_index (N,)`. The file must be loadable with `np.load(..., allow_pickle=False)`.

## Geometry contract

1. Origin and endpoint at the same index belong to the same physical LiDAR ray.
2. Both are already transformed into the frozen map frame.
3. Deskew must be resolved before export when per-point timing is available.
4. The return endpoint is a hit and is never marked free by the P1-E accumulator.
5. A ray high above the local ground does not prove the ground below it is free. Support is accumulated only when the 3D ray segment lies in an explicitly configured low-height band above a separate geometry-only ground reference.
6. The P1-A semantic `ground_surface.npy` is intentionally NaN in unknown cells, so it cannot directly recover evidence in those same unknown areas.
7. P1-E0 therefore uses `ground_reference.npy`, initially an affine z(x,y) model fitted from finite confirmed ground support. Its extrapolated values are geometry references only, never semantic free evidence.
8. P1-E0 outputs support counts and masks only. `semantic_promotion` remains false.

## Ground reference baseline

Locate the original PCD-build `ground_surface.npy` and fit a geometry-only reference:

```bash
python tools/fit_ground_reference_plane.py \
  --ground-surface /path/to/original_build/ground_surface.npy \
  --grid-manifest /path/to/original_build/pipeline_manifest.json \
  --output results/P1/greenhouse_01_region_split/observation/ground_reference_plane
```

The output contains `ground_reference.npy`, `ground_reference_source_mask.npy`, and `ground_reference_manifest.json`. The manifest reports affine coefficients plus RMSE, median, p95, and max residuals. No residual threshold automatically promotes a cell.

## Source provenance requirements

A rosbag2 or log converter must record the bag/log path, point topic, pose source, LiDAR-to-body extrinsic source, timestamp convention and corrections, deskew policy, and any range/ROI/return filtering.

Do not reconstruct this bundle from the final registered aggregate PCD alone because the per-return sensor origin has already been lost.

## Current P1-E0 consumer

```bash
python tools/build_ground_aware_ray_evidence.py \
  --rays observation_rays.npz \
  --ground-reference results/P1/greenhouse_01_region_split/observation/ground_reference_plane/ground_reference.npy \
  --grid-manifest /path/to/pipeline_manifest.json \
  --output results/P1/greenhouse_01_region_split/observation/ray_evidence \
  --min-ground-relative-height-m <EXPLICIT_VALUE> \
  --max-ground-relative-height-m <EXPLICIT_VALUE> \
  --min-support-rays <EXPLICIT_VALUE>
```

The three evidence thresholds are intentionally explicit CLI inputs rather than paper-frozen defaults. They must be selected and sensitivity-tested once real trajectory-aware rays are available.
