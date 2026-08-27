# Observation Ray Bundle v1

This interface is the frozen ROS-independent input contract for P1-E trajectory-aware observation evidence.

## Purpose

An aggregated PCD preserves return geometry but loses the sensor origin of each return. Therefore it cannot distinguish observed-free line of sight from unobserved / occluded space. P1-E restores that missing observation provenance from time-aligned LiDAR rays and poses.

This interface does **not** define a headland label and does **not** modify the static PGM.

## File

```text
observation_rays.npz
```

The file must be readable with:

```python
np.load(path, allow_pickle=False)
```

Required arrays:

| key | shape | dtype | meaning |
| --- | --- | --- | --- |
| `schema_version` | scalar | integer | must equal `1` |
| `frame_id` | scalar | unicode | target frame, currently `map` |
| `ray_origin_xyz_m` | `(N,3)` | float | LiDAR optical origin for each return, expressed in `frame_id` |
| `ray_endpoint_xyz_m` | `(N,3)` | float | corresponding first valid LiDAR return, expressed in `frame_id` |

Optional arrays:

| key | shape | dtype | meaning |
| --- | --- | --- | --- |
| `timestamp_s` | `(N,)` | float | per-return timestamp after the source time base has been resolved; non-decreasing |
| `scan_index` | `(N,)` | integer | non-negative source scan/frame index |

## Geometry contract

1. `ray_origin_xyz_m[i]` and `ray_endpoint_xyz_m[i]` belong to the same physical LiDAR ray.
2. Both arrays are already transformed into the frozen map frame. No TF lookup is performed by the offline benchmark.
3. Motion distortion / deskew must be resolved before the bundle is written when the source driver exposes per-point timing.
4. A return endpoint is a hit, not free-space evidence. The P1-E ray accumulator therefore never marks the endpoint cell free.
5. A ray that passes high above the local ground surface does not prove the ground below it is free. P1-E requires a finite `ground_surface.npy` reference and only accumulates support when the 3D ray segment lies inside an explicitly configured low-height band above that surface.
6. Cells without a finite ground reference remain unsupported.
7. The first P1-E implementation produces `ray_free_support_count.npy` and `ray_free_support_mask.npy` only. `semantic_promotion` remains `false` until a separate replay validates how this evidence should be fused.

## Source provenance requirements

A converter from rosbag2 (or another log format) must preserve a sidecar manifest recording at least:

- bag/log path and immutable identifier when available;
- LiDAR point topic;
- pose / odometry source used to place each ray origin in map;
- LiDAR-to-body extrinsic source;
- timestamp convention and any offset correction;
- whether points were deskewed;
- any range, ROI, return-type, or scan filtering applied before export.

Do **not** reconstruct this bundle from the final registered aggregate PCD alone: the per-return sensor origin has already been lost.

## Current P1-E0 consumer

```bash
python tools/build_ground_aware_ray_evidence.py \
  --rays observation_rays.npz \
  --ground-surface /path/to/ground_surface.npy \
  --grid-manifest /path/to/pipeline_manifest.json \
  --output results/P1/.../observation_evidence \
  --min-ground-relative-height-m <MEASURED_OR_EXPLICIT_VALUE> \
  --max-ground-relative-height-m <MEASURED_OR_EXPLICIT_VALUE> \
  --min-support-rays <EXPLICIT_VALUE>
```

The three evidence thresholds are intentionally explicit CLI inputs rather than paper-frozen defaults. They must be selected and sensitivity-tested once real trajectory-aware rays are available.
