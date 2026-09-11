# agt_pointcloud_map_tools consolidation

## Decision

The historical `Aldoubt/agt_pointcloud_map_tools` repository can be archived
after this consolidation. It should not remain an independently developed ROS 2
package.

## Audit summary

The repository contained only an architecture skeleton:

- `MapCleanPipeline::process()` was declared but never implemented;
- `map_cleaner_node` only started an empty ROS 2 node and logged one line;
- the YAML described voxel/outlier/cluster/dynamic-ghost stages, but none of
  those stages were connected to code;
- there was no subscription, publication, PCD I/O, test, benchmark, or map
  export implementation;
- CMake attempted to install a `launch/` directory that did not exist.

Therefore copying the C++/ROS package would add maintenance cost without
preserving meaningful implementation.

## What was retained

The useful design intent has been absorbed into the offline reconstruction
pipeline:

| Historical intent | New owner |
|---|---|
| PCD loading | `agt_map_reconstruction.io` |
| voxel cleaning | `preprocessing.cleaning` |
| radius outlier removal | `preprocessing.cleaning` |
| cluster-size filtering | `preprocessing.cleaning` |
| removed-point debug export | `tools/clean_pcd.py` |
| PCD -> navigation assets | existing map reconstruction pipeline |
| ground/elevation/traversability | existing algorithms/maps modules |

The old `dynamic_ghost_cleaner: enabled: true` setting was **not** copied as a
fake implementation. A single fused PCD is not sufficient evidence for
reliable human/vehicle ghost classification. Dynamic/static evidence belongs in
a temporal/offline benchmark that has registered frames or observation history.

## Replacement command

~~~bash
python -m pip install -e .
PYTHONPATH=src python tools/clean_pcd.py \
  --pcd /path/to/global_map.pcd \
  --config configs/preprocessing/outdoor_default.yaml \
  --output results/preprocessing/global_map_clean.pcd \
  --removed-output results/preprocessing/global_map_removed.pcd \
  --report results/preprocessing/cleaning_report.json
~~~

The cleaned PCD can then enter ground segmentation, elevation/traversability,
semantic reconstruction, and navigation-map validation.

## Repository lifecycle

Recommended lifecycle:

1. merge this consolidation after tests pass;
2. mark `agt_pointcloud_map_tools` as ARCHIVED;
3. optionally delete it later after confirming no external deployment scripts
   clone it directly.

Do not move ROS runtime filtering into this offline repository. Runtime
self-filter/crop/voxel/diagnostics belong in `agt_pointcloud_pipeline`.
