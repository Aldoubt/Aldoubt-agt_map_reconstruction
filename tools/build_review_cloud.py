#!/usr/bin/env python3
"""Build a denser, bounded NumPy point cache for interactive review."""

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def _sample_binary_pcd(path, max_points, bounds):
    """Read only xyz from a binary float PCD using a memory map."""
    header = []
    with path.open("rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError("PCD header has no DATA line")
            header.append(line)
            if line.startswith(b"DATA"):
                data_offset = stream.tell()
                break
    fields = next(line.decode().split()[1:] for line in header if line.startswith(b"FIELDS"))
    sizes = [int(v) for line in header if line.startswith(b"SIZE") for v in line.decode().split()[1:]]
    types = [v for line in header if line.startswith(b"TYPE") for v in line.decode().split()[1:]]
    points_count = int(next(line.decode().split()[1] for line in header if line.startswith(b"POINTS")))
    if fields[:3] != ["x", "y", "z"] or sizes[:3] != [4, 4, 4] or types[:3] != ["F", "F", "F"]:
        raise ValueError("only float32 xyz-leading binary PCD is supported")
    if any(size != 4 or kind != "F" for size, kind in zip(sizes, types)):
        raise ValueError("PCD fields must all be float32")
    raw = np.memmap(path, mode="r", dtype="<f4", offset=data_offset,
                    shape=(points_count, len(fields)))
    stride = max(1, int(np.ceil(points_count / max_points)))
    points = np.asarray(raw[::stride, :3], dtype=np.float32)
    if bounds is not None:
        xmin, xmax, ymin, ymax = bounds
        keep = ((points[:, 0] >= xmin) & (points[:, 0] <= xmax)
                & (points[:, 1] >= ymin) & (points[:, 1] <= ymax))
        points = points[keep]
    return points


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcd", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--voxel-size", type=float, default=0.04)
    parser.add_argument("--max-points", type=int, default=8_000_000)
    parser.add_argument("--xmin", type=float)
    parser.add_argument("--xmax", type=float)
    parser.add_argument("--ymin", type=float)
    parser.add_argument("--ymax", type=float)
    args = parser.parse_args(argv)
    if args.voxel_size <= 0 or args.max_points <= 0:
        raise ValueError("voxel-size and max-points must be positive")
    bounds = [args.xmin, args.xmax, args.ymin, args.ymax]
    if any(value is not None for value in bounds):
        if not all(value is not None for value in bounds):
            raise ValueError("xmin/xmax/ymin/ymax must be supplied together")
    points = _sample_binary_pcd(args.pcd, args.max_points, bounds if any(v is not None for v in bounds) else None)
    # Keep a small voxel pass after streaming sample to suppress duplicate
    # points while avoiding a full 85M-point Open3D allocation.
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud = cloud.voxel_down_sample(args.voxel_size)
    points = np.asarray(cloud.points, dtype=np.float32)
    if len(points) > args.max_points:
        # Deterministic spatially spread subsampling rather than taking a
        # contiguous prefix of the cloud.
        indices = np.linspace(0, len(points) - 1, args.max_points, dtype=np.int64)
        points = points[indices]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, points, allow_pickle=False)
    metadata = {
        "source": str(args.pcd.resolve()),
        "source_size": args.pcd.stat().st_size,
        "voxel_size_m": args.voxel_size,
        "point_count": int(len(points)),
        "bounds_xyz": np.vstack((points.min(axis=0), points.max(axis=0))).tolist(),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
