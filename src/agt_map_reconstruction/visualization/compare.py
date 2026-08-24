"""Benchmark visualization utilities."""

from pathlib import Path
import matplotlib.pyplot as plt


def save_xy(points, path, title=""):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.scatter(points[:, 0], points[:, 1], s=0.3)
    plt.axis("equal")
    plt.title(title)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def save_segmentation(result, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_xy(result["ground"], out_dir / "ground.png", "ground")
    save_xy(result["non_ground"], out_dir / "non_ground.png", "non ground")
