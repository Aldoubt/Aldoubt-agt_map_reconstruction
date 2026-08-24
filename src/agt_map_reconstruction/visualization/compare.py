"""Benchmark visualization utilities."""

from pathlib import Path
import matplotlib.pyplot as plt

from agt_map_reconstruction.maps.traversability_grid import points_to_grid, save_grid


def save_xy(points, path, title="", color="blue"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.scatter(points[:, 0], points[:, 1], s=0.3, c=color)
    plt.axis("equal")
    plt.title(title)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def save_overlay(result, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    if len(result["non_ground"]):
        plt.scatter(result["non_ground"][:, 0], result["non_ground"][:, 1], s=0.2, c="red", label="non-ground")
    if len(result["ground"]):
        plt.scatter(result["ground"][:, 0], result["ground"][:, 1], s=0.2, c="green", label="ground")
    plt.axis("equal")
    plt.legend()
    plt.title("segmentation overlay")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def save_segmentation(result, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_xy(result["ground"], out_dir / "ground.png", "ground", "green")
    save_xy(result["non_ground"], out_dir / "non_ground.png", "non ground", "red")
    save_overlay(result, out_dir / "overlay.png")

    grid = points_to_grid(result["ground"])
    save_grid(grid, out_dir / "ground_height_grid.png")
