from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _save_array(array, path, title):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 8))
    plt.imshow(array, origin="lower")
    plt.title(title)
    plt.colorbar()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def save_grid_maps(maps, output):
    output = Path(output)
    _save_array(maps["height"], output / "height_map.png", "height")
    _save_array(maps["relative_height"], output / "relative_height.png", "relative height")
    _save_array(maps["traversability"], output / "traversability.png", "traversability")
