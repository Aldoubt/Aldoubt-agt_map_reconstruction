from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GridMetadata:
    resolution: float
    origin_x: float
    origin_y: float
    width: int
    height: int
    frame_id: str = "map"
    origin_yaw: float = 0.0

    def __post_init__(self):
        if self.resolution <= 0.0:
            raise ValueError("resolution must be > 0")

    def grid_to_world(self, x_cell, y_cell):
        local_x = (float(x_cell) + 0.5) * self.resolution
        local_y = (float(y_cell) + 0.5) * self.resolution
        c = math.cos(self.origin_yaw)
        s = math.sin(self.origin_yaw)
        return (
            self.origin_x + c * local_x - s * local_y,
            self.origin_y + s * local_x + c * local_y,
        )

    def world_to_grid(self, x_world, y_world):
        dx = float(x_world) - self.origin_x
        dy = float(y_world) - self.origin_y
        c = math.cos(self.origin_yaw)
        s = math.sin(self.origin_yaw)
        local_x = c * dx + s * dy
        local_y = -s * dx + c * dy
        return (
            int(math.floor(local_x / self.resolution)),
            int(math.floor(local_y / self.resolution)),
        )

    def to_dict(self):
        return {
            "frame_id": self.frame_id,
            "resolution": float(self.resolution),
            "origin": [float(self.origin_x), float(self.origin_y), float(self.origin_yaw)],
            "width": int(self.width),
            "height": int(self.height),
        }
