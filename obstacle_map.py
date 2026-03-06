"""
2D occupancy map for obstacle avoidance.
Grid-based map with dead-reckoning pose; save/load to obstacle_map.npy + meta.
"""

import json
import math
import threading
import numpy as np


class ObstacleMap:
    """2D occupancy grid. Origin at map center (robot start)."""

    def __init__(self, size=400, meters_per_cell=0.05):
        self.size = size
        self.meters_per_cell = meters_per_cell
        # Grid: 0 = free, 1 = occupied (or use counts for decay later)
        self._grid = np.zeros((size, size), dtype=np.uint8)
        self._lock = threading.Lock()
        # World origin in grid coords (center of map)
        self._origin_gx = size // 2
        self._origin_gy = size // 2
        # Robot pose in world frame (meters, radians)
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0
        # Named objects for distance reporting: list of {label, world_x, world_y}
        self._objects = []

    def world_to_grid(self, x, y):
        """World (meters) to grid indices. Origin at center."""
        gx = self._origin_gx + int(round(x / self.meters_per_cell))
        gy = self._origin_gy - int(round(y / self.meters_per_cell))  # y up in world -> grid row down
        return gx, gy

    def grid_to_world(self, gx, gy):
        """Grid indices to world (meters)."""
        x = (gx - self._origin_gx) * self.meters_per_cell
        y = -(gy - self._origin_gy) * self.meters_per_cell
        return x, y

    def add_points(self, world_xy):
        """Rasterize 2D points (Nx2 array or list of (x,y)) into grid."""
        if world_xy is None or len(world_xy) == 0:
            return
        arr = np.asarray(world_xy)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        gx = self._origin_gx + np.round(arr[:, 0] / self.meters_per_cell).astype(int)
        gy = self._origin_gy - np.round(arr[:, 1] / self.meters_per_cell).astype(int)
        valid = (gx >= 0) & (gx < self.size) & (gy >= 0) & (gy < self.size)
        gx, gy = gx[valid], gy[valid]
        with self._lock:
            self._grid[gy, gx] = 1

    def query_ray(self, robot_x, robot_y, theta, max_dist, num_steps=50):
        """
        Return minimum distance to an occupied cell along direction theta (radians).
        theta = 0 typically = forward. Returns (dist, hit) where hit is True if obstacle found.
        """
        step = max_dist / num_steps
        dx = step * math.cos(theta)
        dy = step * math.sin(theta)
        with self._lock:
            for i in range(1, num_steps + 1):
                x = robot_x + i * dx
                y = robot_y + i * dy
                gx, gy = self.world_to_grid(x, y)
                if gx < 0 or gx >= self.size or gy < 0 or gy >= self.size:
                    return (i * step, False)
                if self._grid[gy, gx] > 0:
                    return (i * step, True)
        return (max_dist, False)

    def query_along_direction(self, theta, max_dist=1.0):
        """Convenience: query from current robot pose."""
        return self.query_ray(self.robot_x, self.robot_y, self.robot_theta + theta, max_dist)

    def update_pose(self, dx, dy, dtheta, dt=0.05):
        """Update pose by velocity (vx, vy, vtheta) applied for dt. Body frame: x=forward, y=left."""
        c, s = math.cos(self.robot_theta), math.sin(self.robot_theta)
        self.robot_x += (dx * c - dy * s) * dt
        self.robot_y += (dx * s + dy * c) * dt
        self.robot_theta += dtheta * dt

    def set_pose(self, x, y, theta):
        """Set robot pose explicitly (e.g. reset)."""
        self.robot_x = x
        self.robot_y = y
        self.robot_theta = theta

    def add_object(self, label, world_x, world_y, size_m=None, shape=None, max_objects=80):
        """Add a named obstacle. size_m = optional estimated width/size in meters. shape = 'circle' or 'rectangle'."""
        self._objects.append({
            'label': label, 'world_x': world_x, 'world_y': world_y,
            'size_m': round(size_m, 2) if size_m is not None else None,
            'shape': shape if shape in ('circle', 'rectangle') else None,
        })
        while len(self._objects) > max_objects:
            self._objects.pop(0)

    def get_objects_with_distance(self):
        """Return list of {label, distance, bearing_deg, size_m, shape} relative to current pose."""
        rx, ry, rt = self.robot_x, self.robot_y, self.robot_theta
        out = []
        for o in self._objects:
            dx = o['world_x'] - rx
            dy = o['world_y'] - ry
            dist = math.sqrt(dx * dx + dy * dy)
            bearing = math.atan2(dy, dx) - rt
            bearing_deg = math.degrees(bearing)
            out.append({
                'label': o['label'],
                'distance': round(dist, 2),
                'bearing_deg': round(bearing_deg, 1),
                'size_m': o.get('size_m'),
                'shape': o.get('shape') or 'circle',
            })
        return out

    def get_objects_snapshot(self):
        """Thread-safe copy of objects list (for drawing labels on map)."""
        with self._lock:
            return list(self._objects)

    def clear_objects(self):
        """Clear named objects list (grid unchanged)."""
        self._objects.clear()

    def save(self, path_npy='obstacle_map.npy', path_meta='obstacle_map_meta.json'):
        with self._lock:
            np.save(path_npy, self._grid)
        meta = {
            'size': self.size,
            'meters_per_cell': self.meters_per_cell,
            'robot_x': self.robot_x,
            'robot_y': self.robot_y,
            'robot_theta': self.robot_theta,
        }
        with open(path_meta, 'w') as f:
            json.dump(meta, f, indent=2)

    def load(self, path_npy='obstacle_map.npy', path_meta='obstacle_map_meta.json'):
        try:
            grid = np.load(path_npy)
            if grid.shape != (self.size, self.size):
                self._grid = np.zeros((self.size, self.size), dtype=np.uint8)
                return False
            with self._lock:
                self._grid = grid
            with open(path_meta) as f:
                meta = json.load(f)
            self.robot_x = meta.get('robot_x', 0)
            self.robot_y = meta.get('robot_y', 0)
            self.robot_theta = meta.get('robot_theta', 0)
            return True
        except Exception:
            return False

    def get_grid_for_display(self):
        """Return copy of grid (0-255) for debug image. Robot at center."""
        with self._lock:
            return (self._grid * 255).astype(np.uint8)
