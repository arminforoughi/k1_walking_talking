"""
2D occupancy map for obstacle avoidance.

Grid-based map with dead-reckoning pose; save/load to obstacle_map.npy + meta.

Coordinate system (world frame):
    X = forward (initial robot heading), Y = left, Z = up.
    Origin = where the robot started.
    ``robot_theta`` rotates the body-frame X axis relative to the world X axis.

Objects can be added in two ways:
    1. ``update_from_detections()`` — preferred — takes detection dicts that
       already carry ``floor_pos_m`` from :class:`ground_mapper.GroundMapper`,
       converts body-frame positions to world frame, and rasterizes them into
       the grid.
    2. ``add_object()`` / ``add_points()`` — legacy manual insertion in world
       coordinates.
"""

import json
import math
import threading
import numpy as np
import cv2


class ObstacleMap:
    """2D occupancy grid with exploration support.

    Three cell states tracked across two grids:
    - ``_grid``:    0 = unknown/free, 1 = occupied (obstacle)
    - ``_visited``: 0 = unseen, 1 = observed-free, 2 = robot-was-here

    Origin at map center (robot start).
    """

    UNSEEN = 0
    SEEN_FREE = 1
    ROBOT_VISITED = 2

    def __init__(self, size=400, meters_per_cell=0.05):
        self.size = size
        self.meters_per_cell = meters_per_cell
        self._grid = np.zeros((size, size), dtype=np.uint8)
        self._visited = np.zeros((size, size), dtype=np.uint8)
        self._lock = threading.Lock()
        self._origin_gx = size // 2
        self._origin_gy = size // 2
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0
        self._objects = []

    # ── Coordinate Helpers ────────────────────────────────────────────────

    def world_to_grid(self, x, y):
        """World (meters) to grid indices. Origin at center."""
        gx = self._origin_gx + int(round(x / self.meters_per_cell))
        gy = self._origin_gy - int(round(y / self.meters_per_cell))
        return gx, gy

    def grid_to_world(self, gx, gy):
        """Grid indices to world (meters)."""
        x = (gx - self._origin_gx) * self.meters_per_cell
        y = -(gy - self._origin_gy) * self.meters_per_cell
        return x, y

    def _body_to_world(self, bx, by):
        """Body-frame (x_fwd, y_left) → world-frame (wx, wy)."""
        c = math.cos(self.robot_theta)
        s = math.sin(self.robot_theta)
        wx = self.robot_x + bx * c - by * s
        wy = self.robot_y + bx * s + by * c
        return wx, wy

    # ── Point Insertion ───────────────────────────────────────────────────

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

    # ── Detection-Based Update (uses floor_pos_m from GroundMapper) ──────

    def update_from_detections(self, detections, merge_radius_m=0.5):
        """Ingest detections that carry ``floor_pos_m`` from GroundMapper.

        For each detection with a valid ``floor_pos_m``:
        1. Convert body-frame ``(x_fwd, y_left)`` → world frame.
        2. Add the object to the named-object list (merging nearby duplicates).
        3. Mark the grid cell as occupied.

        Parameters
        ----------
        detections : list of dicts with keys ``class``, ``bbox``,
            ``floor_pos_m`` (tuple or None), ``distance_m``, etc.
        merge_radius_m : merge into existing object if within this distance.
        """
        world_pts = []
        for det in detections:
            floor = det.get('floor_pos_m')
            if floor is None:
                continue
            bx, by = floor
            if not math.isfinite(bx) or not math.isfinite(by):
                continue
            wx, wy = self._body_to_world(bx, by)
            world_pts.append((wx, wy))

            label = det.get('name') or det['class']
            dist_m = det.get('distance_m')
            size_m = None
            if dist_m and dist_m > 0:
                x1, y1, x2, y2 = det['bbox']
                size_m = max(0.1, min(float((x2 - x1) * dist_m / 400.0), 3.0))

            self._merge_or_add_object(label, wx, wy, size_m, merge_radius_m)

        if world_pts:
            self.add_points(world_pts)

    def _merge_or_add_object(self, label, wx, wy, size_m, merge_radius):
        """Insert or merge a named object into ``_objects``."""
        for obj in self._objects:
            if obj['label'] == label:
                dx = obj['world_x'] - wx
                dy = obj['world_y'] - wy
                if math.sqrt(dx * dx + dy * dy) < merge_radius:
                    alpha = 0.3
                    obj['world_x'] = obj['world_x'] * (1 - alpha) + wx * alpha
                    obj['world_y'] = obj['world_y'] * (1 - alpha) + wy * alpha
                    if size_m is not None:
                        old = obj.get('size_m') or size_m
                        obj['size_m'] = round(old * (1 - alpha) + size_m * alpha, 2)
                    return
        self.add_object(label, wx, wy, size_m=size_m)

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

    def clear_all(self):
        """Reset grid, visited, and objects."""
        with self._lock:
            self._grid[:] = 0
            self._visited[:] = 0
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

    # ── Exploration Support ─────────────────────────────────────────────

    def mark_robot_cell(self):
        """Mark the cell under the robot as visited."""
        gx, gy = self.world_to_grid(self.robot_x, self.robot_y)
        if 0 <= gx < self.size and 0 <= gy < self.size:
            with self._lock:
                self._visited[gy, gx] = self.ROBOT_VISITED

    def mark_free_ray(self, angle_world, distance_m, step_m=None):
        """Mark cells as SEEN_FREE along a ray from the robot, stopping at
        the endpoint (which is presumably an obstacle or max range).

        Parameters
        ----------
        angle_world : absolute angle in world frame (radians).
        distance_m : how far the ray extends (metres).
        step_m : step size; defaults to ``meters_per_cell``.
        """
        if step_m is None:
            step_m = self.meters_per_cell
        n_steps = max(1, int(distance_m / step_m))
        dx = step_m * math.cos(angle_world)
        dy = step_m * math.sin(angle_world)
        with self._lock:
            for i in range(n_steps):
                x = self.robot_x + i * dx
                y = self.robot_y + i * dy
                gx, gy = self.world_to_grid(x, y)
                if gx < 0 or gx >= self.size or gy < 0 or gy >= self.size:
                    break
                if self._grid[gy, gx] == 0:
                    if self._visited[gy, gx] == self.UNSEEN:
                        self._visited[gy, gx] = self.SEEN_FREE

    def mark_free_fan(self, depth_zones, fov_rad=1.2):
        """Mark free space from a set of depth zones (as returned by
        ``_get_depth_zones``).  Each zone is a wedge of the camera FOV.

        Parameters
        ----------
        depth_zones : list of distances (metres), left-to-right.
        fov_rad : total horizontal field of view.
        """
        n = len(depth_zones)
        wedge = fov_rad / n
        start_angle = self.robot_theta + fov_rad / 2.0 - wedge / 2.0
        for i, d in enumerate(depth_zones):
            angle = start_angle - i * wedge
            dist = min(d, 5.0)
            if dist > 0.2:
                self.mark_free_ray(angle, dist)

    def score_heading(self, angle_world, max_range=3.0, step_m=None):
        """Score a candidate heading by how many *unseen* cells lie along it
        before hitting an obstacle or the map edge.

        Returns (unseen_count, free_count, obstacle_dist).
        """
        if step_m is None:
            step_m = self.meters_per_cell
        n_steps = int(max_range / step_m)
        dx = step_m * math.cos(angle_world)
        dy = step_m * math.sin(angle_world)
        unseen = 0
        free = 0
        obstacle_dist = max_range
        with self._lock:
            for i in range(1, n_steps + 1):
                x = self.robot_x + i * dx
                y = self.robot_y + i * dy
                gx, gy = self.world_to_grid(x, y)
                if gx < 0 or gx >= self.size or gy < 0 or gy >= self.size:
                    obstacle_dist = i * step_m
                    break
                if self._grid[gy, gx] > 0:
                    obstacle_dist = i * step_m
                    break
                v = self._visited[gy, gx]
                if v == self.UNSEEN:
                    unseen += 1
                else:
                    free += 1
        return unseen, free, obstacle_dist

    def best_frontier_heading(self, n_candidates=36, max_range=3.0):
        """Evaluate ``n_candidates`` evenly-spaced headings and return the one
        that maximises unseen cells while avoiding obstacles.

        Returns ``(best_angle_world, best_unseen, best_obstacle_dist)``.
        If the entire map is explored, returns the heading with the most
        free space (least recently visited).
        """
        step = 2.0 * math.pi / n_candidates
        best_angle = self.robot_theta
        best_score = -1.0
        best_unseen = 0
        best_obs = 0.0

        for i in range(n_candidates):
            angle = -math.pi + i * step
            unseen, free, obs_dist = self.score_heading(angle, max_range)
            # Penalise headings that are blocked close
            if obs_dist < 0.5:
                continue
            # Score: unseen cells + small bonus for obstacle distance
            score = unseen * 1.0 + obs_dist * 0.3
            if score > best_score:
                best_score = score
                best_angle = angle
                best_unseen = unseen
                best_obs = obs_dist

        return best_angle, best_unseen, best_obs

    def explored_fraction(self):
        """Return fraction of reachable area that has been seen."""
        with self._lock:
            seen = np.sum(self._visited > 0)
            occupied = np.sum(self._grid > 0)
        reachable = seen + occupied
        if reachable == 0:
            return 0.0
        return float(seen) / float(seen + max(1, np.sum(self._visited == 0)))

    def get_grid_for_display(self):
        """Return copy of grid (0-255) for debug image. Robot at center."""
        with self._lock:
            return (self._grid * 255).astype(np.uint8)

    def get_exploration_display(self):
        """Return a 3-channel image: red=obstacle, green=visited, dark=unseen."""
        with self._lock:
            img = np.zeros((self.size, self.size, 3), dtype=np.uint8)
            img[self._visited == self.SEEN_FREE] = (40, 80, 40)
            img[self._visited == self.ROBOT_VISITED] = (60, 120, 60)
            img[self._grid > 0] = (0, 0, 180)
            gx, gy = self.world_to_grid(self.robot_x, self.robot_y)
            if 0 <= gx < self.size and 0 <= gy < self.size:
                cv2.circle(img, (gx, gy), 3, (255, 255, 0), -1)
            return img
