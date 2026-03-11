"""2D bird's-eye occupancy grid built from a depth map.

Projects depth pixels onto the ground (XZ) plane, accumulates a persistent
hit/miss grid, extracts solid-surface contours with cv2.findContours, and
renders a floor-plan style map with outlines around every solid object.
"""

import cv2
import numpy as np

UNKNOWN = 0
FREE = 1
OCCUPIED = 2

# Decay: slowly forget old observations so the map adapts if things move
_DECAY_INTERVAL = 30        # frames between decay passes
_DECAY_AMOUNT = 1            # subtract from counts each pass


class OccupancyGrid2D:
    def __init__(self, size_m=10.0, resolution=0.05,
                 fx=320.0, fy=320.0, cx=None, cy=None,
                 min_height=-0.1, max_height=2.0,
                 min_depth=0.15, max_depth=6.0, step=2):
        self.resolution = resolution
        self.size_m = size_m
        self.grid_size = int(size_m / resolution)
        self.origin_col = self.grid_size // 2
        self.origin_row = self.grid_size - 1

        self.fx = fx
        self.fy = fy
        self._cx = cx
        self._cy = cy
        self.min_height = min_height
        self.max_height = max_height
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.step = step

        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self._hit = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self._miss = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self._frame_count = 0

        # Cached contour output (list of cv2 contours in grid coords)
        self.contours = []
        self.surface_rects = []  # list of (cx, cz_m, w_m, l_m, angle) in meters

    # ── public ──────────────────────────────────────────────────────────

    def update(self, depth_map, pose=None):
        """Integrate one depth frame. Always accumulates (persistent map).

        depth_map : uint16 ndarray (mm).
        pose      : (x, y, theta) world pose, or None (robot stays at origin).
        Returns the grid (uint8 2-D: 0=unknown, 1=free, 2=occupied).
        """
        if depth_map is None or depth_map.size == 0:
            return self.grid

        self._frame_count += 1

        occ_rows, occ_cols = self._project(depth_map, pose)
        robot_r, robot_c = self._robot_cell(pose)

        # Mark free cells via raycasting, occupied cells via hit counts
        self._raycast_free(robot_r, robot_c, occ_rows, occ_cols)
        self._rebuild_grid()

        # Periodic decay so map forgets stale data
        if self._frame_count % _DECAY_INTERVAL == 0:
            mask = self._hit > 0
            self._hit[mask] = np.maximum(0, self._hit[mask] - _DECAY_AMOUNT)
            mask = self._miss > 0
            self._miss[mask] = np.maximum(0, self._miss[mask] - _DECAY_AMOUNT)

        # Extract contours of all solid surfaces
        self._extract_contours()

        return self.grid

    def clear(self):
        self.grid[:] = UNKNOWN
        self._hit[:] = 0
        self._miss[:] = 0
        self.contours = []
        self.surface_rects = []
        self._frame_count = 0

    def render(self, tracked_objects=None, robot_pose=None):
        """Render a BGR floor-plan image with solid-surface contour outlines."""
        gs = self.grid_size
        img = np.full((gs, gs, 3), 30, dtype=np.uint8)     # dark background
        img[self.grid == FREE] = [50, 50, 50]               # explored free = dark gray
        img[self.grid == OCCUPIED] = [80, 80, 80]           # occupied fill = slightly lighter

        # Draw solid contour outlines (the main thing the user wants)
        if self.contours:
            cv2.drawContours(img, self.contours, -1, (0, 255, 0), 1)

        # Draw fitted rectangles for large surfaces
        for rect in self.surface_rects:
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            cv2.drawContours(img, [box], 0, (0, 220, 220), 1)

        # Robot marker
        rr, rc = self._robot_cell(robot_pose)
        cv2.circle(img, (rc, rr), 4, (0, 0, 255), -1)
        # Heading arrow
        if robot_pose is not None:
            _, _, theta = robot_pose
        else:
            theta = 0.0
        ar = int(rr - 8 * np.cos(theta))
        ac = int(rc + 8 * np.sin(theta))
        cv2.arrowedLine(img, (rc, rr), (ac, ar), (0, 0, 255), 2, tipLength=0.4)

        # YOLO-labeled tracked objects on top
        if tracked_objects:
            for obj in tracked_objects:
                self._draw_tracked(img, obj)

        return img

    def get_surface_list(self):
        """Return a JSON-serialisable list of detected solid surfaces.

        Each entry: {center_x, center_z, width, length, angle_deg, area_m2}
        all in metres relative to robot origin.
        """
        surfaces = []
        res = self.resolution
        for rect in self.surface_rects:
            (cc, cr), (w_px, h_px), angle = rect
            cx_m = (cc - self.origin_col) * res
            cz_m = (self.origin_row - cr) * res
            w_m = w_px * res
            h_m = h_px * res
            surfaces.append({
                'center_x': round(cx_m, 3),
                'center_z': round(cz_m, 3),
                'width': round(w_m, 3),
                'length': round(h_m, 3),
                'angle_deg': round(angle, 1),
                'area_m2': round(w_m * h_m, 4),
            })
        return surfaces

    # ── depth projection ────────────────────────────────────────────────

    def _project(self, depth_map, pose):
        dh, dw = depth_map.shape
        cx = self._cx if self._cx is not None else dw / 2.0
        cy = self._cy if self._cy is not None else dh / 2.0
        fx_d = self.fx * (dw / 640.0)
        fy_d = self.fy * (dh / 480.0)

        ys = np.arange(0, dh, self.step)
        xs = np.arange(0, dw, self.step)
        uu, vv = np.meshgrid(xs, ys)

        d = depth_map[vv, uu].astype(np.float32) / 1000.0
        valid = (d > self.min_depth) & (d < self.max_depth)

        Y = (vv.astype(np.float32) - cy) * d / fy_d
        valid &= (Y > self.min_height) & (Y < self.max_height)

        X = (uu.astype(np.float32)[valid] - cx) * d[valid] / fx_d
        Z = d[valid]

        if pose is not None:
            px, py, theta = pose
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            wx = px + X * cos_t - Z * sin_t
            wz = py + X * sin_t + Z * cos_t
            cols = (wx / self.resolution + self.origin_col).astype(np.int32)
            rows = (self.origin_row - wz / self.resolution).astype(np.int32)
        else:
            cols = (X / self.resolution + self.origin_col).astype(np.int32)
            rows = (self.origin_row - Z / self.resolution).astype(np.int32)

        in_bounds = (rows >= 0) & (rows < self.grid_size) & \
                    (cols >= 0) & (cols < self.grid_size)
        rows = rows[in_bounds]
        cols = cols[in_bounds]

        np.add.at(self._hit, (rows, cols), 1)
        return rows, cols

    def _robot_cell(self, pose):
        if pose is None:
            return self.origin_row, self.origin_col
        px, py, _ = pose
        c = int(px / self.resolution + self.origin_col)
        r = int(self.origin_row - py / self.resolution)
        return np.clip(r, 0, self.grid_size - 1), \
               np.clip(c, 0, self.grid_size - 1)

    # ── raycasting (vectorised Bresenham) ───────────────────────────────

    def _raycast_free(self, r0, c0, occ_rows, occ_cols):
        if len(occ_rows) == 0:
            return
        endpoints = np.column_stack((occ_rows, occ_cols))
        unique = np.unique(endpoints, axis=0)
        # Subsample to keep fast — 6000 rays is plenty for a clean map
        if len(unique) > 6000:
            idx = np.random.choice(len(unique), 6000, replace=False)
            unique = unique[idx]
        for r1, c1 in unique:
            self._bresenham_free(r0, c0, int(r1), int(c1))

    def _bresenham_free(self, r0, c0, r1, c1):
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r1 > r0 else -1
        sc = 1 if c1 > c0 else -1
        err = dc - dr
        r, c = r0, c0
        gs = self.grid_size
        while True:
            if r == r1 and c == c1:
                break
            if 0 <= r < gs and 0 <= c < gs:
                self._miss[r, c] += 1
            e2 = 2 * err
            if e2 > -dr:
                err -= dr
                c += sc
            if e2 < dc:
                err += dc
                r += sr

    def _rebuild_grid(self):
        occ = self._hit >= 3
        free = (self._miss >= 2) & ~occ
        self.grid[:] = UNKNOWN
        self.grid[free] = FREE
        self.grid[occ] = OCCUPIED

    # ── contour extraction ──────────────────────────────────────────────

    def _extract_contours(self):
        occ_mask = (self.grid == OCCUPIED).astype(np.uint8) * 255

        # Morphological close to connect nearby occupied cells into solid walls
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(occ_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Keep contours with meaningful area (> ~4 cells = 1cm^2 at 5cm res)
        min_area = 4.0
        self.contours = [c for c in contours if cv2.contourArea(c) >= min_area]

        # Fit minimum-area rotated rectangles to large surfaces
        min_rect_area = 20.0  # ~0.05 m^2
        self.surface_rects = []
        for c in self.contours:
            area = cv2.contourArea(c)
            if area >= min_rect_area and len(c) >= 5:
                rect = cv2.minAreaRect(c)
                self.surface_rects.append(rect)

    # ── drawing ─────────────────────────────────────────────────────────

    def _draw_tracked(self, img, obj):
        cx = int(obj['center_x'] / self.resolution + self.origin_col)
        cz = int(self.origin_row - obj['center_z'] / self.resolution)
        hw = max(2, int(obj['width'] / self.resolution / 2))
        hl = max(2, int(obj['length'] / self.resolution / 2))

        color = _class_color(obj.get('class', ''))
        r1, r2 = np.clip(cz - hl, 0, self.grid_size - 1), np.clip(cz + hl, 0, self.grid_size - 1)
        c1, c2 = np.clip(cx - hw, 0, self.grid_size - 1), np.clip(cx + hw, 0, self.grid_size - 1)
        cv2.rectangle(img, (c1, r1), (c2, r2), color, 1)

        label = obj.get('class', '')
        tid = obj.get('track_id')
        if tid is not None:
            label = f"#{tid} {label}"
        name = obj.get('name')
        if name:
            label = f"#{tid} {name}"
        dist = obj.get('distance')
        if dist is not None:
            label += f" {dist:.1f}m"
        cv2.putText(img, label, (c1, max(r1 - 3, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1)


def _class_color(cls):
    palette = {
        'person': (0, 180, 255), 'chair': (255, 120, 0),
        'couch': (180, 0, 255), 'bed': (0, 200, 120),
        'dining table': (200, 200, 0), 'tv': (255, 0, 0),
        'laptop': (0, 255, 200), 'wall': (200, 200, 200),
        'surface': (0, 220, 220),
    }
    return palette.get(cls, (0, 200, 0))
