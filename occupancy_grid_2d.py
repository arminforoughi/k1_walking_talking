"""2D bird's-eye occupancy grid built from a depth map.

Projects depth pixels onto the ground (XZ) plane, accumulates a persistent
hit/miss grid, extracts wall lines via Hough transform, and renders a clean
floor-plan style map with straight lines for walls and surfaces.
"""

import cv2
import numpy as np

UNKNOWN = 0
FREE = 1
OCCUPIED = 2

_DECAY_INTERVAL = 30
_DECAY_AMOUNT = 1


class OccupancyGrid2D:
    def __init__(self, size_m=10.0, resolution=0.05,
                 fx=320.0, fy=320.0, cx=None, cy=None,
                 min_height=-0.1, max_height=2.0,
                 min_depth=0.15, max_depth=4.0, step=2):
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

        # Extracted geometry
        self.wall_lines = []      # list of ((x1,y1),(x2,y2)) in grid coords
        self.surface_rects = []   # list of cv2.minAreaRect for bounded objects

    # ── public ──────────────────────────────────────────────────────────

    def update(self, depth_map, pose=None):
        if depth_map is None or depth_map.size == 0:
            return self.grid

        self._frame_count += 1

        occ_rows, occ_cols = self._project(depth_map, pose)
        robot_r, robot_c = self._robot_cell(pose)
        self._raycast_free(robot_r, robot_c, occ_rows, occ_cols)
        self._rebuild_grid()

        if self._frame_count % _DECAY_INTERVAL == 0:
            mask = self._hit > 0
            self._hit[mask] = np.maximum(0, self._hit[mask] - _DECAY_AMOUNT)
            mask = self._miss > 0
            self._miss[mask] = np.maximum(0, self._miss[mask] - _DECAY_AMOUNT)

        self._extract_geometry()
        return self.grid

    def clear(self):
        self.grid[:] = UNKNOWN
        self._hit[:] = 0
        self._miss[:] = 0
        self.wall_lines = []
        self.surface_rects = []
        self._frame_count = 0

    def render(self, tracked_objects=None, robot_pose=None):
        gs = self.grid_size
        img = np.full((gs, gs, 3), 25, dtype=np.uint8)
        img[self.grid == FREE] = [45, 45, 45]
        img[self.grid == OCCUPIED] = [70, 70, 70]

        for (x1, y1), (x2, y2) in self.wall_lines:
            cv2.line(img, (x1, y1), (x2, y2), (0, 220, 0), 2)

        # Robot marker
        rr, rc = self._robot_cell(robot_pose)
        cv2.circle(img, (rc, rr), 4, (0, 0, 255), -1)
        theta = robot_pose[2] if robot_pose is not None else 0.0
        ar = int(rr - 10 * np.cos(theta))
        ac = int(rc + 10 * np.sin(theta))
        cv2.arrowedLine(img, (rc, rr), (ac, ar), (0, 0, 255), 2, tipLength=0.35)

        if tracked_objects:
            for obj in tracked_objects:
                self._draw_tracked(img, obj)

        return img

    def get_surface_list(self):
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
        # Add wall lines as entries too
        for (x1, y1), (x2, y2) in self.wall_lines:
            cx_m = ((x1 + x2) / 2.0 - self.origin_col) * res
            cz_m = (self.origin_row - (y1 + y2) / 2.0) * res
            length_px = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            length_m = length_px * res
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            surfaces.append({
                'center_x': round(cx_m, 3),
                'center_z': round(cz_m, 3),
                'width': round(length_m, 3),
                'length': round(res, 3),
                'angle_deg': round(angle, 1),
                'area_m2': round(length_m * res, 4),
                'type': 'wall',
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
        return int(np.clip(r, 0, self.grid_size - 1)), \
               int(np.clip(c, 0, self.grid_size - 1))

    # ── raycasting ──────────────────────────────────────────────────────

    def _raycast_free(self, r0, c0, occ_rows, occ_cols):
        if len(occ_rows) == 0:
            return
        endpoints = np.column_stack((occ_rows, occ_cols))
        unique = np.unique(endpoints, axis=0)
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

    # ── geometry extraction (clean straight wall lines) ────────────────

    def _extract_geometry(self):
        occ_mask = (self.grid == OCCUPIED).astype(np.uint8) * 255

        # Heavy morphological cleanup: close gaps then dilate to merge
        # nearby occupied cells into solid wall bands
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(occ_mask, cv2.MORPH_CLOSE, k_close, iterations=2)
        k_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.dilate(closed, k_dilate, iterations=1)

        # Skeletonize to thin walls to single-pixel lines before Hough
        skeleton = cv2.ximgproc.thinning(closed) \
            if hasattr(cv2, 'ximgproc') else cv2.Canny(closed, 50, 150)

        raw_lines = cv2.HoughLinesP(
            skeleton, rho=1, theta=np.pi / 180, threshold=20,
            minLineLength=15, maxLineGap=10)

        merged = []
        if raw_lines is not None:
            segments = [((l[0][0], l[0][1]), (l[0][2], l[0][3]))
                        for l in raw_lines]
            merged = _merge_collinear_segments(segments,
                                               angle_thresh=12.0,
                                               dist_thresh=8.0,
                                               gap_thresh=20.0)
        # Snap near-axis-aligned lines
        self.wall_lines = [_snap_to_axis(p1, p2) for p1, p2 in merged
                           if _seg_length(p1, p2) >= 8]
        self.surface_rects = []

    # ── drawing ─────────────────────────────────────────────────────────

    def _draw_tracked(self, img, obj):
        cx = int(obj['center_x'] / self.resolution + self.origin_col)
        cz = int(self.origin_row - obj['center_z'] / self.resolution)
        hw = max(2, int(obj['width'] / self.resolution / 2))
        hl = max(2, int(obj['length'] / self.resolution / 2))

        color = _class_color(obj.get('class', ''))
        gs = self.grid_size
        r1 = int(np.clip(cz - hl, 0, gs - 1))
        r2 = int(np.clip(cz + hl, 0, gs - 1))
        c1 = int(np.clip(cx - hw, 0, gs - 1))
        c2 = int(np.clip(cx + hw, 0, gs - 1))
        cv2.rectangle(img, (c1, r1), (c2, r2), color, 1)

        label = obj.get('class', '')
        tid = obj.get('track_id')
        name = obj.get('name')
        if name:
            label = f"#{tid} {name}"
        elif tid is not None:
            label = f"#{tid} {label}"
        dist = obj.get('distance')
        if dist is not None:
            label += f" {dist:.1f}m"
        cv2.putText(img, label, (c1, max(r1 - 3, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1)


# ── Line merging helpers ────────────────────────────────────────────────────

def _seg_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.degrees(np.arctan2(dy, dx)) % 180

def _seg_length(p1, p2):
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

def _point_to_line_dist(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    L2 = dx * dx + dy * dy
    if L2 < 1e-6:
        return np.sqrt((px - x1) ** 2 + (py - y1) ** 2)
    return abs(dy * px - dx * py + x2 * y1 - y2 * x1) / np.sqrt(L2)

def _snap_to_axis(p1, p2, snap_deg=8.0):
    """If a line is within snap_deg of horizontal or vertical, make it exact."""
    angle = _seg_angle(p1, p2)
    # Near horizontal (0° or 180°)
    if angle < snap_deg or angle > 180 - snap_deg:
        mid_y = (p1[1] + p2[1]) // 2
        return (p1[0], mid_y), (p2[0], mid_y)
    # Near vertical (90°)
    if abs(angle - 90) < snap_deg:
        mid_x = (p1[0] + p2[0]) // 2
        return (mid_x, p1[1]), (mid_x, p2[1])
    return p1, p2


def _merge_collinear_segments(segments, angle_thresh=10.0,
                              dist_thresh=6.0, gap_thresh=12.0):
    """Merge nearly-collinear line segments into longer straight walls."""
    if not segments:
        return []

    used = [False] * len(segments)
    merged = []

    for i in range(len(segments)):
        if used[i]:
            continue
        group = [segments[i]]
        used[i] = True
        a_i = _seg_angle(*segments[i])

        for j in range(i + 1, len(segments)):
            if used[j]:
                continue
            a_j = _seg_angle(*segments[j])
            angle_diff = abs(a_i - a_j)
            if angle_diff > 90:
                angle_diff = 180 - angle_diff
            if angle_diff > angle_thresh:
                continue
            # Check distance of j's midpoint to i's line
            mx = (segments[j][0][0] + segments[j][1][0]) / 2
            my = (segments[j][0][1] + segments[j][1][1]) / 2
            d = _point_to_line_dist(mx, my, *segments[i][0], *segments[i][1])
            if d > dist_thresh:
                continue
            # Check gap between segments isn't too big
            pts_i = [segments[i][0], segments[i][1]]
            pts_j = [segments[j][0], segments[j][1]]
            min_gap = min(
                _seg_length(pts_i[0], pts_j[0]),
                _seg_length(pts_i[0], pts_j[1]),
                _seg_length(pts_i[1], pts_j[0]),
                _seg_length(pts_i[1], pts_j[1]),
            )
            total_len = _seg_length(*segments[i]) + _seg_length(*segments[j])
            if min_gap > gap_thresh + total_len * 0.5:
                continue
            group.append(segments[j])
            used[j] = True

        # Merge group into one segment: fit a line to all endpoints
        all_pts = []
        for s in group:
            all_pts.append(s[0])
            all_pts.append(s[1])
        pts = np.array(all_pts, dtype=np.float32)
        if len(pts) >= 2:
            # Project onto principal axis and take extremes
            mean = pts.mean(axis=0)
            centered = pts - mean
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            direction = vt[0]
            projections = centered @ direction
            i_min = np.argmin(projections)
            i_max = np.argmax(projections)
            p1 = (int(pts[i_min][0]), int(pts[i_min][1]))
            p2 = (int(pts[i_max][0]), int(pts[i_max][1]))
            if _seg_length(p1, p2) >= 5:
                merged.append((p1, p2))

    return merged


def _class_color(cls):
    palette = {
        'person': (0, 180, 255), 'chair': (255, 120, 0),
        'couch': (180, 0, 255), 'bed': (0, 200, 120),
        'dining table': (200, 200, 0), 'tv': (255, 0, 0),
        'laptop': (0, 255, 200), 'wall': (200, 200, 200),
        'surface': (0, 220, 220),
    }
    return palette.get(cls, (0, 200, 0))
