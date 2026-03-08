"""3D scene reconstruction with Gaussian Splat accumulation from stereo depth,
DUSt3R dense clouds, and YOLO/SAM2 instance segmentation."""

import math
import time
import threading
import numpy as np
import cv2

try:
    from scipy.spatial import ConvexHull, QhullError
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    QhullError = Exception

SEMANTIC_COLORS = {
    'person':       (0.30, 0.69, 0.31),
    'bicycle':      (0.85, 0.55, 0.13),
    'car':          (0.83, 0.18, 0.18),
    'motorcycle':   (0.90, 0.30, 0.30),
    'bus':          (0.78, 0.16, 0.16),
    'truck':        (0.62, 0.12, 0.12),
    'chair':        (1.00, 0.60, 0.00),
    'couch':        (0.47, 0.33, 0.28),
    'bed':          (0.38, 0.49, 0.55),
    'dining table': (0.55, 0.76, 0.29),
    'toilet':       (0.60, 0.60, 0.60),
    'tv':           (0.96, 0.26, 0.21),
    'laptop':       (0.61, 0.15, 0.69),
    'mouse':        (0.50, 0.30, 0.70),
    'keyboard':     (0.45, 0.25, 0.65),
    'cell phone':   (0.55, 0.35, 0.75),
    'bottle':       (0.13, 0.59, 0.95),
    'cup':          (0.00, 0.74, 0.83),
    'bowl':         (0.00, 0.65, 0.70),
    'book':         (0.40, 0.60, 0.20),
    'backpack':     (0.20, 0.40, 0.60),
    'umbrella':     (0.70, 0.20, 0.50),
    'handbag':      (0.60, 0.30, 0.40),
    'suitcase':     (0.30, 0.30, 0.50),
    'dog':          (0.80, 0.50, 0.20),
    'cat':          (0.60, 0.40, 0.80),
    'potted plant': (0.20, 0.60, 0.20),
    'clock':        (0.70, 0.70, 0.20),
    'vase':         (0.40, 0.70, 0.90),
    'scissors':     (0.75, 0.25, 0.25),
    'teddy bear':   (0.70, 0.55, 0.40),
    'remote':       (0.50, 0.50, 0.70),
    'microwave':    (0.45, 0.45, 0.55),
    'oven':         (0.50, 0.40, 0.35),
    'refrigerator': (0.55, 0.65, 0.75),
    'sink':         (0.40, 0.55, 0.65),
}

STRUCTURE_COLOR = np.array([0.18, 0.18, 0.24], dtype=np.float32)


def get_class_color(class_name):
    c = SEMANTIC_COLORS.get(class_name, (0.13, 0.59, 0.95))
    return np.array(c, dtype=np.float32)


class SceneReconstructor:
    """Accumulates a Gaussian Splat scene representation from depth unprojection
    and DUSt3R dense point clouds, with per-object mesh generation."""

    MAX_SPLATS = 80000
    MAX_OBJ_SPLATS = 30000
    MERGE_RADIUS = 0.04       # merge into existing splat if within this distance
    INITIAL_OPACITY = 0.5
    INITIAL_SCALE = 0.04
    OPACITY_BOOST = 0.08      # per re-observation
    SCALE_DECAY = 0.92         # tighter with more observations
    MIN_SCALE = 0.015
    MAX_SCALE = 0.12

    def __init__(self):
        self._lock = threading.Lock()

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0
        self.trajectory = []

        # Splat storage: pre-allocated numpy arrays
        self._positions = np.zeros((self.MAX_SPLATS, 3), dtype=np.float32)
        self._colors = np.zeros((self.MAX_SPLATS, 3), dtype=np.float32)
        self._opacities = np.zeros(self.MAX_SPLATS, dtype=np.float32)
        self._scales = np.full(self.MAX_SPLATS, self.INITIAL_SCALE, dtype=np.float32)
        self._object_ids = np.full(self.MAX_SPLATS, -1, dtype=np.int32)
        self._n_splats = 0

        self._objects = []
        self._object_meshes = []

        self._last_process_time = 0
        self._process_interval = 0.35
        self._fx = 400.0
        self._fy = 400.0

        self._depth_buffer = []
        self._max_depth_frames = 3

    # ── Pose ──────────────────────────────────────────────────────────────

    def update_pose(self, vx, vy, vyaw, dt=0.05):
        c, s = math.cos(self.robot_theta), math.sin(self.robot_theta)
        self.robot_x += (vx * c - vy * s) * dt
        self.robot_y += (vx * s + vy * c) * dt
        self.robot_theta += vyaw * dt
        self.trajectory.append((self.robot_x, self.robot_y))
        if len(self.trajectory) > 5000:
            self.trajectory = self.trajectory[-2500:]

    # ── Depth Filtering ───────────────────────────────────────────────────

    def _filter_depth(self, depth_map):
        depth_f = depth_map.astype(np.float32)
        valid_mask = depth_f > 0
        filtered = cv2.bilateralFilter(depth_f, d=7, sigmaColor=100, sigmaSpace=7)
        filtered[~valid_mask] = 0

        self._depth_buffer.append(filtered.copy())
        if len(self._depth_buffer) > self._max_depth_frames:
            self._depth_buffer.pop(0)

        if len(self._depth_buffer) >= 2:
            shapes = {f.shape for f in self._depth_buffer}
            if len(shapes) == 1:
                stack = np.stack(self._depth_buffer, axis=0)
                valid = stack > 0
                valid_count = valid.sum(axis=0)
                with np.errstate(invalid='ignore', divide='ignore'):
                    stack_masked = np.where(valid, stack, np.nan)
                    median = np.nanmedian(stack_masked, axis=0)
                filtered = np.where(
                    valid_count > 0,
                    np.nan_to_num(median, nan=0.0),
                    0,
                ).astype(np.float32)

        return filtered.astype(np.uint16)

    # ── Splat Accumulation ────────────────────────────────────────────────

    def _merge_splats(self, new_positions, new_colors, new_object_ids):
        """Merge incoming 3D points into the splat buffer.
        Points near existing splats are merged (position averaged, opacity boosted,
        scale tightened). New points create fresh splats."""

        n_new = len(new_positions)
        if n_new == 0:
            return

        n_existing = self._n_splats
        if n_existing == 0:
            n_add = min(n_new, self.MAX_SPLATS)
            self._positions[:n_add] = new_positions[:n_add]
            self._colors[:n_add] = new_colors[:n_add]
            self._opacities[:n_add] = self.INITIAL_OPACITY
            self._scales[:n_add] = self.INITIAL_SCALE
            self._object_ids[:n_add] = new_object_ids[:n_add]
            self._n_splats = n_add
            return

        # Spatial hash for fast nearest-neighbor lookup
        grid_size = self.MERGE_RADIUS * 2
        inv_grid = 1.0 / grid_size
        existing_pos = self._positions[:n_existing]

        grid_keys = (existing_pos * inv_grid).astype(np.int32)
        spatial_hash = {}
        for idx in range(n_existing):
            key = (grid_keys[idx, 0], grid_keys[idx, 1], grid_keys[idx, 2])
            if key not in spatial_hash:
                spatial_hash[key] = []
            spatial_hash[key].append(idx)

        merged_mask = np.zeros(n_new, dtype=bool)
        merge_radius_sq = self.MERGE_RADIUS ** 2

        new_grid_keys = (new_positions * inv_grid).astype(np.int32)

        for i in range(n_new):
            gk = (new_grid_keys[i, 0], new_grid_keys[i, 1], new_grid_keys[i, 2])
            best_idx = -1
            best_dist_sq = merge_radius_sq

            # Check 3x3x3 neighborhood
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    for dz in range(-1, 2):
                        nk = (gk[0] + dx, gk[1] + dy, gk[2] + dz)
                        candidates = spatial_hash.get(nk)
                        if candidates is None:
                            continue
                        for cidx in candidates:
                            diff = existing_pos[cidx] - new_positions[i]
                            dsq = diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]
                            if dsq < best_dist_sq:
                                best_dist_sq = dsq
                                best_idx = cidx

            if best_idx >= 0:
                # Merge: EMA position, boost opacity, tighten scale
                alpha = 0.3
                self._positions[best_idx] = (
                    self._positions[best_idx] * (1 - alpha) + new_positions[i] * alpha
                )
                self._colors[best_idx] = (
                    self._colors[best_idx] * (1 - alpha) + new_colors[i] * alpha
                )
                self._opacities[best_idx] = min(
                    1.0, self._opacities[best_idx] + self.OPACITY_BOOST
                )
                self._scales[best_idx] = max(
                    self.MIN_SCALE,
                    self._scales[best_idx] * self.SCALE_DECAY
                )
                if new_object_ids[i] >= 0:
                    self._object_ids[best_idx] = new_object_ids[i]
                merged_mask[i] = True

        # Add un-merged points as new splats
        unmerged_idx = np.where(~merged_mask)[0]
        n_unmerged = len(unmerged_idx)
        if n_unmerged > 0:
            space = self.MAX_SPLATS - n_existing
            if n_unmerged > space:
                # Evict low-opacity structure splats first
                self._evict_splats(n_unmerged - space)
                n_existing = self._n_splats
                space = self.MAX_SPLATS - n_existing

            n_add = min(n_unmerged, space)
            if n_add > 0:
                add_idx = unmerged_idx[:n_add]
                start = n_existing
                end = start + n_add
                self._positions[start:end] = new_positions[add_idx]
                self._colors[start:end] = new_colors[add_idx]
                self._opacities[start:end] = self.INITIAL_OPACITY
                self._scales[start:end] = self.INITIAL_SCALE
                self._object_ids[start:end] = new_object_ids[add_idx]
                self._n_splats = end

    def _evict_splats(self, n_evict):
        """Remove the lowest-opacity structure splats to make room."""
        n = self._n_splats
        if n == 0:
            return

        is_structure = self._object_ids[:n] < 0
        struct_indices = np.where(is_structure)[0]

        if len(struct_indices) == 0:
            # If all are object splats, evict lowest opacity overall
            sort_idx = np.argsort(self._opacities[:n])
            keep = sort_idx[n_evict:]
        else:
            struct_opacities = self._opacities[struct_indices]
            sort_order = np.argsort(struct_opacities)
            n_remove = min(n_evict, len(struct_indices))
            remove_set = set(struct_indices[sort_order[:n_remove]].tolist())

            if n_remove < n_evict:
                remaining = n_evict - n_remove
                obj_indices = np.where(~is_structure)[0]
                obj_opacities = self._opacities[obj_indices]
                obj_sort = np.argsort(obj_opacities)
                extra_remove = obj_indices[obj_sort[:remaining]]
                remove_set.update(extra_remove.tolist())

            keep = np.array([i for i in range(n) if i not in remove_set], dtype=np.intp)

        if len(keep) == 0:
            self._n_splats = 0
            return

        n_keep = len(keep)
        self._positions[:n_keep] = self._positions[keep]
        self._colors[:n_keep] = self._colors[keep]
        self._opacities[:n_keep] = self._opacities[keep]
        self._scales[:n_keep] = self._scales[keep]
        self._object_ids[:n_keep] = self._object_ids[keep]
        self._n_splats = n_keep

    # ── Mesh Generation ───────────────────────────────────────────────────

    def _build_convex_hull_mesh(self, points_3d):
        if len(points_3d) < 4:
            return None
        if not HAS_SCIPY:
            return self._build_bbox_mesh(points_3d)
        try:
            if len(points_3d) > 500:
                idx = np.random.choice(len(points_3d), 500, replace=False)
                pts = points_3d[idx]
            else:
                pts = points_3d.copy()

            spread = pts.max(axis=0) - pts.min(axis=0)
            if np.any(spread < 0.02):
                return self._build_bbox_mesh(points_3d)

            hull = ConvexHull(pts)
            hull_verts = hull.vertices
            vertices = pts[hull_verts].astype(np.float32)

            old_to_new = {old: new for new, old in enumerate(hull_verts)}
            faces = []
            for simplex in hull.simplices:
                mapped = [old_to_new.get(v) for v in simplex]
                if None not in mapped:
                    faces.append(mapped)

            if not faces:
                return self._build_bbox_mesh(points_3d)

            faces = np.array(faces, dtype=np.int32)
            normals = np.zeros_like(vertices)
            for face in faces:
                v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
                n = np.cross(v1 - v0, v2 - v0)
                ln = np.linalg.norm(n)
                if ln > 1e-8:
                    n /= ln
                normals[face[0]] += n
                normals[face[1]] += n
                normals[face[2]] += n

            lens = np.linalg.norm(normals, axis=1, keepdims=True)
            normals = normals / np.maximum(lens, 1e-8)

            return {
                'vertices': vertices,
                'indices': faces.flatten().astype(np.int32),
                'normals': normals.astype(np.float32),
            }
        except (QhullError, Exception):
            return self._build_bbox_mesh(points_3d)

    def _build_bbox_mesh(self, points_3d):
        if len(points_3d) < 2:
            return None
        mn = points_3d.min(axis=0).astype(np.float32)
        mx = points_3d.max(axis=0).astype(np.float32)
        for i in range(3):
            if mx[i] - mn[i] < 0.05:
                c = (mn[i] + mx[i]) / 2
                mn[i] = c - 0.05
                mx[i] = c + 0.05

        vertices = np.array([
            [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
            [mx[0], mx[1], mn[2]], [mn[0], mx[1], mn[2]],
            [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]],
            [mx[0], mx[1], mx[2]], [mn[0], mx[1], mx[2]],
        ], dtype=np.float32)

        indices = np.array([
            0, 2, 1,  0, 3, 2,  4, 5, 6,  4, 6, 7,
            0, 1, 5,  0, 5, 4,  2, 3, 7,  2, 7, 6,
            0, 4, 7,  0, 7, 3,  1, 2, 6,  1, 6, 5,
        ], dtype=np.int32)

        center = (mn + mx) / 2
        normals = np.zeros_like(vertices)
        for i in range(8):
            n = vertices[i] - center
            ln = np.linalg.norm(n)
            normals[i] = n / ln if ln > 1e-8 else np.array([0, 1, 0])

        return {
            'vertices': vertices,
            'indices': indices,
            'normals': normals.astype(np.float32),
        }

    # ── Main Processing ───────────────────────────────────────────────────

    def process_frame(self, depth_map, detections, frame_shape,
                      seg_instance_map=None, instance_info=None,
                      dense_cloud=None):
        now = time.time()
        if now - self._last_process_time < self._process_interval:
            return
        self._last_process_time = now

        all_positions = []
        all_colors = []
        all_obj_ids = []
        obj_clusters = {}
        new_obj_meshes = []

        # 1) Depth unprojection (existing pipeline)
        if depth_map is not None:
            depth_map = self._filter_depth(depth_map)
            pts, cols, obj_ids, obj_clusters, new_obj_meshes = self._unproject_depth(
                depth_map, frame_shape, seg_instance_map, instance_info
            )
            if pts is not None:
                all_positions.append(pts)
                all_colors.append(cols)
                all_obj_ids.append(obj_ids)

        # 2) DUSt3R dense cloud
        if dense_cloud is not None and dense_cloud['count'] > 0:
            dp = dense_cloud['positions']  # (N, 3) camera frame
            dc = dense_cloud['colors']     # (N, 3) RGB [0,1]

            # Transform DUSt3R points: camera -> world
            # DUSt3R outputs in camera frame; apply robot pose
            c, s = math.cos(self.robot_theta), math.sin(self.robot_theta)
            # DUSt3R: X=right, Y=down, Z=forward (OpenCV convention)
            body_x = dp[:, 2]   # forward = depth Z
            body_y = -dp[:, 0]  # left = -X
            body_h = -dp[:, 1]  # up = -Y

            world_x = self.robot_x + body_x * c - body_y * s
            world_y = self.robot_y + body_x * s + body_y * c
            world_h = body_h

            # Filter outliers
            height_ok = (world_h > -1.0) & (world_h < 3.0)
            dist_sq = (world_x - self.robot_x)**2 + (world_y - self.robot_y)**2
            dist_ok = dist_sq < 36.0
            keep = height_ok & dist_ok

            if np.sum(keep) > 10:
                # Three.js: X=worldX, Y=height, Z=worldY
                threejs_pos = np.column_stack([
                    world_x[keep], world_h[keep], world_y[keep]
                ]).astype(np.float32)
                all_positions.append(threejs_pos)
                all_colors.append(dc[keep].astype(np.float32))
                all_obj_ids.append(np.full(int(np.sum(keep)), -1, dtype=np.int32))

        if not all_positions:
            return

        merged_pos = np.concatenate(all_positions, axis=0)
        merged_col = np.concatenate(all_colors, axis=0)
        merged_obj = np.concatenate(all_obj_ids, axis=0)

        with self._lock:
            self._merge_splats(merged_pos, merged_col, merged_obj)
            self._update_objects(obj_clusters, detections, now)
            self._object_meshes = new_obj_meshes

    def _unproject_depth(self, depth_map, frame_shape, seg_instance_map, instance_info):
        """Unproject depth map to 3D splat points in Three.js coords."""
        dh, dw = depth_map.shape
        fh, fw = frame_shape if frame_shape else (dh, dw)
        cx, cy = dw / 2.0, dh / 2.0
        fx, fy = self._fx, self._fy

        step = max(1, min(dh, dw) // 80)
        vs = np.arange(0, dh, step)
        us = np.arange(0, dw, step)
        uu, vv = np.meshgrid(us, vs)

        z_m = depth_map[vv, uu].astype(np.float32) / 1000.0
        valid = (z_m > 0.15) & (z_m < 6.0)
        if not np.any(valid):
            return None, None, None, {}, []

        uu_v = uu[valid].astype(np.float32)
        vv_v = vv[valid].astype(np.float32)
        z_v = z_m[valid]

        cam_x = (uu_v - cx) * z_v / fx
        cam_y = (vv_v - cy) * z_v / fy

        body_x = z_v
        body_y = -cam_x
        body_h = -cam_y

        c, s = math.cos(self.robot_theta), math.sin(self.robot_theta)
        world_x = self.robot_x + body_x * c - body_y * s
        world_y = self.robot_y + body_x * s + body_y * c
        world_h = body_h

        height_ok = (world_h > -1.0) & (world_h < 3.0)
        dist_sq = (world_x - self.robot_x)**2 + (world_y - self.robot_y)**2
        dist_ok = dist_sq < 36.0
        keep = height_ok & dist_ok
        if np.sum(keep) < 10:
            return None, None, None, {}, []

        world_x = world_x[keep]
        world_y = world_y[keep]
        world_h = world_h[keep]
        uu_v = uu_v[keep]
        vv_v = vv_v[keep]

        n_pts = len(world_x)
        colors = np.tile(STRUCTURE_COLOR, (n_pts, 1))
        is_obj = np.zeros(n_pts, dtype=bool)
        sampled_instances = np.full(n_pts, -1, dtype=np.int16)
        obj_id_arr = np.full(n_pts, -1, dtype=np.int32)

        if seg_instance_map is not None and instance_info:
            ilh, ilw = seg_instance_map.shape
            u_frame = np.clip((uu_v * ilw / dw).astype(np.intp), 0, ilw - 1)
            v_frame = np.clip((vv_v * ilh / dh).astype(np.intp), 0, ilh - 1)
            sampled_instances = seg_instance_map[v_frame, u_frame]

            for inst_id, info in enumerate(instance_info):
                mask = sampled_instances == inst_id
                if not np.any(mask):
                    continue
                color = get_class_color(info['cls_name'])
                colors[mask] = color
                is_obj[mask] = True
                obj_id_arr[mask] = inst_id

            struct_mask = ~is_obj
            if np.any(struct_mask):
                h_min, h_max = -0.5, 2.0
                t = np.clip((world_h[struct_mask] - h_min) / (h_max - h_min), 0, 1)
                colors[struct_mask, 0] = STRUCTURE_COLOR[0] + t * 0.06
                colors[struct_mask, 1] = STRUCTURE_COLOR[1] + t * 0.04
                colors[struct_mask, 2] = STRUCTURE_COLOR[2] + t * 0.10
        else:
            h_min, h_max = -0.5, 2.0
            t = np.clip((world_h - h_min) / (h_max - h_min), 0, 1)
            colors[:, 0] = np.where(t < 0.5, 0.0, (t - 0.5) * 2)
            colors[:, 1] = 1.0 - np.abs(t - 0.5) * 2
            colors[:, 2] = np.where(t > 0.5, 0.0, (0.5 - t) * 2)

        # Three.js: X=worldX, Y=height, Z=worldY
        positions = np.column_stack([world_x, world_h, world_y]).astype(np.float32)

        # Object meshes
        obj_clusters = {}
        new_obj_meshes = []
        if instance_info:
            for inst_id, info in enumerate(instance_info):
                mask = sampled_instances == inst_id
                count = int(np.sum(mask))
                if count < 15:
                    continue
                ox = world_x[mask]
                oy = world_y[mask]
                oh = world_h[mask]

                obj_clusters[inst_id] = {
                    'label': info['cls_name'],
                    'cx': float(np.mean(ox)),
                    'cy': float(np.mean(oy)),
                    'ch': float(np.mean(oh)),
                    'sx': max(0.15, float(np.ptp(ox))),
                    'sy': max(0.15, float(np.ptp(oy))),
                    'sh': max(0.15, float(np.ptp(oh))),
                    'count': count,
                }

                obj_pts = np.column_stack([ox, oh, oy])
                mesh_data = self._build_convex_hull_mesh(obj_pts)
                if mesh_data:
                    new_obj_meshes.append({
                        'label': info['cls_name'],
                        'vertices': mesh_data['vertices'],
                        'indices': mesh_data['indices'],
                        'normals': mesh_data['normals'],
                        'color': [float(x) for x in get_class_color(info['cls_name'])],
                    })

        return positions, colors, obj_id_arr, obj_clusters, new_obj_meshes

    # ── Object Tracking ───────────────────────────────────────────────────

    def _update_objects(self, obj_clusters, detections, now):
        if not detections:
            self._objects = [o for o in self._objects if now - o['last_seen'] < 120]
            return

        for det in detections:
            label = det.get('name') or det['class']
            dist = det.get('distance_m')

            cluster = None
            for clu in obj_clusters.values():
                if clu['label'] == det['class']:
                    cluster = clu
                    break

            if cluster and cluster['count'] > 5:
                wx, wy, wh = cluster['cx'], cluster['cy'], cluster['ch']
                sx, sy, sh = cluster['sx'], cluster['sy'], cluster['sh']
                pt_count = cluster['count']
            elif dist and dist > 0:
                det_cx, det_cy = det['center']
                fx, fy = self._fx, self._fy
                cam_x_d = (det_cx - 320) * dist / fx
                bx, by = dist, -cam_x_d
                bh = -(det_cy - 240) * dist / fy
                cos_t = math.cos(self.robot_theta)
                sin_t = math.sin(self.robot_theta)
                wx = self.robot_x + bx * cos_t - by * sin_t
                wy = self.robot_y + bx * sin_t + by * cos_t
                wh = bh
                x1, y1, x2, y2 = det['bbox']
                size_m = max(0.2, min(float((x2 - x1) * dist / fx), 3.0))
                sx = sy = sh = size_m
                pt_count = 0
            else:
                continue

            color = [float(x) for x in get_class_color(det['class'])]

            merged = False
            for obj in self._objects:
                if obj['label'] == label:
                    dx = obj['x'] - wx
                    dy = obj['y'] - wy
                    if math.sqrt(dx * dx + dy * dy) < 1.5:
                        a = 0.3
                        obj['x'] = obj['x'] * (1 - a) + wx * a
                        obj['y'] = obj['y'] * (1 - a) + wy * a
                        obj['z'] = obj['z'] * (1 - a) + wh * a
                        obj['sx'] = obj['sx'] * (1 - a) + sx * a
                        obj['sy'] = obj['sy'] * (1 - a) + sy * a
                        obj['sz'] = obj['sz'] * (1 - a) + sh * a
                        obj['point_count'] = pt_count
                        obj['last_seen'] = now
                        merged = True
                        break

            if not merged:
                self._objects.append({
                    'label': label,
                    'x': float(wx), 'y': float(wy), 'z': float(wh),
                    'sx': float(sx), 'sy': float(sy), 'sz': float(sh),
                    'point_count': pt_count, 'color': color,
                    'last_seen': now,
                })

            if cluster:
                del obj_clusters[next(k for k, v in obj_clusters.items()
                                      if v is cluster)]

        self._objects = [o for o in self._objects if now - o['last_seen'] < 120]
        if len(self._objects) > 100:
            self._objects = self._objects[-50:]

    # ── Data Output ───────────────────────────────────────────────────────

    def get_scene_data(self):
        """Return JSON-serializable scene with gaussian splats for Three.js."""
        with self._lock:
            n = self._n_splats
            positions = self._positions[:n].copy()
            colors = self._colors[:n].copy()
            opacities = self._opacities[:n].copy()
            scales = self._scales[:n].copy()
            objects = [dict(o) for o in self._objects]
            traj = list(self.trajectory[-2000:])
            obj_meshes = list(self._object_meshes)

        scene_objects = []
        for o in objects:
            scene_objects.append({
                'label': o['label'],
                'x': round(float(o['x']), 2),
                'y': round(float(o['z']), 2),
                'z': round(float(o['y']), 2),
                'sx': round(float(o.get('sx', 0.3)), 2),
                'sy': round(float(o.get('sz', 0.3)), 2),
                'sz': round(float(o.get('sy', 0.3)), 2),
                'point_count': int(o.get('point_count', 0)),
                'color': [float(c) for c in o.get('color', [0.13, 0.59, 0.95])],
            })

        mesh_list = []
        for m in obj_meshes:
            mesh_list.append({
                'label': m['label'],
                'vertices': [round(float(x), 3) for x in m['vertices'].flatten()],
                'indices': [int(x) for x in m['indices']],
                'normals': [round(float(x), 3) for x in m['normals'].flatten()],
                'color': [float(c) for c in m['color']],
            })

        scene_traj = []
        for tx, ty in traj[::3]:
            scene_traj.extend([round(tx, 2), 0.02, round(ty, 2)])

        return {
            'robot': {
                'x': round(self.robot_x, 2),
                'y': 0,
                'z': round(self.robot_y, 2),
                'theta': round(self.robot_theta, 3),
            },
            'trajectory': scene_traj,
            'objects': scene_objects,
            'splats': {
                'positions': [round(float(x), 3) for x in positions.flatten()],
                'colors': [round(float(x), 3) for x in colors.flatten()],
                'opacities': [round(float(x), 3) for x in opacities],
                'scales': [round(float(x), 4) for x in scales],
                'count': n,
            },
            'object_meshes': mesh_list,
        }

    def get_tracked_objects(self):
        """Return current tracked objects for scene graph."""
        with self._lock:
            return [dict(o) for o in self._objects]

    def clear(self):
        with self._lock:
            self._n_splats = 0
            self._positions[:] = 0
            self._colors[:] = 0
            self._opacities[:] = 0
            self._scales[:] = self.INITIAL_SCALE
            self._object_ids[:] = -1
            self._objects.clear()
            self._object_meshes.clear()
            self.trajectory.clear()
            self.robot_x = self.robot_y = self.robot_theta = 0.0
            self._depth_buffer.clear()
