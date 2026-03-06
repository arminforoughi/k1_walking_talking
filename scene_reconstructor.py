"""3D scene reconstruction with per-object triangle meshes from depth + YOLO-seg.

Generates actual surface meshes for detected objects using depth map triangulation,
plus an accumulated voxel cloud for background structure.
"""

import math
import time
import threading
import numpy as np
import cv2
from scipy.spatial import Delaunay

from stereo_depth import clean_depth_for_reconstruction

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

STRUCTURE_COLOR = (0.18, 0.18, 0.24)


def get_class_color(class_name):
    return SEMANTIC_COLORS.get(class_name, (0.13, 0.59, 0.95))


class SceneReconstructor:
    """Builds per-object triangle meshes from depth + YOLO-seg masks,
    plus an accumulated voxel cloud for background structure."""

    def __init__(self, voxel_size=0.10, max_points=15000):
        self._lock = threading.Lock()
        self.voxel_size = voxel_size
        self.max_points = max_points

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0
        self.trajectory = []

        # voxel key -> (wx, wy, wh, r, g, b, hit_count)
        self._voxels = {}
        self._objects = []
        self._latest_meshes = []
        self._class_counts = {}

        self._last_process_time = 0
        self._process_interval = 0.5
        self._frame_count = 0
        self._min_struct_confidence = 0.55
        self._min_hits_to_show = 2
        self._fx = 400.0
        self._fy = 400.0

    def update_pose(self, vx, vy, vyaw, dt=0.05):
        c, s = math.cos(self.robot_theta), math.sin(self.robot_theta)
        self.robot_x += (vx * c - vy * s) * dt
        self.robot_y += (vx * s + vy * c) * dt
        self.robot_theta += vyaw * dt
        self.trajectory.append((self.robot_x, self.robot_y))
        if len(self.trajectory) > 5000:
            self.trajectory = self.trajectory[-2500:]

    def process_frame(self, depth_map, detections, frame_shape,
                      seg_instance_map=None, instance_info=None,
                      depth_is_meters=False):
        """Process a frame: build object meshes + accumulate structure voxels."""
        now = time.time()
        if now - self._last_process_time < self._process_interval:
            return
        self._last_process_time = now
        if depth_map is None:
            return

        dh, dw = depth_map.shape
        fh, fw = frame_shape if frame_shape else (dh, dw)
        cx, cy = dw / 2.0, dh / 2.0
        fx, fy = self._fx, self._fy

        if depth_is_meters:
            cleaned_m = depth_map.astype(np.float32).copy()
            valid_mask = (cleaned_m > 0.1) & (cleaned_m < 8.0)
            cleaned_m[~valid_mask] = 0
            depth_conf = np.where(valid_mask, 1.0, 0.0).astype(np.float32)
        else:
            cleaned_m, depth_conf = clean_depth_for_reconstruction(
                depth_map,
                seg_instance_map=seg_instance_map,
                instance_info=instance_info,
            )

        # --- Build per-object meshes ---
        new_meshes = []
        seg_at_depth = None
        if seg_instance_map is not None and instance_info:
            seg = seg_instance_map
            if seg.shape != (dh, dw):
                seg_at_depth = cv2.resize(
                    seg.astype(np.float32), (dw, dh),
                    interpolation=cv2.INTER_NEAREST
                ).astype(np.int16)
            else:
                seg_at_depth = seg

            for inst_id, info in enumerate(instance_info):
                mask = (seg_at_depth == inst_id) & (cleaned_m > 0.1)
                if int(np.sum(mask)) < 20:
                    continue
                mesh = self._build_mesh_from_mask(
                    cleaned_m, mask, fx, fy, cx, cy,
                )
                if mesh is None:
                    continue
                color = list(get_class_color(info['cls_name']))
                new_meshes.append({
                    'label': info['cls_name'],
                    'vertices': mesh[0],
                    'faces': mesh[1],
                    'color': color,
                })

        # --- Accumulate structure voxels (non-object pixels, very sparse) ---
        step = max(2, min(dh, dw) // 40)
        vs = np.arange(0, dh, step)
        us = np.arange(0, dw, step)
        uu, vv = np.meshgrid(us, vs)

        z_m = cleaned_m[vv, uu]
        conf_s = depth_conf[vv, uu]
        valid = (z_m > 0.2) & (z_m < 6.0) & (conf_s > self._min_struct_confidence)

        is_object_pixel = np.zeros_like(valid)
        if seg_at_depth is not None:
            sampled_seg = seg_at_depth[vv, uu]
            is_object_pixel = sampled_seg >= 0

        struct_valid = valid & ~is_object_pixel

        if np.any(struct_valid):
            uu_v = uu[struct_valid].astype(np.float32)
            vv_v = vv[struct_valid].astype(np.float32)
            z_v = z_m[struct_valid]

            cam_x = (uu_v - cx) * z_v / fx
            cam_y = (vv_v - cy) * z_v / fy

            body_x, body_y, body_h = z_v, -cam_x, -cam_y
            c, s = math.cos(self.robot_theta), math.sin(self.robot_theta)
            world_x = self.robot_x + body_x * c - body_y * s
            world_y = self.robot_y + body_x * s + body_y * c
            world_h = body_h

            h_min, h_max = -0.3, 2.0
            t = np.clip((world_h - h_min) / (h_max - h_min), 0, 1)
            r_c = STRUCTURE_COLOR[0] + t * 0.08
            g_c = STRUCTURE_COLOR[1] + t * 0.06
            b_c = STRUCTURE_COLOR[2] + t * 0.12

            inv_vs = 1.0 / self.voxel_size
            vk = np.column_stack([
                (world_x * inv_vs).astype(int),
                (world_y * inv_vs).astype(int),
                (world_h * inv_vs).astype(int),
            ])
            _, unique_idx = np.unique(vk, axis=0, return_index=True)

            with self._lock:
                for i in unique_idx:
                    key = (int(vk[i, 0]), int(vk[i, 1]), int(vk[i, 2]))
                    existing = self._voxels.get(key)
                    if existing is not None:
                        self._voxels[key] = (
                            existing[0], existing[1], existing[2],
                            existing[3], existing[4], existing[5],
                            existing[6] + 1,
                        )
                    else:
                        self._voxels[key] = (
                            float(world_x[i]), float(world_y[i]), float(world_h[i]),
                            float(r_c[i]), float(g_c[i]), float(b_c[i]),
                            1,
                        )

                if len(self._voxels) > self.max_points:
                    items = sorted(self._voxels.items(), key=lambda kv: kv[1][6])
                    for k, _ in items[:len(self._voxels) - self.max_points]:
                        del self._voxels[k]

                self._frame_count += 1
                if self._frame_count % 5 == 0:
                    self._prune_isolated_voxels(min_neighbors=2)
        else:
            with self._lock:
                self._frame_count += 1
                if self._frame_count % 5 == 0:
                    self._prune_isolated_voxels(min_neighbors=2)

        # --- Update object tracking ---
        obj_clusters = {}
        if instance_info and seg_at_depth is not None:
            for inst_id, info in enumerate(instance_info):
                mask = (seg_at_depth == inst_id) & (cleaned_m > 0.1)
                if int(np.sum(mask)) < 5:
                    continue
                ys_m, xs_m = np.where(mask)
                z_pts = cleaned_m[ys_m, xs_m]
                cam_x_pts = (xs_m.astype(float) - cx) * z_pts / fx
                cam_y_pts = (ys_m.astype(float) - cy) * z_pts / fy
                bx, by, bh = z_pts, -cam_x_pts, -cam_y_pts
                co, si = math.cos(self.robot_theta), math.sin(self.robot_theta)
                wx = self.robot_x + bx * co - by * si
                wy = self.robot_y + bx * si + by * co
                wh = bh
                obj_clusters[inst_id] = {
                    'label': info['cls_name'],
                    'cx': float(np.mean(wx)), 'cy': float(np.mean(wy)),
                    'ch': float(np.mean(wh)),
                    'sx': max(0.15, float(np.ptp(wx))),
                    'sy': max(0.15, float(np.ptp(wy))),
                    'sh': max(0.15, float(np.ptp(wh))),
                    'count': len(z_pts),
                }

        with self._lock:
            self._update_objects(obj_clusters, detections, now)
            self._latest_meshes = new_meshes
            struct_count = len(self._voxels)
            obj_count = sum(len(m['faces']) for m in new_meshes)
            self._class_counts = {
                '_structure': struct_count,
                '_mesh_tris': obj_count,
            }

    def _build_mesh_from_mask(self, depth_m, mask, fx, fy, cx, cy,
                              max_verts=1200, max_edge_m=0.25):
        """Build a triangle mesh from depth pixels within a 2D mask.

        Uses grid sampling + Delaunay triangulation, filters stretched triangles
        at depth discontinuities. Returns (vertices_Nx3, faces_Mx3) in world
        coords or None.
        """
        ys, xs = np.where(mask)
        n_mask = len(ys)
        if n_mask < 10:
            return None

        step = max(2, int(math.ceil(math.sqrt(n_mask / max_verts))))
        if step > 1:
            # Subsample on a regular grid within the bounding box
            y_min, y_max = int(ys.min()), int(ys.max())
            x_min, x_max = int(xs.min()), int(xs.max())
            grid_ys = np.arange(y_min, y_max + 1, step)
            grid_xs = np.arange(x_min, x_max + 1, step)
            gg_x, gg_y = np.meshgrid(grid_xs, grid_ys)
            gg_x, gg_y = gg_x.ravel(), gg_y.ravel()
            h, w = depth_m.shape
            in_bounds = (gg_x >= 0) & (gg_x < w) & (gg_y >= 0) & (gg_y < h)
            gg_x, gg_y = gg_x[in_bounds], gg_y[in_bounds]
            in_mask = mask[gg_y, gg_x] & (depth_m[gg_y, gg_x] > 0.1)
            xs, ys = gg_x[in_mask], gg_y[in_mask]
        else:
            valid = depth_m[ys, xs] > 0.1
            xs, ys = xs[valid], ys[valid]

        if len(xs) < 4:
            return None

        z = depth_m[ys, xs]
        x_cam = (xs.astype(np.float32) - cx) * z / fx
        y_cam = (ys.astype(np.float32) - cy) * z / fy

        # Camera frame → body → world
        body_x, body_y, body_h = z, -x_cam, -y_cam
        co, si = math.cos(self.robot_theta), math.sin(self.robot_theta)
        wx = self.robot_x + body_x * co - body_y * si
        wy = self.robot_y + body_x * si + body_y * co
        wh = body_h

        # Three.js coords: (world_x, world_h, world_y)
        verts = np.column_stack([wx, wh, wy]).astype(np.float32)

        # Triangulate in 2D pixel space for topologically correct mesh
        pts_2d = np.column_stack([xs, ys]).astype(np.float64)
        try:
            tri = Delaunay(pts_2d)
        except Exception:
            return None
        faces = tri.simplices

        # Filter stretched triangles (depth discontinuities)
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        e0 = np.linalg.norm(v1 - v0, axis=1)
        e1 = np.linalg.norm(v2 - v1, axis=1)
        e2 = np.linalg.norm(v0 - v2, axis=1)
        max_edges = np.maximum(e0, np.maximum(e1, e2))
        good = max_edges < max_edge_m
        faces = faces[good]

        if len(faces) < 2:
            return None

        # Re-index to only include used vertices
        used = np.unique(faces)
        remap = np.full(len(verts), -1, dtype=np.int32)
        remap[used] = np.arange(len(used), dtype=np.int32)
        verts = verts[used]
        faces = remap[faces]

        return verts, faces.astype(np.int32)

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
            elif dist and dist > 0:
                det_cx, det_cy = det['center']
                fx, fy = self._fx, self._fy
                cam_x_d = (det_cx - 320) * dist / fx
                bx, by = dist, -cam_x_d
                bh = -(det_cy - 240) * dist / fy
                cos_t, sin_t = math.cos(self.robot_theta), math.sin(self.robot_theta)
                wx = self.robot_x + bx * cos_t - by * sin_t
                wy = self.robot_y + bx * sin_t + by * cos_t
                wh = bh
                x1, y1, x2, y2 = det['bbox']
                size_m = max(0.2, min(float((x2 - x1) * dist / fx), 3.0))
                sx = sy = sh = size_m
            else:
                continue

            color = list(get_class_color(det['class']))
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
                        obj['last_seen'] = now
                        merged = True
                        break

            if not merged:
                self._objects.append({
                    'label': label,
                    'x': float(wx), 'y': float(wy), 'z': float(wh),
                    'sx': float(sx), 'sy': float(sy), 'sz': float(sh),
                    'color': color, 'last_seen': now,
                })

            if cluster:
                del obj_clusters[next(k for k, v in obj_clusters.items() if v is cluster)]

        self._objects = [o for o in self._objects if now - o['last_seen'] < 120]
        if len(self._objects) > 100:
            self._objects = self._objects[-50:]

    def _prune_isolated_voxels(self, min_neighbors=1):
        _OFF = ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1))
        to_remove = []
        for key in self._voxels:
            count = 0
            for dx, dy, dz in _OFF:
                if (key[0]+dx, key[1]+dy, key[2]+dz) in self._voxels:
                    count += 1
                    if count >= min_neighbors:
                        break
            if count < min_neighbors:
                to_remove.append(key)
        for key in to_remove:
            del self._voxels[key]

    def get_scene_data(self):
        """Return JSON-serializable scene for the Three.js viewer.

        Contains: object meshes, structure point cloud, robot pose, trajectory.
        Three.js mapping: X = world X, Y = height, Z = world Y.
        """
        with self._lock:
            voxels = list(self._voxels.values())
            objects = [dict(o) for o in self._objects]
            traj = list(self.trajectory[-2000:])
            meshes = list(self._latest_meshes)
            class_counts = dict(self._class_counts)

        # --- Object meshes ---
        scene_meshes = []
        current_mesh_labels = set()
        for m in meshes:
            verts = m['vertices']
            faces = m['faces']
            center = verts.mean(axis=0).tolist() if len(verts) > 0 else [0, 0, 0]
            scene_meshes.append({
                'label': m['label'],
                'vertices': np.round(verts, 3).ravel().tolist(),
                'faces': faces.ravel().tolist(),
                'color': m['color'],
                'center': [round(c, 2) for c in center],
                'type': 'mesh',
            })
            current_mesh_labels.add(m['label'])

        # Persistent objects not in current frame → show as boxes
        for o in objects:
            if o['label'] in current_mesh_labels:
                continue
            scene_meshes.append({
                'label': o['label'],
                'type': 'box',
                'center': [round(o['x'], 2), round(o['z'], 2), round(o['y'], 2)],
                'size': [round(o.get('sx', 0.3), 2),
                         round(o.get('sz', 0.3), 2),
                         round(o.get('sy', 0.3), 2)],
                'color': o.get('color', [0.5, 0.5, 0.5]),
            })

        # --- Structure point cloud (only voxels seen multiple times) ---
        min_hits = self._min_hits_to_show
        positions = []
        colors = []
        for v in voxels:
            if v[6] < min_hits:
                continue
            positions.extend([round(v[0], 2), round(v[2], 2), round(v[1], 2)])
            colors.extend([round(v[3], 2), round(v[4], 2), round(v[5], 2)])

        # --- Trajectory ---
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
            'meshes': scene_meshes,
            'structure': {
                'positions': positions,
                'colors': colors,
                'count': len(voxels),
            },
            'class_counts': class_counts,
        }

    def clear(self):
        with self._lock:
            self._voxels.clear()
            self._objects.clear()
            self._latest_meshes.clear()
            self._class_counts.clear()
            self.trajectory.clear()
            self.robot_x = self.robot_y = self.robot_theta = 0.0
