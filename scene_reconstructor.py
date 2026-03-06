"""3D scene reconstruction from stereo depth + YOLO-seg instance masks."""

import math
import time
import threading
import numpy as np

# Semantic class colors (R, G, B) in 0-1 range
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
    """Accumulates a semantically-colored voxelized 3D point cloud
    from stereo depth + YOLO-seg instance masks as the robot moves."""

    def __init__(self, voxel_size=0.05, max_points=50000):
        self._lock = threading.Lock()
        self.voxel_size = voxel_size
        self.max_points = max_points

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0
        self.trajectory = []

        # (gx,gy,gz) -> (wx, wy, wh, r, g, b, is_object)
        self._voxels = {}
        self._objects = []
        self._class_counts = {}

        self._last_process_time = 0
        self._process_interval = 0.5
        self._fx = 400.0
        self._fy = 400.0

    def update_pose(self, vx, vy, vyaw, dt=0.05):
        """Dead-reckoning from body-frame velocities. x=forward, y=left."""
        c, s = math.cos(self.robot_theta), math.sin(self.robot_theta)
        self.robot_x += (vx * c - vy * s) * dt
        self.robot_y += (vx * s + vy * c) * dt
        self.robot_theta += vyaw * dt
        self.trajectory.append((self.robot_x, self.robot_y))
        if len(self.trajectory) > 5000:
            self.trajectory = self.trajectory[-2500:]

    def process_frame(self, depth_map, detections, frame_shape,
                      seg_instance_map=None, instance_info=None):
        """Project depth to 3D, color semantically using seg masks, accumulate."""
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

        step = max(1, min(dh, dw) // 60)
        vs = np.arange(0, dh, step)
        us = np.arange(0, dw, step)
        uu, vv = np.meshgrid(us, vs)

        z_m = depth_map[vv, uu].astype(np.float32) / 1000.0
        valid = (z_m > 0.1) & (z_m < 8.0)
        if not np.any(valid):
            return

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

        n_pts = len(z_v)
        r = np.full(n_pts, STRUCTURE_COLOR[0], dtype=np.float32)
        g = np.full(n_pts, STRUCTURE_COLOR[1], dtype=np.float32)
        b = np.full(n_pts, STRUCTURE_COLOR[2], dtype=np.float32)
        is_obj = np.zeros(n_pts, dtype=bool)
        sampled_instances = np.full(n_pts, -1, dtype=np.int16)

        if seg_instance_map is not None and instance_info:
            ilh, ilw = seg_instance_map.shape
            u_frame = (uu_v * ilw / dw).astype(int)
            v_frame = (vv_v * ilh / dh).astype(int)
            np.clip(u_frame, 0, ilw - 1, out=u_frame)
            np.clip(v_frame, 0, ilh - 1, out=v_frame)
            sampled_instances = seg_instance_map[v_frame, u_frame]

            for inst_id, info in enumerate(instance_info):
                mask = sampled_instances == inst_id
                if not np.any(mask):
                    continue
                color = get_class_color(info['cls_name'])
                r[mask] = color[0]
                g[mask] = color[1]
                b[mask] = color[2]
                is_obj[mask] = True

            struct_mask = ~is_obj
            if np.any(struct_mask):
                h_min, h_max = -0.5, 2.0
                t = np.clip((world_h[struct_mask] - h_min) / (h_max - h_min), 0, 1)
                r[struct_mask] = STRUCTURE_COLOR[0] + t * 0.06
                g[struct_mask] = STRUCTURE_COLOR[1] + t * 0.04
                b[struct_mask] = STRUCTURE_COLOR[2] + t * 0.10
        else:
            h_min, h_max = -0.5, 2.0
            t = np.clip((world_h - h_min) / (h_max - h_min), 0, 1)
            r[:] = np.where(t < 0.5, 0.0, (t - 0.5) * 2)
            g[:] = 1.0 - np.abs(t - 0.5) * 2
            b[:] = np.where(t > 0.5, 0.0, (0.5 - t) * 2)

        inv_vs = 1.0 / self.voxel_size
        vk = np.column_stack([
            (world_x * inv_vs).astype(int),
            (world_y * inv_vs).astype(int),
            (world_h * inv_vs).astype(int),
        ])
        _, unique_idx = np.unique(vk, axis=0, return_index=True)

        obj_clusters = {}
        if instance_info:
            for inst_id, info in enumerate(instance_info):
                mask = sampled_instances == inst_id
                count = int(np.sum(mask))
                if count < 5:
                    continue
                obj_clusters[inst_id] = {
                    'label': info['cls_name'],
                    'cx': float(np.mean(world_x[mask])),
                    'cy': float(np.mean(world_y[mask])),
                    'ch': float(np.mean(world_h[mask])),
                    'sx': max(0.15, float(np.ptp(world_x[mask]))),
                    'sy': max(0.15, float(np.ptp(world_y[mask]))),
                    'sh': max(0.15, float(np.ptp(world_h[mask]))),
                    'count': count,
                }

        with self._lock:
            for i in unique_idx:
                key = (int(vk[i, 0]), int(vk[i, 1]), int(vk[i, 2]))
                obj_flag = bool(is_obj[i])
                existing = self._voxels.get(key)
                if existing is None or (obj_flag and not existing[6]):
                    self._voxels[key] = (
                        float(world_x[i]), float(world_y[i]), float(world_h[i]),
                        float(r[i]), float(g[i]), float(b[i]),
                        obj_flag,
                    )

            if len(self._voxels) > self.max_points:
                items = list(self._voxels.items())
                struct_keys = [k for k, v in items if not v[6]]
                excess = len(self._voxels) - self.max_points
                to_remove = struct_keys[:excess]
                if len(to_remove) < excess:
                    obj_keys = [k for k, v in items if v[6]]
                    to_remove.extend(obj_keys[:excess - len(to_remove)])
                for k in to_remove:
                    del self._voxels[k]

            self._update_objects(obj_clusters, detections, now)

            counts = {}
            for v in self._voxels.values():
                counts['_structure' if not v[6] else '_object'] = \
                    counts.get('_structure' if not v[6] else '_object', 0) + 1
            self._class_counts = counts

    def _update_objects(self, obj_clusters, detections, now):
        """Merge per-instance 3D clusters into persistent object list."""
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
                cos_t, sin_t = math.cos(self.robot_theta), math.sin(self.robot_theta)
                wx = self.robot_x + bx * cos_t - by * sin_t
                wy = self.robot_y + bx * sin_t + by * cos_t
                wh = bh
                x1, y1, x2, y2 = det['bbox']
                size_m = max(0.2, min(float((x2 - x1) * dist / fx), 3.0))
                sx = sy = sh = size_m
                pt_count = 0
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
                del obj_clusters[next(k for k, v in obj_clusters.items() if v is cluster)]

        self._objects = [o for o in self._objects if now - o['last_seen'] < 120]
        if len(self._objects) > 100:
            self._objects = self._objects[-50:]

    def get_scene_data(self):
        """Return JSON-serializable scene for the Three.js UI.
        Coordinate mapping: Three.js X = world X, Y = height, Z = world Y."""
        with self._lock:
            voxels = list(self._voxels.values())
            objects = [dict(o) for o in self._objects]
            traj = list(self.trajectory[-2000:])
            class_counts = dict(self._class_counts)

        positions = []
        colors = []
        for v in voxels:
            positions.extend([round(v[0], 2), round(v[2], 2), round(v[1], 2)])
            colors.extend([round(v[3], 2), round(v[4], 2), round(v[5], 2)])

        scene_objects = []
        for o in objects:
            scene_objects.append({
                'label': o['label'],
                'x': round(o['x'], 2),
                'y': round(o['z'], 2),
                'z': round(o['y'], 2),
                'sx': round(o.get('sx', 0.3), 2),
                'sy': round(o.get('sz', 0.3), 2),
                'sz': round(o.get('sy', 0.3), 2),
                'point_count': o.get('point_count', 0),
                'color': o.get('color', [0.13, 0.59, 0.95]),
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
            'points': {
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
            self._class_counts.clear()
            self.trajectory.clear()
            self.robot_x = self.robot_y = self.robot_theta = 0.0
