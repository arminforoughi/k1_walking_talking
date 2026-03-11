"""Project depth and YOLO detections into 2D ground-plane objects.

Two independent sources of objects:
1. **Depth surfaces** — contours extracted from the occupancy grid give
   bounding-box outlines of every solid surface (walls, furniture, pillars)
   regardless of whether YOLO recognises them.
2. **YOLO labels** — classified detections are projected onto the ground plane
   so that recognised objects (chair, person, table…) get a class label.

The two are merged: if a YOLO label overlaps a depth surface, the surface
inherits the label.  Unlabelled surfaces are emitted as 'surface' or 'wall'.
"""

import numpy as np


class ObjectMap2D:
    def __init__(self, fx=320.0, fy=320.0, cx=None, cy=None,
                 min_depth=0.15, max_depth=6.0,
                 min_height=-0.3, max_height=2.0,
                 sample_grid=8):
        self.fx = fx
        self.fy = fy
        self._cx = cx
        self._cy = cy
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.min_height = min_height
        self.max_height = max_height
        self.sample_grid = sample_grid

    # ── public ──────────────────────────────────────────────────────────

    def update(self, detections, depth_map, frame_shape=None,
               surface_rects=None, grid_resolution=0.05,
               grid_origin_col=100, grid_origin_row=199):
        """Return ground-plane objects from depth surfaces + YOLO labels.

        surface_rects : list of cv2.minAreaRect tuples from the occupancy grid.
        """
        yolo_objs = self._project_yolo(detections, depth_map, frame_shape)
        surface_objs = self._surfaces_to_objects(
            surface_rects, grid_resolution, grid_origin_col, grid_origin_row)

        # Merge: if a YOLO box overlaps a surface, label the surface
        merged = self._merge(yolo_objs, surface_objs)
        return merged

    # ── YOLO projection (kept from before) ──────────────────────────────

    def _project_yolo(self, detections, depth_map, frame_shape):
        if depth_map is None or depth_map.size == 0 or not detections:
            return []

        dh, dw = depth_map.shape
        if frame_shape is not None:
            fh, fw = frame_shape
        else:
            fh, fw = dh * 2, dw * 2

        cx = self._cx if self._cx is not None else dw / 2.0
        cy = self._cy if self._cy is not None else dh / 2.0
        fx_d = self.fx * (dw / fw)
        fy_d = self.fy * (dh / fh)
        sx, sy = dw / fw, dh / fh

        objects = []
        for det in detections:
            bbox = det.get('bbox')
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            dx1 = max(0, int(x1 * sx))
            dy1 = max(0, int(y1 * sy))
            dx2 = min(dw - 1, int(x2 * sx))
            dy2 = min(dh - 1, int(y2 * sy))
            if dx2 <= dx1 or dy2 <= dy1:
                continue

            us = np.linspace(dx1, dx2, self.sample_grid, dtype=np.int32)
            vs = np.linspace(dy1, dy2, self.sample_grid, dtype=np.int32)
            uu, vv = np.meshgrid(us, vs)
            uu, vv = uu.ravel(), vv.ravel()

            d = depth_map[vv, uu].astype(np.float32) / 1000.0
            valid = (d > self.min_depth) & (d < self.max_depth)
            Y = (vv.astype(np.float32) - cy) * d / fy_d
            valid &= (Y > self.min_height) & (Y < self.max_height)
            if valid.sum() < 3:
                continue

            X = (uu[valid].astype(np.float32) - cx) * d[valid] / fx_d
            Z = d[valid]

            min_x, max_x = float(X.min()), float(X.max())
            min_z, max_z = float(Z.min()), float(Z.max())
            width = max(max_x - min_x, 0.05)
            length = max(max_z - min_z, 0.05)

            objects.append({
                'class': det.get('class', 'object'),
                'center_x': (min_x + max_x) / 2.0,
                'center_z': (min_z + max_z) / 2.0,
                'width': width,
                'length': length,
                'distance': float(np.median(Z)),
                'name': det.get('name'),
                'confidence': det.get('confidence'),
                'source': 'yolo',
            })
        return objects

    # ── depth-surface objects from occupancy grid contours ──────────────

    def _surfaces_to_objects(self, surface_rects, res, origin_col, origin_row):
        if not surface_rects:
            return []
        objects = []
        for rect in surface_rects:
            (cc, cr), (w_px, h_px), angle = rect
            cx_m = (cc - origin_col) * res
            cz_m = (origin_row - cr) * res
            w_m = w_px * res
            h_m = h_px * res
            if w_m < 0.08 and h_m < 0.08:
                continue
            aspect = max(w_m, h_m) / max(min(w_m, h_m), 0.01)
            cls = 'wall' if aspect > 4.0 else 'surface'
            dist = max(abs(cz_m), 0.01)
            objects.append({
                'class': cls,
                'center_x': cx_m,
                'center_z': cz_m,
                'width': w_m,
                'length': h_m,
                'distance': dist,
                'name': None,
                'confidence': None,
                'source': 'depth',
            })
        return objects

    # ── merge YOLO labels onto depth surfaces ───────────────────────────

    def _merge(self, yolo_objs, surface_objs):
        used_surfaces = set()
        merged = []

        for yo in yolo_objs:
            best_idx = -1
            best_overlap = 0.0
            for si, so in enumerate(surface_objs):
                if si in used_surfaces:
                    continue
                ov = _overlap_1d(
                    yo['center_x'] - yo['width'] / 2,
                    yo['center_x'] + yo['width'] / 2,
                    so['center_x'] - so['width'] / 2,
                    so['center_x'] + so['width'] / 2,
                ) * _overlap_1d(
                    yo['center_z'] - yo['length'] / 2,
                    yo['center_z'] + yo['length'] / 2,
                    so['center_z'] - so['length'] / 2,
                    so['center_z'] + so['length'] / 2,
                )
                if ov > best_overlap:
                    best_overlap = ov
                    best_idx = si

            if best_idx >= 0 and best_overlap > 0.01:
                so = surface_objs[best_idx]
                used_surfaces.add(best_idx)
                merged.append({
                    'class': yo['class'],
                    'center_x': so['center_x'],
                    'center_z': so['center_z'],
                    'width': so['width'],
                    'length': so['length'],
                    'distance': yo['distance'],
                    'name': yo.get('name'),
                    'confidence': yo.get('confidence'),
                    'source': 'merged',
                })
            else:
                merged.append(yo)

        for si, so in enumerate(surface_objs):
            if si not in used_surfaces:
                merged.append(so)

        return merged


def _overlap_1d(a1, a2, b1, b2):
    """Overlap length of two 1-D intervals."""
    return max(0.0, min(a2, b2) - max(a1, b1))
