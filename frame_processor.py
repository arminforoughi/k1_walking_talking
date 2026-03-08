"""Frame processing: YOLO detection, face recognition, depth gradient views."""

import os
import json
import threading
import base64
import time
import struct
import zlib
from datetime import datetime

import numpy as np
import cv2
from ultralytics import YOLO
try:
    import face_recognition
except (ImportError, OSError):
    face_recognition = None  # e.g. dlib wrong arch on Apple Silicon

import re

from stereo_depth import get_gradient_map_from_depth
from scene_reconstructor import SceneReconstructor

# Detection colors (BGR)
_COLORS = [
    (0, 255, 0), (255, 128, 0), (0, 128, 255), (255, 0, 255),
    (0, 255, 255), (128, 255, 0), (255, 0, 128), (128, 0, 255),
]

_NAME_PATTERNS = [
    re.compile(r"\bmy name is (\w+)", re.IGNORECASE),
    re.compile(r"\bi'm (\w+)", re.IGNORECASE),
    re.compile(r"\bi am (\w+)", re.IGNORECASE),
    re.compile(r"\bcall me (\w+)", re.IGNORECASE),
]


def _color_for_class(cls_id):
    return _COLORS[cls_id % len(_COLORS)]


# ── Face Cache ───────────────────────────────────────────────────────────────

FACE_CACHE_DIR = os.path.expanduser('~/.face_cache')
FACE_CACHE_FILE = os.path.join(FACE_CACHE_DIR, 'known_faces.json')


class FaceCache:
    def __init__(self, tolerance=0.6):
        self.tolerance = tolerance
        self.entries = []
        self._lock = threading.Lock()
        os.makedirs(FACE_CACHE_DIR, exist_ok=True)
        self._load()

    def _load(self):
        if not os.path.exists(FACE_CACHE_FILE):
            return
        try:
            with open(FACE_CACHE_FILE) as f:
                data = json.load(f)
            for entry in data:
                self.entries.append({
                    'name': entry['name'],
                    'encoding': np.array(entry['encoding'], dtype=np.float64),
                    'saved_at': entry.get('saved_at', ''),
                })
            print(f"Face cache: loaded {len(self.entries)} known face(s)")
        except Exception as e:
            print(f"Warning: failed to load face cache: {e}")

    def _persist(self):
        try:
            data = [
                {'name': e['name'], 'encoding': e['encoding'].tolist(), 'saved_at': e['saved_at']}
                for e in self.entries
            ]
            with open(FACE_CACHE_FILE, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Warning: failed to save face cache: {e}")

    def recognize(self, encoding):
        if face_recognition is None:
            return None
        with self._lock:
            if not self.entries:
                return None
            known = [e['encoding'] for e in self.entries]
            distances = face_recognition.face_distance(known, encoding)
            best_idx = int(np.argmin(distances))
            if distances[best_idx] <= self.tolerance:
                return self.entries[best_idx]['name']
            return None

    def save_face(self, name, encoding):
        with self._lock:
            for e in self.entries:
                if e['name'].lower() == name.lower():
                    e['encoding'] = encoding
                    e['saved_at'] = datetime.now().isoformat()
                    self._persist()
                    return
            self.entries.append({
                'name': name, 'encoding': encoding,
                'saved_at': datetime.now().isoformat(),
            })
            self._persist()
        print(f"Face cache: saved '{name}'")

    def delete_face(self, name):
        with self._lock:
            self.entries = [e for e in self.entries if e['name'].lower() != name.lower()]
            self._persist()

    def list_known(self):
        with self._lock:
            return [{'name': e['name'], 'saved_at': e['saved_at']} for e in self.entries]


# ── Frame Processor ──────────────────────────────────────────────────────────


class FrameProcessor:
    """Receives frames from robot WebSocket, runs YOLO + face recognition."""

    def __init__(self, model_path='yolov8m-seg.pt', confidence=0.5,
                 face_cache=None, enable_faces=True, on_transcript=None,
                 depth_estimator=None, sam2_segmenter=None,
                 dust3r_reconstructor=None):
        print(f'Loading YOLO model: {model_path}')
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.enable_faces = enable_faces
        self.face_cache = face_cache
        self._on_transcript = on_transcript
        self.depth_estimator = depth_estimator
        self.sam2_segmenter = sam2_segmenter
        self.dust3r_reconstructor = dust3r_reconstructor
        self._dust3r_frame_count = 0

        self._unknown_faces = {}
        self._next_unknown_id = 1
        self._last_face_time = 0.0
        self._face_interval = 0.5
        self._cached_face_results = []

        self._lock = threading.Lock()
        self.latest_frame = None       # annotated frame (for web UI)
        self.latest_detections = []
        self._raw_frame = None         # left camera (original decoded)
        self._raw_frame_right = None   # right camera for DUSt3R stereo
        self._depth_map = None        # uint16 depth
        self._depth_map_enhanced = None  # neural-enhanced depth
        self._frame_shape = None      # (h, w)
        self._fps = 0.0
        self._fps_counter = 0
        self._fps_time = time.time()

        self.scene_reconstructor = SceneReconstructor()

        self._detect_thread = threading.Thread(target=self._detect_loop, daemon=True)
        self._pending_frame = None
        self._detect_event = threading.Event()
        self._detect_thread.start()

    def on_video_frame(self, jpeg_bytes):
        frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is not None:
            self._raw_frame = frame
            self._frame_shape = frame.shape[:2]
            self._pending_frame = frame
            self._detect_event.set()

    def on_video_frame_right(self, jpeg_bytes):
        frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is not None:
            self._raw_frame_right = frame

    def on_depth_frame(self, data):
        try:
            w, h = struct.unpack('<HH', data[:4])
            raw = zlib.decompress(data[4:])
            depth = np.frombuffer(raw, dtype=np.uint16).reshape((h, w))
            self._depth_map = depth
        except Exception as e:
            print(f"Depth decode error: {e}")

    def _detect_loop(self):
        while True:
            self._detect_event.wait()
            self._detect_event.clear()
            frame = self._pending_frame
            if frame is None:
                continue
            try:
                self._run_detection(frame)
            except Exception as e:
                import traceback
                traceback.print_exc()

    def _get_depth_at(self, x, y, window=5):
        depth_map = self._depth_map
        if depth_map is None:
            return None
        dh, dw = depth_map.shape
        fh, fw = self._frame_shape or (dh * 2, dw * 2)
        sx = x * dw / fw
        sy = y * dh / fh
        ix, iy = int(sx), int(sy)
        half = window // 2
        y1, y2 = max(0, iy - half), min(dh, iy + half + 1)
        x1, x2 = max(0, ix - half), min(dw, ix + half + 1)
        patch = depth_map[y1:y2, x1:x2].astype(np.float32)
        valid = patch[(patch > 0) & (patch < 65535)]
        if len(valid) == 0:
            return None
        return float(np.median(valid)) / 1000.0

    def _run_detection(self, frame):
        results = self.model(frame, conf=self.confidence, verbose=False)
        annotated = frame.copy()
        detections = []
        depth_available = self._depth_map is not None
        has_persons = False

        h, w = frame.shape[:2]
        seg_instance_map = np.full((h, w), -1, dtype=np.int16)
        instance_info = []
        has_masks = False

        yolo_boxes = []
        yolo_cls_ids = []
        yolo_masks_data = None

        for result in results:
            if hasattr(result, 'masks') and result.masks is not None:
                yolo_masks_data = result.masks.data.cpu().numpy()

            for i, box in enumerate(result.boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                if cls_name == 'person':
                    has_persons = True
                cx, cy = int((x1 + x2) // 2), int((y1 + y2) // 2)
                distance_m = self._get_depth_at(cx, cy) if depth_available else None
                detections.append({
                    'class': cls_name, 'confidence': round(float(conf), 2),
                    'distance_m': round(float(distance_m), 2) if distance_m else None,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'center': [int(cx), int(cy)],
                    'name': None, 'unknown_id': None,
                })
                yolo_boxes.append([int(x1), int(y1), int(x2), int(y2)])
                yolo_cls_ids.append(cls_id)

        use_sam2 = (self.sam2_segmenter is not None and self.sam2_segmenter.ready
                    and len(yolo_boxes) > 0)
        sam2_masks = None
        sam2_str = ""
        if use_sam2:
            t0 = time.time()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            sam2_masks = self.sam2_segmenter.segment_from_boxes(rgb, yolo_boxes)
            sam2_str = f" | SAM2: {(time.time() - t0) * 1000:.0f}ms"

        for i, det in enumerate(detections):
            cls_id = yolo_cls_ids[i]
            cls_name = det['class']
            mask_bool = None

            if sam2_masks and i < len(sam2_masks):
                mask_bool = sam2_masks[i]
                has_masks = True
            elif yolo_masks_data is not None and i < len(yolo_masks_data):
                mask_resized = cv2.resize(
                    yolo_masks_data[i], (w, h), interpolation=cv2.INTER_NEAREST
                )
                mask_bool = mask_resized > 0.5
                has_masks = True

            if mask_bool is not None:
                if mask_bool.dtype != np.bool_:
                    mask_bool = mask_bool.astype(np.bool_)
                inst_id = len(instance_info)
                seg_instance_map[mask_bool] = inst_id
                instance_info.append({'cls_id': cls_id, 'cls_name': cls_name})

                color_bgr = _color_for_class(cls_id)
                annotated[mask_bool] = (
                    annotated[mask_bool].astype(np.float32) * 0.55 +
                    np.array(color_bgr, dtype=np.float32) * 0.45
                ).astype(np.uint8)
                contours, _ = cv2.findContours(
                    mask_bool.astype(np.uint8),
                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                )
                cv2.drawContours(annotated, contours, -1, color_bgr, 1)

        now = time.time()
        if self.enable_faces and has_persons and now - self._last_face_time >= self._face_interval:
            self._last_face_time = now
            self._cached_face_results = self._run_face_recognition(frame)

        for fr in self._cached_face_results:
            matched_det = self._match_face_to_person(fr, detections)
            if matched_det:
                matched_det['name'] = fr['name']
                matched_det['unknown_id'] = fr.get('unknown_id')
            top, right, bottom, left = fr['face_loc']
            is_known = fr['unknown_id'] is None
            face_color = (0, 255, 255) if is_known else (0, 165, 255)
            cv2.rectangle(annotated, (left, top), (right, bottom), face_color, 2)
            name = fr['name']
            (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(annotated, (left, bottom), (left + tw + 4, bottom + th + 8), face_color, -1)
            cv2.putText(annotated, name, (left + 2, bottom + th + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cls_name = det['class']
            conf = det['confidence']
            distance_m = det['distance_m']
            cls_id = list(self.model.names.values()).index(cls_name) if cls_name in self.model.names.values() else 0
            color = _color_for_class(cls_id)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {conf:.0%}"
            if det['name'] and cls_name == 'person':
                label = f"{det['name']} {conf:.0%}"
            if distance_m is not None:
                label += f" {distance_m:.1f}m"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        self._fps_counter += 1
        if now - self._fps_time >= 1.0:
            self._fps = self._fps_counter / (now - self._fps_time)
            self._fps_counter = 0
            self._fps_time = now

        # Neural depth enhancement
        depth_for_scene = self._depth_map
        neural_str = ""
        if self.depth_estimator:
            if self.depth_estimator.ready:
                if depth_for_scene is not None:
                    depth_for_scene = self.depth_estimator.refine_depth(frame, depth_for_scene)
                else:
                    depth_for_scene = self.depth_estimator.estimate_depth(frame)
                self._depth_map_enhanced = depth_for_scene
                neural_str = f" | Neural: {self.depth_estimator._inference_ms:.0f}ms"
            else:
                neural_str = " | Neural: loading..."

        # DUSt3R status (async, runs in background)
        dust3r_str = ""
        if self.dust3r_reconstructor and self._raw_frame_right is not None:
            if self.dust3r_reconstructor.ready:
                dust3r_str = f" | DUSt3R: {self.dust3r_reconstructor._inference_ms:.0f}ms/{self.dust3r_reconstructor._point_count}pts"
            else:
                dust3r_str = " | DUSt3R: loading..."

        faces_str = f"Faces: {len(self._cached_face_results)}" if self.enable_faces else "Faces: off"
        depth_str = "Depth: ON" if depth_available else "Depth: waiting..."
        seg_str = "SAM2" if use_sam2 else ("YOLO-seg" if has_masks else "off")
        status = f"FPS: {self._fps:.0f} | {len(detections)} obj | {faces_str} | {depth_str} | Seg: {seg_str}{sam2_str}{neural_str}{dust3r_str}"
        cv2.putText(annotated, status, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

        with self._lock:
            self.latest_frame = annotated
            self.latest_detections = detections

        # Feed stereo pair to DUSt3R (runs async in background thread)
        if self.dust3r_reconstructor and self._raw_frame_right is not None:
            self._dust3r_frame_count += 1
            if self._dust3r_frame_count % 30 == 1:  # ~every 3s at 10fps
                self.dust3r_reconstructor.set_frames(frame, self._raw_frame_right)

        if self.scene_reconstructor:
            dense_cloud = None
            if self.dust3r_reconstructor:
                dense_cloud = self.dust3r_reconstructor.get_latest_points()
            self.scene_reconstructor.process_frame(
                depth_for_scene, detections, self._frame_shape,
                seg_instance_map=seg_instance_map if has_masks else None,
                instance_info=instance_info if has_masks else None,
                dense_cloud=dense_cloud,
            )

    def _run_face_recognition(self, frame):
        if face_recognition is None:
            return []
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locs = face_recognition.face_locations(rgb, model='cnn')
        if not face_locs:
            return []
        face_encs = face_recognition.face_encodings(rgb, face_locs, model='small')
        results = []
        for loc, enc in zip(face_locs, face_encs):
            top, right, bottom, left = loc
            name = self.face_cache.recognize(enc) if self.face_cache else None
            unknown_id = None
            if name is None:
                unknown_id = self._get_or_assign_unknown_id(enc)
                name = f"Unknown #{unknown_id}"
            results.append({
                'name': name, 'unknown_id': unknown_id,
                'face_loc': (top, right, bottom, left), 'encoding': enc,
            })
        return results

    def _get_or_assign_unknown_id(self, encoding):
        if face_recognition is None:
            return 0
        best_dist, best_id = 999.0, None
        for uid, enc in self._unknown_faces.items():
            dist = float(face_recognition.face_distance([enc], encoding)[0])
            if dist < best_dist:
                best_dist, best_id = dist, uid
        if best_id is not None and best_dist < 0.5:
            self._unknown_faces[best_id] = encoding
            return best_id
        uid = self._next_unknown_id
        self._next_unknown_id += 1
        self._unknown_faces[uid] = encoding
        return uid

    def _match_face_to_person(self, face_result, detections):
        top, right, bottom, left = face_result['face_loc']
        face_cx = (left + right) // 2
        face_cy = (top + bottom) // 2
        for det in detections:
            if det['class'] != 'person':
                continue
            bx1, by1, bx2, by2 = det['bbox']
            if bx1 <= face_cx <= bx2 and by1 <= face_cy <= by2:
                return det
        return None

    def save_unknown_face(self, unknown_id, name):
        enc = self._unknown_faces.get(unknown_id)
        if enc is None:
            return False
        self.face_cache.save_face(name, enc)
        del self._unknown_faces[unknown_id]
        self._cached_face_results = []
        self._last_face_time = 0
        return True

    def try_learn_name_from_transcript(self, text):
        if not self.enable_faces or not self._unknown_faces:
            return
        for pattern in _NAME_PATTERNS:
            match = pattern.search(text)
            if match:
                name = match.group(1).capitalize()
                latest_uid = max(self._unknown_faces.keys())
                if self.save_unknown_face(latest_uid, name):
                    print(f"Auto-learned face: '{name}' from speech")
                    if self._on_transcript:
                        self._on_transcript("System", f"Learned face: {name}")
                return

    def get_frame_b64jpeg(self, max_dim=640, quality=60):
        with self._lock:
            if self.latest_frame is None:
                return None
            frame = self.latest_frame.copy()
        h, w = frame.shape[:2]
        if max(h, w) > max_dim:
            s = max_dim / max(h, w)
            frame = cv2.resize(frame, (int(w * s), int(h * s)))
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buf.tobytes()).decode('utf-8')

    def get_depth_gradient_frame(self, max_dim=640):
        """Build a side-by-side depth + gradient map view for the UI. Returns BGR image or None."""
        depth = self._depth_map
        if depth is None:
            # Placeholder when no depth yet
            placeholder = np.zeros((120, 400, 3), dtype=np.uint8)
            placeholder[:] = (40, 40, 40)
            cv2.putText(placeholder, "Depth: waiting for stream...", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
            return placeholder
        try:
            g = get_gradient_map_from_depth(depth, output_normals=False)
            depth_m = g["depth_m"]
            grad_mag = g["grad_mag"]
            # Normalize for display: depth 0–5 m -> colormap, gradient -> colormap
            depth_display = np.nan_to_num(depth_m, nan=0.0)
            depth_display = np.clip(depth_display, 0, 5.0)
            depth_vis = (depth_display / 5.0 * 255).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
            invalid = (depth_display <= 0) | (depth_display > 5.0)
            depth_vis[invalid] = 0

            g_max = float(np.nanmax(grad_mag)) if np.any(np.isfinite(grad_mag)) else 1.0
            if g_max < 1e-6:
                g_max = 1.0
            grad_vis = (np.clip(grad_mag / g_max, 0, 1) * 255).astype(np.uint8)
            grad_vis = cv2.applyColorMap(grad_vis, cv2.COLORMAP_VIRIDIS)
            grad_vis[invalid] = 0

            # Side by side: [Depth | separator | Gradient], with labels
            h, w = depth_vis.shape[:2]
            pad = 8
            label_h = 28
            sep = np.full((h, 4, 3), 60, dtype=np.uint8)
            row1 = np.hstack([depth_vis, sep, grad_vis])
            row1 = cv2.copyMakeBorder(row1, label_h, pad, pad, pad, cv2.BORDER_CONSTANT, value=(30, 30, 30))
            cv2.putText(row1, "Depth (m)", (pad, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv2.putText(row1, "3D Gradient (edges)", (w + pad + 4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            out = row1
            if max(out.shape[0], out.shape[1]) > max_dim:
                s = max_dim / max(out.shape[0], out.shape[1])
                out = cv2.resize(out, (int(out.shape[1] * s), int(out.shape[0] * s)))
            return out
        except Exception as e:
            print(f"Depth/gradient view error: {e}")
            ph = np.zeros((120, 400, 3), dtype=np.uint8)
            ph[:] = (40, 40, 40)
            cv2.putText(ph, f"Error: {e}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            return ph

    def get_detection_summary(self):
        with self._lock:
            dets = list(self.latest_detections)
        if not dets:
            return ""
        lines = []
        for d in dets:
            dist = f"{d['distance_m']:.1f}m away" if d['distance_m'] else "unknown dist"
            name_str = f" ({d['name']})" if d.get('name') else ""
            lines.append(f"- {d['class']}{name_str} ({d['confidence']:.0%}, {dist})")
        return "Detected objects:\n" + "\n".join(lines)
