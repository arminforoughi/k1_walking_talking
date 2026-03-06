"""
DUSt3R Dense Stereo 3D Reconstructor.

Runs DUSt3R ViTLarge asynchronously in a background thread on left+right
stereo camera pairs. Produces dense 3D point clouds in camera frame.
"""

import sys
import os
import threading
import time
import numpy as np
import torch
import cv2

DUST3R_PATH = os.path.join(os.path.dirname(__file__), 'third_party', 'dust3r')
if DUST3R_PATH not in sys.path:
    sys.path.insert(0, DUST3R_PATH)


class DUSt3RReconstructor:
    """Background DUSt3R stereo reconstruction, producing dense 3D point clouds."""

    MODEL_NAME = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    INPUT_SIZE = 512

    def __init__(self, device=None, interval=3.0):
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device
        self.interval = interval

        self.model = None
        self.ready = False
        self._lock = threading.Lock()
        self._latest_points = None    # (N, 3) float32 world-frame
        self._latest_colors = None    # (N, 3) float32 [0,1]
        self._latest_confidence = None  # (N,) float32
        self._inference_ms = 0.0
        self._point_count = 0

        self._left_frame = None
        self._right_frame = None
        self._robot_pose = None
        self._new_pair = threading.Event()

        self._load_thread = threading.Thread(target=self._load, daemon=True)
        self._load_thread.start()

        self._worker = threading.Thread(target=self._run_loop, daemon=True)
        self._worker.start()

    def _load(self):
        print(f"[DUSt3R] Loading {self.MODEL_NAME} on {self.device}...")
        t0 = time.time()
        from dust3r.model import AsymmetricCroCo3DStereo
        self.model = AsymmetricCroCo3DStereo.from_pretrained(self.MODEL_NAME)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.ready = True
        print(f"[DUSt3R] Ready in {time.time() - t0:.1f}s")

    def set_frames(self, left_bgr, right_bgr, robot_pose=None):
        """Called by frame_processor with new stereo pair."""
        self._left_frame = left_bgr.copy()
        self._right_frame = right_bgr.copy()
        self._robot_pose = robot_pose
        self._new_pair.set()

    def get_latest_points(self):
        """Return latest dense 3D reconstruction or None."""
        with self._lock:
            if self._latest_points is None:
                return None
            return {
                'positions': self._latest_points.copy(),
                'colors': self._latest_colors.copy(),
                'confidence': self._latest_confidence.copy(),
                'count': self._point_count,
            }

    def _run_loop(self):
        while True:
            self._new_pair.wait()
            self._new_pair.clear()
            if not self.ready:
                continue
            left = self._left_frame
            right = self._right_frame
            if left is None or right is None:
                continue
            try:
                self._reconstruct(left, right)
            except Exception as e:
                print(f"[DUSt3R] Reconstruction error: {e}")
            time.sleep(self.interval)

    def _prepare_image(self, bgr_frame):
        """Prepare a BGR frame for DUSt3R: resize to INPUT_SIZE, convert to tensor."""
        from dust3r.utils.image import imread_cv2
        h, w = bgr_frame.shape[:2]
        target = self.INPUT_SIZE
        scale = target / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        # Pad to make square
        new_h = ((new_h + 15) // 16) * 16
        new_w = ((new_w + 15) // 16) * 16
        resized = cv2.resize(bgr_frame, (new_w, new_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(rgb).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        true_shape = torch.tensor([[new_h, new_w]], dtype=torch.long)
        return {
            'img': img_tensor.to(self.device),
            'true_shape': true_shape.to(self.device),
            'idx': 0,
            'instance': '0',
        }, (h, w, new_h, new_w, scale)

    @torch.no_grad()
    def _reconstruct(self, left_bgr, right_bgr):
        t0 = time.time()

        view1, (oh1, ow1, nh1, nw1, s1) = self._prepare_image(left_bgr)
        view2, (oh2, ow2, nh2, nw2, s2) = self._prepare_image(right_bgr)
        view1['idx'] = 0
        view1['instance'] = '0'
        view2['idx'] = 1
        view2['instance'] = '1'

        pred1, pred2 = self.model(view1, view2)

        pts3d_1 = pred1['pts3d'].squeeze(0).cpu().numpy()       # (H, W, 3)
        conf_1 = pred1['conf'].squeeze(0).cpu().numpy()          # (H, W)
        pts3d_2 = pred2['pts3d_in_other_view'].squeeze(0).cpu().numpy()
        conf_2 = pred2['conf'].squeeze(0).cpu().numpy()

        rgb_left = cv2.cvtColor(
            cv2.resize(left_bgr, (nw1, nh1)), cv2.COLOR_BGR2RGB
        ).astype(np.float32) / 255.0
        rgb_right = cv2.cvtColor(
            cv2.resize(right_bgr, (nw2, nh2)), cv2.COLOR_BGR2RGB
        ).astype(np.float32) / 255.0

        pts_all = np.concatenate([
            pts3d_1.reshape(-1, 3), pts3d_2.reshape(-1, 3)
        ], axis=0)
        colors_all = np.concatenate([
            rgb_left.reshape(-1, 3), rgb_right.reshape(-1, 3)
        ], axis=0)
        conf_all = np.concatenate([
            conf_1.reshape(-1), conf_2.reshape(-1)
        ], axis=0)

        # Filter by confidence
        conf_threshold = np.percentile(conf_all, 30)
        valid = conf_all > conf_threshold
        # Also reject extreme outliers
        if np.sum(valid) > 100:
            pts_valid = pts_all[valid]
            med = np.median(pts_valid, axis=0)
            dists = np.linalg.norm(pts_valid - med, axis=1)
            dist_thresh = np.percentile(dists, 95)
            valid_sub = dists < dist_thresh
            pts_out = pts_valid[valid_sub]
            colors_out = colors_all[valid][valid_sub]
            conf_out = conf_all[valid][valid_sub]
        else:
            pts_out = pts_all[valid]
            colors_out = colors_all[valid]
            conf_out = conf_all[valid]

        # Subsample to max 50k points for performance
        max_pts = 50000
        if len(pts_out) > max_pts:
            idx = np.random.choice(len(pts_out), max_pts, replace=False)
            pts_out = pts_out[idx]
            colors_out = colors_out[idx]
            conf_out = conf_out[idx]

        elapsed = (time.time() - t0) * 1000
        self._inference_ms = elapsed

        with self._lock:
            self._latest_points = pts_out.astype(np.float32)
            self._latest_colors = colors_out.astype(np.float32)
            self._latest_confidence = conf_out.astype(np.float32)
            self._point_count = len(pts_out)

        print(f"[DUSt3R] {self._point_count} pts in {elapsed:.0f}ms")
