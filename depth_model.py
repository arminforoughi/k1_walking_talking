"""Neural depth estimation using Depth Anything V2.

Produces dramatically cleaner, more detailed depth maps than noisy stereo.
Used to refine or replace StereoNet depth for high-quality 3D reconstruction.

Models (auto-downloaded on first use):
  small  — 25M params, ~30ms/frame on MPS, good quality
  base   — 98M params, ~80ms/frame on MPS, better detail
  large  — 335M params, ~250ms/frame on MPS, best quality

Usage:
    python server.py --depth-model small    # enable neural depth (fast)
    python server.py --depth-model base     # better quality
"""

import numpy as np
import cv2
import time
import threading


class DepthEstimator:
    """Monocular depth via Depth Anything V2, fused with stereo for metric scale."""

    MODELS = {
        'small': 'depth-anything/Depth-Anything-V2-Small-hf',
        'base':  'depth-anything/Depth-Anything-V2-Base-hf',
        'large': 'depth-anything/Depth-Anything-V2-Large-hf',
    }

    def __init__(self, model_size='small', max_input_dim=518):
        self.model_size = model_size
        self.max_input_dim = max_input_dim
        self._lock = threading.Lock()
        self._pipe = None
        self._ready = False
        self._loading = False
        self._device_name = 'unknown'
        self._inference_ms = 0

        self._scale_a = -3000.0
        self._scale_b = 4000.0
        self._calibrated = False

        t = threading.Thread(target=self._load, daemon=True)
        t.start()

    def _load(self):
        self._loading = True
        try:
            import torch
            from transformers import pipeline as hf_pipeline

            if torch.cuda.is_available():
                device = 0
                self._device_name = f'cuda:{torch.cuda.get_device_name(0)}'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
                self._device_name = 'mps (Apple Metal)'
            else:
                device = -1
                self._device_name = 'cpu'

            model_id = self.MODELS.get(self.model_size, self.MODELS['small'])
            print(f"[DepthModel] Loading {model_id} on {self._device_name}...")

            self._pipe = hf_pipeline(
                task="depth-estimation",
                model=model_id,
                device=device,
            )

            self._ready = True
            print(f"[DepthModel] Ready ({self.model_size}, {self._device_name})")
        except Exception as e:
            print(f"[DepthModel] Failed to load: {e}")
            print("[DepthModel] Install: pip install transformers torch")
        finally:
            self._loading = False

    @property
    def ready(self):
        return self._ready

    @property
    def status(self):
        if self._ready:
            return f"Depth Anything V2 ({self.model_size}) on {self._device_name} [{self._inference_ms:.0f}ms]"
        if self._loading:
            return f"Loading Depth Anything V2 ({self.model_size})..."
        return "Depth model not available"

    def _run_inference(self, bgr_frame):
        """Run Depth Anything V2 on a BGR frame.
        Returns (H, W) float32 with relative depth values (higher = closer)."""
        from PIL import Image

        h, w = bgr_frame.shape[:2]

        scale_factor = min(self.max_input_dim / max(h, w), 1.0)
        if scale_factor < 1.0:
            small = cv2.resize(bgr_frame, (int(w * scale_factor), int(h * scale_factor)))
        else:
            small = bgr_frame

        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        t0 = time.time()
        with self._lock:
            result = self._pipe(pil_img)
        self._inference_ms = (time.time() - t0) * 1000

        if 'predicted_depth' in result:
            import torch
            depth_tensor = result['predicted_depth']
            if isinstance(depth_tensor, torch.Tensor):
                depth_np = depth_tensor.squeeze().cpu().numpy().astype(np.float32)
            else:
                depth_np = np.array(depth_tensor, dtype=np.float32)
        else:
            depth_np = np.array(result['depth'], dtype=np.float32)

        if depth_np.shape != (h, w):
            depth_np = cv2.resize(depth_np, (w, h), interpolation=cv2.INTER_LINEAR)

        return depth_np

    def _calibrate_scale(self, neural_depth, stereo_mm):
        """Fit linear mapping: stereo_mm = a * neural + b using robust statistics."""
        stereo_f = stereo_mm.astype(np.float32)
        valid = (stereo_f > 150) & (stereo_f < 8000) & np.isfinite(neural_depth) & (neural_depth > 1e-3)

        if np.sum(valid) < 200:
            return

        nv = neural_depth[valid]
        sv = stereo_f[valid]

        n_med = np.median(nv)
        near = nv > n_med
        far = nv <= n_med

        if np.sum(near) > 50 and np.sum(far) > 50:
            n_near, s_near = np.median(nv[near]), np.median(sv[near])
            n_far, s_far = np.median(nv[far]), np.median(sv[far])

            dn = n_near - n_far
            if abs(dn) > 1e-6:
                a = (s_near - s_far) / dn
                b = s_near - a * n_near
                alpha = 0.3 if self._calibrated else 1.0
                self._scale_a = self._scale_a * (1 - alpha) + a * alpha
                self._scale_b = self._scale_b * (1 - alpha) + b * alpha
                self._calibrated = True

    def refine_depth(self, bgr_frame, stereo_depth_mm):
        """Enhance stereo depth using neural depth estimation.

        Returns (H, W) uint16 depth in millimeters - same format as input.
        Neural depth provides clean structure; stereo provides metric scale.
        """
        if not self._ready:
            return stereo_depth_mm

        try:
            neural = self._run_inference(bgr_frame)
        except Exception as e:
            print(f"[DepthModel] Inference error: {e}")
            return stereo_depth_mm

        sh, sw = stereo_depth_mm.shape
        if neural.shape != (sh, sw):
            neural = cv2.resize(neural, (sw, sh), interpolation=cv2.INTER_LINEAR)

        self._calibrate_scale(neural, stereo_depth_mm)

        neural_mm = (self._scale_a * neural + self._scale_b).astype(np.float32)
        neural_mm = np.clip(neural_mm, 100, 10000)

        stereo_f = stereo_depth_mm.astype(np.float32)
        valid_stereo = (stereo_f > 150) & (stereo_f < 8000)

        result = np.where(
            valid_stereo,
            0.6 * neural_mm + 0.4 * stereo_f,
            neural_mm,
        )

        return np.clip(result, 0, 65000).astype(np.uint16)

    def estimate_depth(self, bgr_frame):
        """Estimate depth from RGB alone (no stereo reference).
        Returns (H, W) uint16 depth in millimeters using default indoor scale."""
        if not self._ready:
            return None

        try:
            neural = self._run_inference(bgr_frame)
        except Exception as e:
            print(f"[DepthModel] Inference error: {e}")
            return None

        d_min, d_max = neural.min(), neural.max()
        if d_max - d_min < 1e-6:
            return None

        norm = (neural - d_min) / (d_max - d_min)
        depth_m = 0.2 + norm * 5.8
        return (depth_m * 1000).clip(100, 8000).astype(np.uint16)
