"""Monocular depth estimation using Depth Anything V2 for clean 3D reconstruction.

Provides much cleaner depth maps than raw stereo, which dramatically reduces
noise in 3D scene reconstruction. Uses stereo depth as a metric reference to
convert the relative monocular output into calibrated meters.
"""

import time
import threading
import numpy as np
import cv2


class DepthEstimator:
    """Lazy-loading wrapper for Depth Anything V2 monocular depth."""

    def __init__(self, model_size="Small"):
        self._model = None
        self._processor = None
        self._model_name = f"depth-anything/Depth-Anything-V2-{model_size}-hf"
        self._lock = threading.Lock()
        self._device = None
        self._torch = None
        self._loading = False
        self._load_error = None
        self._scale = 1.0
        self._offset = 0.0
        self._calibrated = False

    def _ensure_loaded(self):
        if self._model is not None:
            return True
        if self._load_error:
            return False
        if self._loading:
            return False
        self._loading = True
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            self._torch = torch
            if torch.backends.mps.is_available():
                self._device = torch.device("mps")
            elif torch.cuda.is_available():
                self._device = torch.device("cuda")
            else:
                self._device = torch.device("cpu")

            print(f"[DepthModel] Loading {self._model_name} on {self._device}...")
            self._processor = AutoImageProcessor.from_pretrained(self._model_name)
            self._model = AutoModelForDepthEstimation.from_pretrained(self._model_name)
            self._model.to(self._device)
            self._model.eval()
            print("[DepthModel] Ready!")
            return True
        except Exception as e:
            self._load_error = str(e)
            print(f"[DepthModel] Failed to load: {e}")
            return False
        finally:
            self._loading = False

    def load_async(self):
        """Start loading model in a background thread."""
        t = threading.Thread(target=self._ensure_loaded, daemon=True)
        t.start()

    @property
    def is_ready(self):
        return self._model is not None

    def predict(self, bgr_frame, target_h=None, target_w=None):
        """Predict relative depth from a BGR frame.

        Returns (H, W) float32 where larger values = closer.
        Output is at target resolution, or depth_map resolution if provided later.
        """
        with self._lock:
            if not self._ensure_loaded():
                return None

        torch = self._torch
        from PIL import Image

        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        inputs = self._processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            prediction = outputs.predicted_depth

        oh = target_h or bgr_frame.shape[0]
        ow = target_w or bgr_frame.shape[1]
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(oh, ow),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        return prediction.cpu().numpy().astype(np.float32)

    def align_to_metric(self, mono_relative, stereo_depth_mm,
                        min_depth_m=0.15, max_depth_m=8.0):
        """Align monocular relative depth to metric using stereo as reference.

        mono_relative: (H,W) float32 from predict() — larger = closer (inverse depth)
        stereo_depth_mm: (H,W) uint16 in mm

        Returns: (H,W) float32 metric depth in meters.
        """
        stereo_m = stereo_depth_mm.astype(np.float32) * 0.001

        if mono_relative.shape != stereo_m.shape:
            mono_relative = cv2.resize(
                mono_relative,
                (stereo_m.shape[1], stereo_m.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        valid = (
            (stereo_m > min_depth_m) & (stereo_m < max_depth_m)
            & (mono_relative > 1e-6) & np.isfinite(mono_relative)
        )
        n_valid = int(np.sum(valid))
        if n_valid < 100:
            return stereo_m

        m_vals = mono_relative[valid]
        s_vals = stereo_m[valid]

        # DepthAnything outputs disparity-like values: larger = closer
        # So metric_depth ≈ scale / mono + offset
        inv_m = 1.0 / m_vals
        A = np.column_stack([inv_m, np.ones_like(inv_m)])
        result, _, _, _ = np.linalg.lstsq(A, s_vals, rcond=None)
        scale, offset = float(result[0]), float(result[1])

        if scale <= 0:
            A2 = np.column_stack([m_vals, np.ones_like(m_vals)])
            result2, _, _, _ = np.linalg.lstsq(A2, s_vals, rcond=None)
            scale2, offset2 = float(result2[0]), float(result2[1])
            if abs(scale2) < 1e-6:
                return stereo_m
            metric = scale2 * mono_relative + offset2
        else:
            safe_mono = np.maximum(mono_relative, 1e-6)
            metric = scale / safe_mono + offset

        metric = np.clip(metric, 0, max_depth_m).astype(np.float32)
        metric[mono_relative <= 1e-6] = 0
        return metric
