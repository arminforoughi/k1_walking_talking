"""
SAM2 Instance Segmenter — produces pixel-perfect masks from YOLO bounding boxes.

Uses SAM2.1 hiera-tiny for fast inference on Apple MPS.
"""

import threading
import time
import numpy as np
import torch


class SAM2Segmenter:
    """Wraps SAM2 image predictor for box-prompted segmentation."""

    def __init__(self, model_id="facebook/sam2.1-hiera-tiny", device=None):
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.model_id = model_id
        self.predictor = None
        self.ready = False
        self._lock = threading.Lock()

        t = threading.Thread(target=self._load, daemon=True)
        t.start()

    def _load(self):
        print(f"[SAM2] Loading {self.model_id} on {self.device}...")
        t0 = time.time()
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.predictor = SAM2ImagePredictor.from_pretrained(
            self.model_id, device=self.device
        )
        self.ready = True
        print(f"[SAM2] Ready in {time.time() - t0:.1f}s")

    def segment_from_boxes(self, rgb_image, boxes):
        """
        Given an RGB image and YOLO-format bounding boxes, return pixel-perfect masks.

        Args:
            rgb_image: (H, W, 3) uint8 RGB array
            boxes: list of [x1, y1, x2, y2] in pixel coords

        Returns:
            list of (H, W) bool arrays, one per box
        """
        if not self.ready or len(boxes) == 0:
            return []

        with self._lock:
            self.predictor.set_image(rgb_image)
            masks_out = []
            for box in boxes:
                box_np = np.array(box, dtype=np.float32)
                masks, scores, _ = self.predictor.predict(
                    box=box_np, multimask_output=False
                )
                masks_out.append(masks[0])  # best mask: (H, W) bool
            self.predictor.reset_predictor()
            return masks_out

    def segment_batch(self, rgb_image, boxes):
        """
        Batch version: pass all boxes at once for better throughput.

        Args:
            rgb_image: (H, W, 3) uint8 RGB array
            boxes: list of [x1, y1, x2, y2]

        Returns:
            list of (H, W) bool arrays
        """
        if not self.ready or len(boxes) == 0:
            return []

        with self._lock:
            self.predictor.set_image(rgb_image)
            box_tensor = np.array(boxes, dtype=np.float32)
            masks, scores, _ = self.predictor.predict(
                box=box_tensor,
                multimask_output=False,
            )
            self.predictor.reset_predictor()
            if masks.ndim == 3:
                return [masks[i] for i in range(masks.shape[0])]
            return [masks[0]]
