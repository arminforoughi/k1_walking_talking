"""
Lightweight segmentation for object labeling.
Uses YOLOv8-seg for real-time instance segmentation; outputs class IDs and masks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class SegmentationResult:
    """Per-frame segmentation: class IDs and mask per pixel."""

    label_map: np.ndarray   # (H, W) int, 0 = background, 1..K = class id
    class_ids: np.ndarray   # (K,) unique class ids in this frame
    class_names: list       # names for class_ids
    masks: Optional[np.ndarray] = None  # (K, H, W) bool if available
    scores: Optional[np.ndarray] = None  # (K,) confidence


class Segmenter:
    """YOLOv8-seg wrapper for real-time object segmentation."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_det: int = 50,
        min_area_pixels: int = 50,
    ):
        self._model_path = model_path
        self._device = device
        self._conf = conf_threshold
        self._iou = iou_threshold
        self._max_det = max_det
        self._min_area_pixels = max(1, min_area_pixels)
        self._model = None
        self._names = []

    def load(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError:
            raise RuntimeError(
                "Segmenter requires ultralytics. Install: pip install ultralytics"
            )
        path = self._model_path
        if not path:
            path = Path(__file__).resolve().parent.parent / "yolov8m-seg.pt"
        if not Path(path).exists():
            path = "yolov8m-seg.pt"  # will download
        self._model = YOLO(str(path))
        self._names = list(self._model.names.values()) if hasattr(self._model, "names") else []

    def segment(self, image: np.ndarray) -> SegmentationResult:
        """
        Run segmentation on BGR image. Returns label_map (H,W) with 0=bg, 1..K=class index.
        Class index here is 1-based in label_map; class_ids are model's original IDs.
        """
        if self._model is None:
            self.load()
        results = self._model.predict(
            image,
            conf=self._conf,
            iou=self._iou,
            max_det=self._max_det,
            device=self._device,
            verbose=False,
        )
        H, W = image.shape[:2]
        label_map = np.zeros((H, W), dtype=np.int32)
        class_ids_list = []
        class_names_list = []
        masks_list = []
        scores_list = []

        for r in results:
            if r.masks is None:
                continue
            boxes = r.boxes
            masks = r.masks
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                mask = masks.data[i].cpu().numpy()
                if mask.shape[0] != H or mask.shape[1] != W:
                    import cv2
                    mask = cv2.resize(
                        mask.astype(np.float32), (W, H),
                        interpolation=cv2.INTER_LINEAR,
                    )
                mask_bool = mask > 0.5
                area = int(np.sum(mask_bool))
                if area < self._min_area_pixels:
                    continue
                # In label_map we use 1-based index so 0 stays background
                idx = len(class_ids_list) + 1
                label_map[mask_bool] = idx
                class_ids_list.append(cls_id)
                name = self._names[cls_id] if cls_id < len(self._names) else f"class_{cls_id}"
                class_names_list.append(name)
                masks_list.append(mask_bool)
                scores_list.append(conf)

        return SegmentationResult(
            label_map=label_map,
            class_ids=np.array(class_ids_list, dtype=np.int32) if class_ids_list else np.array([], dtype=np.int32),
            class_names=class_names_list,
            masks=np.stack(masks_list, axis=0) if masks_list else None,
            scores=np.array(scores_list, dtype=np.float32) if scores_list else None,
        )