"""
Main vision pipeline: capture -> segment (CNN) -> point cloud -> semantic map -> optional VLM.
Simplified: uses YOLOv8-seg CNN to define solid objects; only those points go into the map.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .capture import CaptureBackend, FrameData, create_capture
from .point_cloud import (
    PointCloudFrame,
    depth_to_point_cloud_with_labels,
    subsample_point_cloud,
)
from .segmenter import Segmenter, SegmentationResult
from .semantic_map import SemanticMap
from .vision_language import VisionLanguageModel


class VisionPipeline:
    """
    Simplified pipeline: CNN segmentation defines what is solid; only those points are mapped.
    """

    def __init__(
        self,
        capture_backend: str = "opencv",
        capture_device: int = 0,
        capture_kwargs: Optional[dict] = None,
        segmenter_model: Optional[str] = None,
        voxel_size: float = 0.05,
        point_cloud_stride: int = 2,
        map_max_points_per_frame: int = 50_000,
        use_vlm: bool = False,
        vlm_model: str = "Salesforce/blip2-opt-2.7b",
        vlm_interval_frames: int = 30,
        min_area_pixels: int = 50,
        cnn_filter: bool = True,
    ):
        self._capture_backend = capture_backend
        self._capture_device = capture_device
        self._capture_kwargs = capture_kwargs or {}
        self._segmenter_model = segmenter_model
        self._voxel_size = voxel_size
        self._point_cloud_stride = point_cloud_stride
        self._map_max_points_per_frame = map_max_points_per_frame
        self._use_vlm = use_vlm
        self._vlm_model = vlm_model
        self._vlm_interval = vlm_interval_frames
        self._min_area_pixels = min_area_pixels
        self._cnn_filter = cnn_filter

        self._capture: Optional[CaptureBackend] = None
        self._segmenter: Optional[Segmenter] = None
        self._semantic_map: Optional[SemanticMap] = None
        self._vlm: Optional[VisionLanguageModel] = None
        self._frame_count = 0
        self._last_vlm_answer: Optional[str] = None

    def start(self) -> bool:
        """Open capture and initialize models."""
        self._capture = create_capture(
            backend=self._capture_backend,
            device=self._capture_device,
            **self._capture_kwargs,
        )
        if not self._capture.open():
            return False
        self._segmenter = Segmenter(
            model_path=self._segmenter_model,
            min_area_pixels=self._min_area_pixels,
        )
        self._segmenter.load()
        self._semantic_map = SemanticMap(voxel_size=self._voxel_size)
        if self._use_vlm:
            self._vlm = VisionLanguageModel(model_name=self._vlm_model)
            self._vlm.load()
        return True

    def _segmentation_to_class_map(self, seg: SegmentationResult, height: int, width: int) -> np.ndarray:
        """Build (H, W) array of global class ids (0 = background)."""
        class_map = np.zeros((height, width), dtype=np.int32)
        if seg.label_map.shape[0] != height or seg.label_map.shape[1] != width:
            import cv2
            label_map = cv2.resize(
                seg.label_map.astype(np.float32), (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int32)
        else:
            label_map = seg.label_map
        for idx in range(1, len(seg.class_ids) + 1):
            mask = label_map == idx
            if idx <= len(seg.class_ids):
                class_map[mask] = seg.class_ids[idx - 1]
        return class_map

    def step(
        self,
        pose_R: Optional[np.ndarray] = None,
        pose_t: Optional[np.ndarray] = None,
    ) -> Optional[dict]:
        """
        Run one pipeline step: grab frame -> segment -> point cloud -> update map.
        Optionally run VLM every N frames.
        Returns dict with keys: frame, point_cloud, semantic_map, vlm_answer (if run).
        """
        if self._capture is None or not self._capture.is_opened:
            return None
        frame_data = self._capture.grab()
        if frame_data is None:
            return None

        self._frame_count += 1
        rgb = frame_data.rgb_left
        depth = frame_data.depth
        h, w = rgb.shape[:2]
        fx, fy = frame_data.fx, frame_data.fy
        cx, cy = frame_data.cx, frame_data.cy

        # Segment
        seg = self._segmenter.segment(rgb)
        class_map = self._segmentation_to_class_map(seg, h, w)

        # Align depth to RGB if needed
        if depth.shape[0] != h or depth.shape[1] != w:
            import cv2
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

        # Labeled point cloud (CNN already filtered small blobs via min_area_pixels)
        pc = depth_to_point_cloud_with_labels(
            depth, fx, fy, cx, cy,
            rgb=rgb,
            label_map=class_map,
            depth_min=frame_data.depth_min,
            depth_max=frame_data.depth_max,
            stride=self._point_cloud_stride,
        )
        # Use CNN to filter: only keep points on segmented objects (labels > 0)
        if self._cnn_filter and pc.labels is not None and pc.xyz.shape[0] > 0:
            keep = pc.labels > 0
            if np.any(keep):
                pc = PointCloudFrame(
                    xyz=pc.xyz[keep],
                    rgb=pc.rgb[keep] if pc.rgb is not None else None,
                    labels=pc.labels[keep],
                    mask_valid=pc.mask_valid[keep] if pc.mask_valid is not None else None,
                )
        pc_small = subsample_point_cloud(pc, voxel_size=self._voxel_size, max_points=self._map_max_points_per_frame)

        # Update global map
        self._semantic_map.update(pc_small, pose_R=pose_R, pose_t=pose_t, max_points=self._map_max_points_per_frame)

        out = {
            "frame": rgb,
            "frame_data": frame_data,
            "segmentation": seg,
            "point_cloud": pc,
            "semantic_map": self._semantic_map,
        }

        if self._use_vlm and self._vlm is not None and self._frame_count % self._vlm_interval == 0:
            self._last_vlm_answer = self._vlm.describe(rgb)
            out["vlm_answer"] = self._last_vlm_answer

        return out

    def get_map_point_cloud(self) -> PointCloudFrame:
        """Return current global semantic map as a point cloud."""
        if self._semantic_map is None:
            return PointCloudFrame(xyz=np.zeros((0, 3)))
        return self._semantic_map.get_point_cloud()

    def query_scene(self, question: str) -> Optional[str]:
        """Grab current frame and ask the VLM a question."""
        if self._capture is None or self._vlm is None:
            return None
        frame_data = self._capture.grab()
        if frame_data is None:
            return None
        return self._vlm.query(frame_data.rgb_left, question)

    def stop(self) -> None:
        if self._capture is not None:
            self._capture.close()
            self._capture = None
