"""
Robot vision pipeline: stereo/depth capture, point cloud, segmentation, semantic map, VLM.
"""

from .capture import (
    CaptureBackend,
    FrameData,
    create_capture,
    ZEDCapture,
    OAKCapture,
    ROSCapture,
    OpenCVCapture,
)
from .point_cloud import (
    PointCloudFrame,
    depth_to_point_cloud,
    depth_to_point_cloud_with_labels,
    filter_depth,
    filter_point_cloud_outliers,
    transform_point_cloud,
    merge_point_clouds,
    subsample_point_cloud,
)
from .segmenter import Segmenter, SegmentationResult
from .semantic_map import SemanticMap
from .vision_language import VisionLanguageModel, create_vlm
from .pipeline import VisionPipeline

__all__ = [
    "CaptureBackend",
    "FrameData",
    "create_capture",
    "ZEDCapture",
    "OAKCapture",
    "ROSCapture",
    "OpenCVCapture",
    "PointCloudFrame",
    "depth_to_point_cloud",
    "depth_to_point_cloud_with_labels",
    "filter_depth",
    "filter_point_cloud_outliers",
    "transform_point_cloud",
    "merge_point_clouds",
    "subsample_point_cloud",
    "Segmenter",
    "SegmentationResult",
    "SemanticMap",
    "VisionLanguageModel",
    "create_vlm",
    "VisionPipeline",
]
