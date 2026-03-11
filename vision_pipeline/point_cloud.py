"""
Real-time 3D point cloud generation from depth and RGB.
Back-projects depth + optional per-point labels for semantic map.
Includes depth denoising and point cloud outlier removal for cleaner reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PointCloudFrame:
    """One frame of 3D points with optional color and labels."""

    xyz: np.ndarray   # (N, 3) in camera frame
    rgb: Optional[np.ndarray] = None   # (N, 3) 0–255
    labels: Optional[np.ndarray] = None  # (N,) int class ids, -1 = unlabeled
    mask_valid: Optional[np.ndarray] = None  # (N,) bool

    @property
    def n_points(self) -> int:
        return self.xyz.shape[0]


def filter_depth(
    depth: np.ndarray,
    method: str = "bilateral",
    bilateral_d: int = 5,
    bilateral_sigma_color: float = 50.0,
    bilateral_sigma_space: float = 50.0,
    median_ksize: int = 5,
) -> np.ndarray:
    """
    Denoise depth map while preserving edges (good for solid object boundaries).
    - bilateral: edge-preserving smoothing (recommended).
    - median: removes salt-and-pepper; use small ksize (3 or 5).
    """
    import cv2
    depth_f = np.float32(depth)
    valid = depth_f > 1e-6
    if method == "bilateral":
        out = cv2.bilateralFilter(
            depth_f, d=bilateral_d,
            sigmaColor=bilateral_sigma_color,
            sigmaSpace=bilateral_sigma_space,
        )
    elif method == "median":
        out = cv2.medianBlur(depth_f.astype(np.float32), median_ksize)
    else:
        out = depth_f.copy()
    out[~valid] = 0
    return out


def statistical_outlier_removal(
    xyz: np.ndarray,
    k: int = 20,
    std_ratio: float = 2.0,
    *,
    labels: Optional[np.ndarray] = None,
    rgb: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove points whose mean distance to k nearest neighbors is too large (noise).
    Returns (xyz_filtered, keep_mask) of length N.
    If labels/rgb provided, also returns filtered versions via keep_mask.
    Uses scipy.spatial.cKDTree if available, else a simple distance threshold.
    """
    n = xyz.shape[0]
    if n == 0:
        keep = np.array([], dtype=bool)
        return xyz, keep
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(xyz)
        dd, _ = tree.query(xyz, k=min(k + 1, n), workers=-1)
        mean_d = np.mean(dd[:, 1:], axis=1)
        thresh = np.median(mean_d) + std_ratio * np.std(mean_d)
        keep = mean_d <= thresh
    except ImportError:
        # Fallback: keep all (no scipy)
        keep = np.ones(n, dtype=bool)
    return xyz[keep], keep


def radius_outlier_removal(
    xyz: np.ndarray,
    radius: float,
    min_neighbors: int = 5,
    *,
    labels: Optional[np.ndarray] = None,
    rgb: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove points that have fewer than min_neighbors within radius (isolated noise).
    Returns (xyz_filtered, keep_mask). Optional labels/rgb filtered by keep_mask.
    """
    n = xyz.shape[0]
    if n == 0:
        return xyz, np.array([], dtype=bool)
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(xyz)
        counts = tree.query_ball_point(xyz, radius, return_length=True)
        keep = np.asarray(counts, dtype=np.int32) >= min_neighbors
    except ImportError:
        keep = np.ones(n, dtype=bool)
    return xyz[keep], keep


def filter_point_cloud_outliers(
    pc: PointCloudFrame,
    method: str = "statistical",
    k: int = 20,
    std_ratio: float = 2.0,
    radius: float = 0.03,
    min_neighbors: int = 5,
) -> PointCloudFrame:
    """
    Remove noisy points from a point cloud. Returns new PointCloudFrame.
    - statistical: drop points far from k-NN (good for general noise).
    - radius: drop points with fewer than min_neighbors within radius (good for flyers).
    """
    xyz = pc.xyz
    if method == "radius":
        xyz_out, keep = radius_outlier_removal(
            xyz, radius=radius, min_neighbors=min_neighbors,
        )
    else:
        xyz_out, keep = statistical_outlier_removal(
            xyz, k=k, std_ratio=std_ratio,
        )
    rgb_out = pc.rgb[keep] if pc.rgb is not None else None
    labels_out = pc.labels[keep] if pc.labels is not None else None
    mask_out = pc.mask_valid[keep] if pc.mask_valid is not None else None
    return PointCloudFrame(xyz=xyz_out, rgb=rgb_out, labels=labels_out, mask_valid=mask_out)


def depth_to_point_cloud(
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    rgb: Optional[np.ndarray] = None,
    depth_min: float = 0.1,
    depth_max: float = 10.0,
    stride: int = 1,
) -> PointCloudFrame:
    """
    Back-project depth to 3D points in camera frame.
    Optionally attach RGB. Use stride to subsample for speed.
    """
    h, w = depth.shape
    # Pixel coordinates
    u = np.arange(0, w, stride, dtype=np.float32)
    v = np.arange(0, h, stride, dtype=np.float32)
    u, v = np.meshgrid(u, v)
    z = depth[::stride, ::stride].astype(np.float32).ravel()
    valid = (z >= depth_min) & (z <= depth_max) & (z > 1e-6)
    u = u.ravel()[valid]
    v = v.ravel()[valid]
    z = z[valid]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    xyz = np.stack([x, y, z], axis=1)

    rgb_out = None
    if rgb is not None:
        rgb_sub = rgb[::stride, ::stride].reshape(-1, 3)[valid]
        rgb_out = rgb_sub.astype(np.uint8)

    return PointCloudFrame(
        xyz=xyz,
        rgb=rgb_out,
        mask_valid=np.ones(xyz.shape[0], dtype=bool),
    )


def depth_to_point_cloud_with_labels(
    depth: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    rgb: Optional[np.ndarray] = None,
    label_map: Optional[np.ndarray] = None,
    depth_min: float = 0.1,
    depth_max: float = 10.0,
    stride: int = 1,
) -> PointCloudFrame:
    """
    Same as depth_to_point_cloud but attaches per-pixel labels from label_map
    (same shape as depth, int per pixel). -1 or 0 = unlabeled.
    """
    pc = depth_to_point_cloud(
        depth, fx, fy, cx, cy, rgb,
        depth_min=depth_min, depth_max=depth_max, stride=stride,
    )
    if label_map is not None:
        labels_flat = label_map[::stride, ::stride].ravel()
        h, w = depth.shape
        u = np.arange(0, w, stride)
        v = np.arange(0, h, stride)
        u, v = np.meshgrid(u, v)
        z = depth[::stride, ::stride].astype(np.float32).ravel()
        valid = (z >= depth_min) & (z <= depth_max) & (z > 1e-6)
        labels = labels_flat[valid]
        pc.labels = labels
    return pc


def transform_point_cloud(
    xyz: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Apply rigid transform: xyz_new = R @ xyz.T + t. R (3,3), t (3,)."""
    return (R @ xyz.T).T + t


def merge_point_clouds(
    frames: list[PointCloudFrame],
    poses: Optional[list[tuple[np.ndarray, np.ndarray]]] = None,
) -> PointCloudFrame:
    """
    Merge multiple point cloud frames into one.
    poses: list of (R, t) per frame; if None, assume identity (all in same frame).
    """
    if not frames:
        return PointCloudFrame(xyz=np.zeros((0, 3)))

    if poses is None:
        poses = [(np.eye(3), np.zeros(3))] * len(frames)
    assert len(poses) == len(frames)

    xyz_list, rgb_list, label_list = [], [], []
    for frame, (R, t) in zip(frames, poses):
        xyz = transform_point_cloud(frame.xyz, R, t)
        xyz_list.append(xyz)
        if frame.rgb is not None:
            rgb_list.append(frame.rgb)
        if frame.labels is not None:
            label_list.append(frame.labels)

    xyz = np.vstack(xyz_list)
    rgb = np.vstack(rgb_list) if rgb_list and len(rgb_list) == len(frames) else None
    labels = np.concatenate(label_list) if label_list and len(label_list) == len(frames) else None
    return PointCloudFrame(xyz=xyz, rgb=rgb, labels=labels)


def subsample_point_cloud(
    pc: PointCloudFrame,
    voxel_size: float = 0.02,
    max_points: Optional[int] = None,
) -> PointCloudFrame:
    """
    Voxel-grid subsampling for faster map updates.
    Each voxel keeps one point (first or centroid). Optionally cap total points.
    """
    xyz = pc.xyz
    if voxel_size <= 0:
        if max_points and xyz.shape[0] > max_points:
            idx = np.random.choice(xyz.shape[0], max_points, replace=False)
            return PointCloudFrame(
                xyz=xyz[idx],
                rgb=pc.rgb[idx] if pc.rgb is not None else None,
                labels=pc.labels[idx] if pc.labels is not None else None,
            )
        return pc

    voxel_idx = np.floor(xyz / voxel_size).astype(np.int32)
    voxel_keys = voxel_idx[:, 0] + voxel_idx[:, 1] * 1000000 + voxel_idx[:, 2] * 1000000000
    unq, inv = np.unique(voxel_keys, return_inverse=True)
    n_vox = len(unq)
    idx_rep = np.zeros(n_vox, dtype=np.int64)
    for i in range(n_vox):
        idx_rep[i] = np.where(inv == i)[0][0]
    xyz_out = xyz[idx_rep]
    rgb_out = pc.rgb[idx_rep] if pc.rgb is not None else None
    labels_out = pc.labels[idx_rep] if pc.labels is not None else None
    if max_points and n_vox > max_points:
        idx = np.random.choice(n_vox, max_points, replace=False)
        xyz_out = xyz_out[idx]
        rgb_out = rgb_out[idx] if rgb_out is not None else None
        labels_out = labels_out[idx] if labels_out is not None else None
    return PointCloudFrame(xyz=xyz_out, rgb=rgb_out, labels=labels_out)
