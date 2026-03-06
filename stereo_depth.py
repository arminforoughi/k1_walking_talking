"""
Stereo depth estimation and 3D gradient map from stereo cameras or precomputed depth.

Two ways to get depth:
  1. Use existing StereoNet depth from ROS (/StereoNetNode/stereonet_depth) — already in your pipeline.
  2. Compute depth from left + right images (OpenCV stereo) — use when you have both image streams.

From depth we build:
  - 3D point cloud (unproject with camera intrinsics)
  - Depth gradient map (where depth changes quickly = edges/obstacles)
  - Surface normal / slope map (for traversability or 3D structure)
  - Optional 2D world-frame gradient map (e.g. for path planning)
"""

import numpy as np
import cv2
from typing import Optional, Tuple

# Default intrinsics (approximate; replace with calibrated values from camera_info)
# Typical for many stereo rigs: fx ≈ fy, resolution-dependent.
DEFAULT_FX = 400.0
DEFAULT_FY = 400.0
DEFAULT_CX = 320.0
DEFAULT_CY = 240.0
# Depth from StereoNet is usually uint16 in millimeters
DEPTH_SCALE_MM_TO_M = 0.001


def depth_from_stereo(
    left: np.ndarray,
    right: np.ndarray,
    *,
    num_disparities: int = 128,
    block_size: int = 11,
    min_disparity: int = 0,
    baseline_m: float = 0.12,
    fx: float = DEFAULT_FX,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute disparity and depth from rectified left/right stereo images.

    Left and right should be grayscale (or we convert). Images must be rectified
    (horizontal epipolar lines). If your rig is uncalibrated, use cv2.stereoRectify
    with calibration matrices first.

    Returns:
        disparity: (H, W) float32 disparity in pixels
        depth_m: (H, W) float32 depth in meters (invalid = 0 or inf)
    """
    if left.ndim == 3:
        left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    if right.ndim == 3:
        right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    # StereoSGBM usually gives better quality than StereoBM
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disparity = stereo.compute(left, right)
    disparity = np.float32(disparity) / 16.0  # StereoSGBM returns 16x scaled

    # depth = (baseline * fx) / disparity
    invalid = disparity <= 0
    depth_m = np.zeros_like(disparity, dtype=np.float32)
    depth_m[~invalid] = (baseline_m * fx) / disparity[~invalid]
    depth_m[invalid] = 0
    depth_m = np.clip(depth_m, 0, 50.0)  # cap at 50 m

    return disparity, depth_m


def depth_to_point_cloud(
    depth: np.ndarray,
    fx: float = DEFAULT_FX,
    fy: float = DEFAULT_FY,
    cx: float = DEFAULT_CX,
    cy: float = DEFAULT_CY,
    depth_scale: float = DEPTH_SCALE_MM_TO_M,
    depth_is_meters: bool = False,
    mask_valid: bool = True,
) -> np.ndarray:
    """
    Unproject depth image to 3D points in camera frame (X right, Y down, Z forward).

    depth: (H, W) uint16 (mm) or float (m if depth_is_meters=True)
    Returns: (H, W, 3) float32 point cloud in meters; invalid pixels are (0,0,0) if mask_valid.
    """
    if depth_is_meters:
        z = np.float32(depth)
    else:
        z = depth.astype(np.float32) * depth_scale
    h, w = z.shape
    u = np.arange(w, dtype=np.float32)
    v = np.arange(h, dtype=np.float32)
    u, v = np.meshgrid(u, v)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    xyz = np.stack([x, y, z], axis=-1)
    if mask_valid:
        invalid = (z <= 0) | (z > 50.0)
        xyz[invalid] = 0
    return xyz


def depth_to_gradient_map(
    depth: np.ndarray,
    depth_scale: float = DEPTH_SCALE_MM_TO_M,
    depth_is_meters: bool = False,
    kernel_size: int = 5,
    sobel_ksize: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute depth gradient magnitude and direction in image space (edges = obstacles/occlusions).

    depth: (H, W) uint16 (mm) or float (m)
    Returns:
        grad_mag: (H, W) gradient magnitude (large where depth changes fast)
        grad_dir: (H, W) gradient direction in radians [0, 2*pi)
        depth_float: (H, W) depth in meters for convenience
    """
    if depth.dtype == np.uint16:
        depth_float = depth.astype(np.float32) * depth_scale
    else:
        depth_float = np.float32(depth)
    invalid = (depth_float <= 0) | (depth_float > 50.0)
    depth_float[invalid] = np.nan

    sobel_x = cv2.Sobel(depth_float, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    sobel_y = cv2.Sobel(depth_float, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
    grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    grad_dir = np.arctan2(sobel_y, sobel_x)
    grad_dir = np.mod(grad_dir, 2 * np.pi)

    grad_mag[invalid] = 0
    grad_dir[invalid] = 0
    return grad_mag, grad_dir, depth_float


def depth_to_surface_normals(
    depth: np.ndarray,
    fx: float = DEFAULT_FX,
    fy: float = DEFAULT_FY,
    cx: float = DEFAULT_CX,
    cy: float = DEFAULT_CY,
    depth_scale: float = DEPTH_SCALE_MM_TO_M,
    depth_is_meters: bool = False,
    kernel: int = 3,
) -> np.ndarray:
    """
    Compute surface normal map from depth (good for slope/traversability).

    Returns: (H, W, 3) unit normals in camera frame; invalid = (0,0,0).
    """
    xyz = depth_to_point_cloud(
        depth, fx=fx, fy=fy, cx=cx, cy=cy,
        depth_scale=depth_scale, depth_is_meters=depth_is_meters, mask_valid=True
    )
    h, w = xyz.shape[:2]
    # Central differences for dX/du, dX/dv
    dx_du = np.zeros_like(xyz)
    dx_du[:, 1:-1] = (xyz[:, 2:] - xyz[:, :-2]) / 2.0
    dx_dv = np.zeros_like(xyz)
    dx_dv[1:-1, :] = (xyz[2:, :] - xyz[:-2, :]) / 2.0

    normals = np.cross(dx_du, dx_dv)
    length = np.linalg.norm(normals, axis=-1, keepdims=True)
    length = np.maximum(length, 1e-8)
    normals = normals / length
    # Flip so normals point toward camera (Z positive = forward)
    normals[normals[:, :, 2] > 0] *= -1
    invalid = (np.linalg.norm(xyz, axis=-1) < 1e-6)
    normals[invalid] = 0
    return normals


def gradient_map_2d_world(
    depth: np.ndarray,
    robot_theta: float = 0.0,
    fx: float = DEFAULT_FX,
    fy: float = DEFAULT_FY,
    cx: float = DEFAULT_CX,
    cy: float = DEFAULT_CY,
    depth_scale: float = DEPTH_SCALE_MM_TO_M,
    grid_resolution: float = 0.05,
    grid_size: int = 200,
    max_range_m: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rasterize depth + gradient into a 2D world grid (robot-centric, for path planning).

    Robot at center of grid; x forward, y left. Each cell gets max gradient magnitude
    in that region (high = obstacle edge or slope).

    Returns:
        grid_gradient: (grid_size, grid_size) gradient magnitude per cell
        grid_count: (grid_size, grid_size) number of points per cell (for confidence)
    """
    xyz = depth_to_point_cloud(depth, fx=fx, fy=fy, cx=cx, cy=cy, depth_scale=depth_scale)
    grad_mag, _, _ = depth_to_gradient_map(depth, depth_scale=depth_scale)

    h, w = depth.shape
    # Camera frame: X right, Y down, Z forward. Robot frame: X forward, Z up -> so robot X = camera Z, robot Y = -camera X
    x_robot = xyz[:, :, 2]   # forward
    y_robot = -xyz[:, :, 0]  # left
    valid = (xyz[:, :, 2] > 0.1) & (xyz[:, :, 2] < max_range_m)
    x_robot = x_robot[valid]
    y_robot = y_robot[valid]
    g = grad_mag[valid]

    # Rotate by robot heading (e.g. from odometry)
    c, s = np.cos(-robot_theta), np.sin(-robot_theta)
    xw = c * x_robot - s * y_robot
    yw = s * x_robot + c * y_robot

    origin = grid_size // 2
    cells_per_m = 1.0 / grid_resolution
    gx = (origin + xw * cells_per_m).astype(int)
    gy = (origin - yw * cells_per_m).astype(int)

    grid_gradient = np.zeros((grid_size, grid_size), dtype=np.float32)
    grid_count = np.zeros((grid_size, grid_size), dtype=np.int32)
    in_bounds = (gx >= 0) & (gx < grid_size) & (gy >= 0) & (gy < grid_size)
    for i in np.where(in_bounds)[0]:
        grid_gradient[gy[i], gx[i]] = max(grid_gradient[gy[i], gx[i]], g[i])
        grid_count[gy[i], gx[i]] += 1

    return grid_gradient, grid_count


def get_gradient_map_from_depth(
    depth: np.ndarray,
    *,
    fx: float = DEFAULT_FX,
    fy: float = DEFAULT_FY,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    depth_scale: float = DEPTH_SCALE_MM_TO_M,
    depth_is_meters: bool = False,
    output_normals: bool = False,
) -> dict:
    """
    One-shot: from a single depth map (e.g. from StereoNet), compute gradient map and optional normals.

    If cx/cy are None, uses image center from depth shape.
    Returns dict with: grad_mag, grad_dir, depth_m, [normals (H,W,3) if output_normals].
    """
    h, w = depth.shape
    cx = cx if cx is not None else (w - 1) / 2.0
    cy = cy if cy is not None else (h - 1) / 2.0

    grad_mag, grad_dir, depth_m = depth_to_gradient_map(
        depth, depth_scale=depth_scale, depth_is_meters=depth_is_meters
    )
    out = {"grad_mag": grad_mag, "grad_dir": grad_dir, "depth_m": depth_m}
    if output_normals:
        out["normals"] = depth_to_surface_normals(
            depth, fx=fx, fy=fy, cx=cx, cy=cy,
            depth_scale=depth_scale, depth_is_meters=depth_is_meters
        )
    return out


def clean_depth_for_reconstruction(
    depth_mm: np.ndarray,
    seg_instance_map: Optional[np.ndarray] = None,
    instance_info: Optional[list] = None,
    bilateral_d: int = 7,
    sigma_color: float = 0.03,
    sigma_space: float = 6.0,
    gradient_reject: float = 0.05,
    mask_erode_px: int = 3,
    min_depth_m: float = 0.15,
    max_depth_m: float = 6.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Clean raw stereo depth for 3D reconstruction.

    Pipeline:
      1. Median pre-filter (salt-and-pepper from stereo matching)
      2. Bilateral filter (edge-preserving Gaussian denoising)
      3. Depth-gradient rejection (mixed pixels at depth discontinuities)
      4. Per-segment outlier rejection via median absolute deviation
      5. Segment boundary erosion (stereo is worst at mask edges)

    Returns:
        cleaned_m:  (H,W) float32 depth in meters, 0 = invalid
        confidence: (H,W) float32 per-pixel confidence [0, 1]
    """
    dh, dw = depth_mm.shape
    depth_m = depth_mm.astype(np.float32) * 0.001
    valid = (depth_m > min_depth_m) & (depth_m < max_depth_m)
    depth_m[~valid] = 0.0

    depth_m = cv2.medianBlur(depth_m, 3)
    valid = valid & (depth_m > min_depth_m)
    depth_m[~valid] = 0.0

    smoothed = cv2.bilateralFilter(depth_m, bilateral_d, sigma_color, sigma_space)
    smoothed = cv2.medianBlur(smoothed, 3)
    smoothed[~valid] = 0.0

    gx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
    rel_grad = np.sqrt(gx * gx + gy * gy) / np.maximum(smoothed, 0.01)
    grad_ok = rel_grad < gradient_reject

    confidence = np.where(valid, 1.0, 0.0).astype(np.float32)
    safe_ratio = np.clip(rel_grad / max(gradient_reject, 1e-6), 0.0, 1.0)
    confidence *= np.clip(1.0 - safe_ratio * safe_ratio, 0.0, 1.0)

    if seg_instance_map is not None and instance_info:
        seg = seg_instance_map
        if seg.shape != (dh, dw):
            seg = cv2.resize(
                seg.astype(np.float32), (dw, dh),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int16)

        if mask_erode_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (mask_erode_px * 2 + 1, mask_erode_px * 2 + 1),
            )
            for iid in range(len(instance_info)):
                mask_u8 = (seg == iid).astype(np.uint8)
                if np.sum(mask_u8) < 10:
                    continue
                eroded = cv2.erode(mask_u8, kernel)
                boundary = (mask_u8 > 0) & (eroded == 0)
                confidence[boundary] *= 0.1

        for iid in range(len(instance_info)):
            seg_px = (seg == iid) & valid
            if int(np.sum(seg_px)) < 10:
                continue
            seg_d = smoothed[seg_px]
            med = float(np.median(seg_d))
            mad = max(float(np.median(np.abs(seg_d - med))), 0.02)
            outlier = seg_px & (np.abs(smoothed - med) > 3.5 * mad)
            smoothed[outlier] = 0.0
            confidence[outlier] = 0.0

    final_valid = valid & grad_ok & (smoothed > min_depth_m)
    smoothed[~final_valid] = 0.0
    confidence[~final_valid] = 0.0
    return smoothed, confidence


# ---------------------------------------------------------------------------
# Usage with your pipeline
# ---------------------------------------------------------------------------
#
# 1) Using existing StereoNet depth (e.g. in server.py or gemini_live_camera.py):
#
#   from stereo_depth import get_gradient_map_from_depth, gradient_map_2d_world
#
#   depth = fp._depth_map  # (H, W) uint16 mm
#   if depth is not None:
#       g = get_gradient_map_from_depth(depth, output_normals=True)
#       grad_mag, grad_dir = g["grad_mag"], g["grad_dir"]  # 3D gradient in image space
#       normals = g["normals"]  # (H, W, 3) surface normals for slope
#
#       # Optional: 2D world grid for path planning (same style as ObstacleMap)
#       grid_grad, grid_count = gradient_map_2d_world(depth, robot_theta=0.0)
#
# 2) Computing depth from left + right images (e.g. subscribe to both):
#
#   from stereo_depth import depth_from_stereo
#
#   # Subscribe to /image_left_raw and /image_right_raw (or booster_camera_bridge/...)
#   disparity, depth_m = depth_from_stereo(left_frame, right_frame,
#                                         baseline_m=0.12, fx=400.0)
#   # Then use depth_m with get_gradient_map_from_depth(..., depth_is_meters=True)
#
# 3) Camera intrinsics: get from ROS /camera_info or calibrate. Override defaults:
#
#   get_gradient_map_from_depth(depth, fx=420, fy=420, cx=320, cy=240)
