#!/usr/bin/env python3
"""
Offline TSDF fusion from a recorded stereo directory:

Expected layout:
  RECORD_STEREO_DIR/
    left/000000.png ...
    right/000000.png ...

We compute stereo disparity (OpenCV SGBM) -> metric depth (m), then optionally run
Open3D RGBD odometry to estimate camera motion, and integrate frames into a TSDF.

Example:
  python run_tsdf_stereo_dir.py "$HOME/Documents/RECORD_STEREO_DIR" --output tsdf_mesh.ply
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def _sorted_pngs(p: Path) -> list[Path]:
    return sorted([*p.glob("*.png"), *p.glob("*.jpg"), *p.glob("*.jpeg")])


def _make_sgbm(
    num_disparities: int,
    block_size: int,
    p1: int,
    p2: int,
    min_disparity: int = 0,
):
    import cv2

    num_disparities = int(np.ceil(num_disparities / 16.0) * 16)
    block_size = int(block_size)
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(3, block_size)

    return cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=p1,
        P2=p2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def _stereo_depth_m(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    fx: float,
    baseline_m: float,
    depth_min: float,
    depth_max: float,
    sgbm,
) -> np.ndarray:
    import cv2

    left_g = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    right_g = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
    disp = sgbm.compute(left_g, right_g).astype(np.float32) / 16.0
    disp[disp <= 0.0] = np.nan
    depth = (fx * baseline_m) / disp
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    depth[(depth < depth_min) | (depth > depth_max)] = 0.0
    return depth


def _o3d_intrinsics(w: int, h: int, fx: float, fy: float, cx: float, cy: float):
    import open3d as o3d

    return o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline TSDF from stereo dir (left/right PNG sequences)")
    parser.add_argument("stereo_dir", type=str, help="Path to directory containing left/ and right/")
    parser.add_argument("--output", type=str, default="tsdf_mesh.ply", help="Output mesh path (.ply)")
    parser.add_argument("--voxel-size", type=float, default=0.02, help="TSDF voxel size in meters")
    parser.add_argument("--max-frames", type=int, default=600, help="Process up to N frames (0=all)")
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth frame")
    parser.add_argument("--no-odometry", action="store_true", help="Disable RGBD odometry (all poses identity)")
    parser.add_argument("--fps", type=float, default=0.0, help="Optional playback pacing (0=as fast as possible)")
    parser.add_argument("--depth-min", type=float, default=0.3)
    parser.add_argument("--depth-max", type=float, default=6.0)

    # Camera model (rough defaults; override if you know them)
    parser.add_argument("--fx", type=float, default=0.0, help="Focal length in pixels (0=guess from image size)")
    parser.add_argument("--fy", type=float, default=0.0, help="Focal length in pixels (0=fx)")
    parser.add_argument("--cx", type=float, default=0.0, help="Principal point x (0=w/2)")
    parser.add_argument("--cy", type=float, default=0.0, help="Principal point y (0=h/2)")
    parser.add_argument("--baseline", type=float, default=0.12, help="Stereo baseline in meters (ZED2 ~0.12m)")

    # Stereo SGBM tuning
    parser.add_argument("--num-disparities", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=7)
    args = parser.parse_args()

    stereo_dir = Path(args.stereo_dir).expanduser().resolve()
    left_dir = stereo_dir / "left"
    right_dir = stereo_dir / "right"
    if not left_dir.exists() or not right_dir.exists():
        print(f"Expected {left_dir} and {right_dir}", file=sys.stderr)
        return 2

    left_paths = _sorted_pngs(left_dir)
    right_paths = _sorted_pngs(right_dir)
    n = min(len(left_paths), len(right_paths))
    if n == 0:
        print("No images found under left/ and right/", file=sys.stderr)
        return 2

    if args.max_frames and args.max_frames > 0:
        n = min(n, args.max_frames)

    try:
        import cv2
    except ImportError:
        print("Missing dependency: opencv-python (cv2). Install: pip install opencv-python", file=sys.stderr)
        return 2

    try:
        from reconstruction.tsdf_live import LiveTSDF
    except ImportError:
        sys.path.insert(0, ".")
        from reconstruction.tsdf_live import LiveTSDF

    # Read one frame to set intrinsics defaults
    sample = cv2.imread(str(left_paths[0]), cv2.IMREAD_COLOR)
    if sample is None:
        print(f"Failed to read {left_paths[0]}", file=sys.stderr)
        return 2
    h, w = sample.shape[:2]
    fx = float(args.fx) if args.fx > 0 else float(max(w, h) * 1.2)
    fy = float(args.fy) if args.fy > 0 else fx
    cx = float(args.cx) if args.cx > 0 else (w / 2.0)
    cy = float(args.cy) if args.cy > 0 else (h / 2.0)

    # SGBM penalties (common heuristic)
    p1 = 8 * 1 * args.block_size * args.block_size
    p2 = 32 * 1 * args.block_size * args.block_size
    sgbm = _make_sgbm(args.num_disparities, args.block_size, p1=p1, p2=p2)

    tsdf = LiveTSDF(
        voxel_size=args.voxel_size,
        depth_scale=1.0,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
    )

    # Odometry (optional)
    o3d = None
    intrinsic = None
    if not args.no_odometry:
        try:
            import open3d as o3d  # noqa: F401
            o3d = __import__("open3d")
            intrinsic = _o3d_intrinsics(w, h, fx, fy, cx, cy)
            jac = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()
            odo_opt = o3d.pipelines.odometry.OdometryOption()
        except Exception as e:
            print(f"Odometry disabled (Open3D odometry not available): {e}", file=sys.stderr)
            args.no_odometry = True

    world_T_cam = np.eye(4, dtype=np.float64)
    prev_rgb = None
    prev_depth = None

    t0 = time.perf_counter()
    used = 0
    for i in range(0, n, max(1, args.stride)):
        lp = left_paths[i]
        rp = right_paths[i]
        left = cv2.imread(str(lp), cv2.IMREAD_COLOR)
        right = cv2.imread(str(rp), cv2.IMREAD_COLOR)
        if left is None or right is None:
            continue

        depth = _stereo_depth_m(
            left,
            right,
            fx=fx,
            baseline_m=float(args.baseline),
            depth_min=args.depth_min,
            depth_max=args.depth_max,
            sgbm=sgbm,
        )

        # TSDF expects RGB
        rgb = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)

        if not args.no_odometry and prev_rgb is not None and prev_depth is not None:
            try:
                src_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(prev_rgb),
                    o3d.geometry.Image(prev_depth),
                    depth_scale=1.0,
                    depth_trunc=args.depth_max,
                    convert_rgb_to_intensity=False,
                )
                tgt_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(rgb),
                    o3d.geometry.Image(depth),
                    depth_scale=1.0,
                    depth_trunc=args.depth_max,
                    convert_rgb_to_intensity=False,
                )
                success, trans, _info = o3d.pipelines.odometry.compute_rgbd_odometry(
                    src_rgbd,
                    tgt_rgbd,
                    intrinsic,
                    np.eye(4),
                    jac,
                    odo_opt,
                )
                if success:
                    # trans maps src camera -> tgt camera (tgt_T_src). Accumulate in world frame.
                    world_T_cam = world_T_cam @ np.linalg.inv(np.asarray(trans))
            except Exception:
                pass

        tsdf.integrate(depth, rgb, fx, fy, cx, cy, world_T_cam)
        prev_rgb, prev_depth = rgb, depth
        used += 1

        if used % 50 == 0:
            elapsed = time.perf_counter() - t0
            fps = used / elapsed if elapsed > 0 else 0.0
            print(f"Frames integrated: {used} (i={i}/{n}) | {fps:.1f} fps")

        if args.fps and args.fps > 0:
            time.sleep(max(0.0, (1.0 / args.fps)))

    if used == 0:
        print("No frames integrated.", file=sys.stderr)
        return 1

    out = tsdf.save_mesh(args.output, simplify=True, target_triangles=120_000)
    elapsed = time.perf_counter() - t0
    print(f"Mesh saved: {out} (frames={used}, elapsed={elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

