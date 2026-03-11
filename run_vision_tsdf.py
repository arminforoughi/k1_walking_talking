#!/usr/bin/env python3
"""
Run vision pipeline with live TSDF: build a smooth 3D mesh from depth as the robot/camera moves.
Good for solid objects >~2 inches; use for table edges, beams, obstacles.
Saves mesh (.ply) and optionally plane contours (table tops, etc.).

Usage:
  # Default webcam (synthetic depth)
  python run_vision_tsdf.py

  # ZED or OAK-D
  python run_vision_tsdf.py --backend zed
  python run_vision_tsdf.py --backend oak

  # Voxel size: smaller = finer but heavier; ~1.5cm is a good balance for smooth rough models
  python run_vision_tsdf.py --voxel-size 0.015 --output scene.ply

  # Save plane contours (table edges) to a JSON file
  python run_vision_tsdf.py --save-planes planes.json
"""

import argparse
import json
import sys
import time

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Vision pipeline + live TSDF for rough 3D mesh and edges")
    parser.add_argument("--backend", choices=["zed", "ros", "opencv", "oak"], default="opencv")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--voxel-size", type=float, default=0.015,
                        help="TSDF voxel size in m (~0.01–0.02 for smooth surfaces)")
    parser.add_argument("--output", type=str, default="tsdf_mesh.ply", help="Output mesh path")
    parser.add_argument("--save-planes", type=str, default=None,
                        help="Save plane contours (table edges) to this JSON file")
    parser.add_argument("--max-frames", type=int, default=0, help="Integrate N frames then stop (0 = until 'q')")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    try:
        from vision_pipeline.capture import create_capture
        from reconstruction.tsdf_live import LiveTSDF, extract_plane_contours
    except ImportError:
        sys.path.insert(0, ".")
        from vision_pipeline.capture import create_capture
        from reconstruction.tsdf_live import LiveTSDF, extract_plane_contours

    capture = create_capture(backend=args.backend, device=args.device)
    if not capture.open():
        print("Failed to open capture", file=sys.stderr)
        return 1

    tsdf = LiveTSDF(
        voxel_size=args.voxel_size,
        depth_min=0.2,
        depth_max=5.0,
    )
    # Identity pose: camera frame = world frame (robot can pass pose_R, pose_t later)
    world_T_cam = np.eye(4)
    frame_count = 0
    t0 = time.perf_counter()

    try:
        import cv2
    except ImportError:
        args.no_display = True

    print("Integrating depth into TSDF. Move the camera slowly for best results. Press 'q' to save and quit.")

    try:
        while True:
            frame_data = capture.grab()
            if frame_data is None:
                time.sleep(0.02)
                continue
            rgb = frame_data.rgb_left
            # Capture is BGR; TSDF expects RGB
            try:
                import cv2 as _cv2
                rgb = _cv2.cvtColor(rgb, _cv2.COLOR_BGR2RGB)
            except Exception:
                pass
            depth = frame_data.depth.astype(np.float32)
            if frame_data.depth_scale != 1.0:
                depth = depth * frame_data.depth_scale
            fx = frame_data.fx or (max(rgb.shape[1], rgb.shape[0]) * 1.2)
            fy = frame_data.fy or fx
            cx = frame_data.cx or (rgb.shape[1] / 2.0)
            cy = frame_data.cy or (rgb.shape[0] / 2.0)
            tsdf.integrate(depth, rgb, fx, fy, cx, cy, world_T_cam)
            frame_count += 1

            if not args.no_display:
                cv2.imshow("tsdf", frame_data.rgb_left)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.max_frames and frame_count >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        capture.close()

    if frame_count == 0:
        print("No frames integrated.", file=sys.stderr)
        return 1

    mesh_path = tsdf.save_mesh(args.output, simplify=True, target_triangles=100_000)
    print(f"Mesh saved: {mesh_path} ({frame_count} frames)")

    if args.save_planes:
        mesh = tsdf.get_mesh(simplify=True, target_triangles=80_000)
        contours = extract_plane_contours(
            mesh,
            distance_threshold=0.02,
            min_inliers=80,
            max_planes=15,
        )
        out = []
        for plane_eq, boundary_xyz, _ in contours:
            out.append({
                "plane": plane_eq.tolist(),
                "boundary_xyz": boundary_xyz.tolist(),
            })
        with open(args.save_planes, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Plane contours (e.g. table edges) saved: {args.save_planes} ({len(out)} planes)")

    if not args.no_display:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
