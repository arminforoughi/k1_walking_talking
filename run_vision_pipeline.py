#!/usr/bin/env python3
"""
Run the robot vision pipeline: stereo/depth capture -> point cloud -> segmentation -> semantic map -> VLM.

Usage:
  # With default webcam (synthetic depth for testing)
  python run_vision_pipeline.py

  # With ZED camera
  python run_vision_pipeline.py --backend zed

  # With OAK-D camera (Luxonis)
  python run_vision_pipeline.py --backend oak

  # With ROS (set frames via ROSCapture.set_frame from your ROS node)
  python run_vision_pipeline.py --backend ros

  # Options
  python run_vision_pipeline.py --backend opencv --device 0 --vlm --no-display
"""

import argparse
import sys
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Robot vision pipeline")
    parser.add_argument("--backend", choices=["zed", "ros", "opencv", "oak"], default="opencv",
                        help="oak = OAK-D / OAK-D Lite / OAK-D Pro (Luxonis)")
    parser.add_argument("--device", type=int, default=0, help="Camera device for opencv")
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--stride", type=int, default=2, help="Point cloud subsample stride")
    parser.add_argument("--vlm", action="store_true", help="Enable vision-language model")
    parser.add_argument("--vlm-interval", type=int, default=30, help="Run VLM every N frames")
    parser.add_argument("--no-display", action="store_true", help="Do not open OpenCV window")
    parser.add_argument("--max-frames", type=int, default=0, help="Exit after N frames (0 = infinite)")
    parser.add_argument("--segmenter", type=str, default=None, help="Path to YOLOv8-seg .pt model")
    parser.add_argument("--min-area-pixels", type=int, default=50, help="Drop segments smaller than this (CNN filter)")
    parser.add_argument("--no-cnn-filter", action="store_true", help="Include all depth points; default is only points on segmented objects")
    args = parser.parse_args()

    try:
        from vision_pipeline.pipeline import VisionPipeline
    except ImportError:
        sys.path.insert(0, ".")
        from vision_pipeline.pipeline import VisionPipeline

    pipeline = VisionPipeline(
        capture_backend=args.backend,
        capture_device=args.device,
        segmenter_model=args.segmenter,
        voxel_size=args.voxel_size,
        point_cloud_stride=args.stride,
        use_vlm=args.vlm,
        vlm_interval_frames=args.vlm_interval,
        min_area_pixels=args.min_area_pixels,
        cnn_filter=not args.no_cnn_filter,
    )

    if not pipeline.start():
        print("Failed to start pipeline (check camera and dependencies)", file=sys.stderr)
        return 1

    try:
        import cv2
    except ImportError:
        args.no_display = True

    frame_count = 0
    t0 = time.perf_counter()
    print("Pipeline running. Press 'q' in the window to quit (or Ctrl+C).")

    try:
        while True:
            result = pipeline.step()
            if result is None:
                time.sleep(0.02)
                continue
            frame_count += 1
            if args.max_frames and frame_count >= args.max_frames:
                break

            rgb = result["frame"]
            seg = result["segmentation"]
            if not args.no_display:
                overlay = rgb.copy()
                if seg.masks is not None and seg.masks.any():
                    color = (0, 255, 0)
                    for m in seg.masks:
                        overlay[m] = (overlay[m] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
                cv2.putText(
                    overlay, f"Frame {frame_count} | Map voxels: {pipeline._semantic_map.num_voxels}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                )
                if "vlm_answer" in result:
                    cv2.putText(
                        overlay, "VLM: " + result["vlm_answer"][:80] + "...",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1,
                    )
                cv2.imshow("vision_pipeline", overlay)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if frame_count % 100 == 0:
                elapsed = time.perf_counter() - t0
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"Frames: {frame_count}, Map voxels: {pipeline._semantic_map.num_voxels}, FPS: {fps:.1f}")
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        if not args.no_display:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    print(f"Done. Processed {frame_count} frames. Final map: {pipeline.get_map_point_cloud().n_points} points.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
