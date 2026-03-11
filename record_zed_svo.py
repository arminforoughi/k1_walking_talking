#!/usr/bin/env python3
"""
Record stereo footage from a ZED camera to an SVO file.

SVO is Stereolabs' native format: it stores left + right images, depth, and
metadata (timestamps, IMU if available). You can later convert to AVI or
image sequences with the ZED SDK's SVO Export tool or pyzed.

Usage:
  python record_zed_svo.py output.svo [--duration 60] [--resolution HD720] [--fps 30]
  # Press Ctrl+C to stop early.

Requires: ZED SDK installed, then: pip install pyzed
"""

from __future__ import annotations

import argparse
import signal
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description="Record ZED stereo to SVO")
    parser.add_argument("output", help="Output .svo file path")
    parser.add_argument("--duration", type=float, default=None,
                        help="Record for N seconds (default: until Ctrl+C)")
    parser.add_argument("--resolution", default="HD720",
                        choices=["HD2K", "HD1080", "HD720", "VGA"],
                        help="Camera resolution")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--compression", default="H264",
                        choices=["LOSSLESS", "H264", "H265"],
                        help="SVO compression mode")
    args = parser.parse_args()

    if not args.output.lower().endswith(".svo"):
        args.output = args.output.rstrip("/") + ".svo"

    try:
        import pyzed.sl as sl
    except ImportError:
        print("ZED backend requires pyzed. Install: pip install pyzed", file=sys.stderr)
        return 1

    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = getattr(sl.RESOLUTION, args.resolution, sl.RESOLUTION.HD720)
    init_params.camera_fps = args.fps

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open ZED: {err}", file=sys.stderr)
        return 1

    rec_params = sl.RecordingParameters()
    rec_params.video_filename = args.output
    rec_params.compression_mode = getattr(
        sl.SVO_COMPRESSION_MODE, args.compression, sl.SVO_COMPRESSION_MODE.H264
    )

    err = zed.enable_recording(rec_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to enable recording: {err}", file=sys.stderr)
        zed.close()
        return 1

    print(f"Recording to {args.output} (resolution={args.resolution}, fps={args.fps})")
    print("Press Ctrl+C to stop.")
    if args.duration is not None:
        print(f"Will stop after {args.duration} seconds.")

    exit_app = [False]  # mutable so signal handler can set it

    def on_sigint(_sig, _frame):  # noqa: ARG001
        exit_app[0] = True

    signal.signal(signal.SIGINT, on_sigint)

    runtime = sl.RuntimeParameters()
    start = time.monotonic()
    frame_count = 0

    while not exit_app[0]:
        if args.duration is not None and (time.monotonic() - start) >= args.duration:
            break
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"  {frame_count} frames ...")
        else:
            time.sleep(0.001)

    zed.disable_recording()
    zed.close()
    print(f"Saved {frame_count} frames to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
