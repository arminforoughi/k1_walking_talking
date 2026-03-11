#!/usr/bin/env python3
"""
High-quality 3D reconstruction (Mac-friendly).

Native (no external repos): DPT/MiDaS + Open3D TSDF — works out of the box on Mac.
  python run_high_quality_recon.py --backend native --images path/to/images --poses poses.txt --output mesh.ply

SimpleRecon / CasMVSNet: require one-time setup (clone + download weights).
  python run_high_quality_recon.py --setup
  python run_high_quality_recon.py --backend simplerecon --images path/to/images --poses poses.txt --output out_dir/

Poses file: 16 floats per image (4x4 world_T_cam, row-major), one line per image.
Or use --colmap path/to/colmap/sparse and --images path/to/images for COLMAP output.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="High-quality 3D reconstruction (Mac)")
    parser.add_argument("--backend", choices=["native", "simplerecon", "casmvsnet"], default="native",
                        help="native = DPT+Open3D (no setup); simplerecon/casmvsnet need --setup first")
    parser.add_argument("--setup", action="store_true", help="Clone SimpleRecon/CasMVSNet and print weight URLs")
    parser.add_argument("--images", type=str, help="Directory of images or list file")
    parser.add_argument("--poses", type=str, help="Poses file: 16 floats per line (4x4 row-major world_T_cam)")
    parser.add_argument("--colmap", type=str, help="COLMAP sparse dir (sparse/cameras.txt, images.txt); use with --images for image folder")
    parser.add_argument("--intrinsics", type=str, default=None, help="Optional: fx,fy,cx,cy (one set for all)")
    parser.add_argument("--output", type=str, default="recon_output", help="Output mesh .ply or output directory")
    parser.add_argument("--voxel-size", type=float, default=0.01, help="TSDF voxel size (native backend)")
    parser.add_argument("--device", type=str, default=None, help="Device: cpu, mps, cuda (default: auto)")
    args = parser.parse_args()

    if args.setup:
        from reconstruction.run_simplerecon import setup_simplerecon
        ok = setup_simplerecon()
        print("CasMVSNet: clone manually from https://github.com/kwea123/CasMVSNet_pl and see their README for DTU eval.")
        return 0 if ok else 1

    if not args.images and not args.colmap:
        parser.error("Provide --images (and --poses or --colmap)")

    # Load images + poses
    from reconstruction.adapters import collect_images_and_poses
    try:
        image_paths, poses, intrinsics = collect_images_and_poses(
            images_dir=args.images or "",
            poses_file=args.poses,
            colmap_dir=args.colmap,
        )
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1
    if args.intrinsics:
        fx, fy, cx, cy = map(float, args.intrinsics.split(","))
        intrinsics = [(fx, fy, cx, cy)] * len(image_paths)

    if not image_paths or not poses:
        print("No images or poses found.", file=sys.stderr)
        return 1

    if args.backend == "native":
        from reconstruction.native_recon import run_native_reconstruction
        out = run_native_reconstruction(
            image_paths,
            poses,
            intrinsics,
            output_mesh=args.output,
            voxel_size=args.voxel_size,
            device=args.device,
        )
        print(f"Mesh saved: {out}")
        return 0

    if args.backend == "simplerecon":
        from reconstruction.run_simplerecon import run_simplerecon
        out = run_simplerecon(
            image_paths,
            poses,
            intrinsics,
            output_dir=args.output,
        )
        print(f"SimpleRecon output: {out}")
        return 0

    if args.backend == "casmvsnet":
        print("CasMVSNet runner not fully wired; use --backend native or simplerecon.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
