"""
Incremental global semantic map.
Fuses labeled point clouds as the robot moves; supports pose updates for integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .point_cloud import PointCloudFrame, transform_point_cloud


@dataclass
class SemanticMap:
    """
    Global 3D semantic map: voxel grid storing dominant label and optionally color.
    Incrementally updated with new labeled point clouds and optional pose.
    """

    def __init__(
        self,
        voxel_size: float = 0.05,
        max_points_per_voxel: int = 1,
        world_range: Optional[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = None,
        class_names: Optional[list[str]] = None,
    ):
        self.voxel_size = voxel_size
        self.max_points_per_voxel = max_points_per_voxel
        # world_range: ((xmin, xmax), (ymin, ymax), (zmin, zmax)) or None for unbounded
        self.world_range = world_range
        self.class_names = class_names or []
        # Voxel key -> (label, count, [rgb sum]) for fusion
        self._voxels: dict[tuple[int, int, int], tuple[int, int, np.ndarray]] = {}
        self._n_points = 0

    def _key(self, x: float, y: float, z: float) -> tuple[int, int, int]:
        v = self.voxel_size
        return (int(np.floor(x / v)), int(np.floor(y / v)), int(np.floor(z / v)))

    def _in_range(self, x: float, y: float, z: float) -> bool:
        if self.world_range is None:
            return True
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = self.world_range
        return xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax

    def update(
        self,
        pc: PointCloudFrame,
        pose_R: Optional[np.ndarray] = None,
        pose_t: Optional[np.ndarray] = None,
        max_points: Optional[int] = 100_000,
    ) -> None:
        """
        Integrate a labeled point cloud into the map.
        pose_R (3,3), pose_t (3,): transform from camera to world. If None, identity.
        """
        xyz = pc.xyz
        labels = pc.labels if pc.labels is not None else np.full(xyz.shape[0], -1, dtype=np.int32)
        rgb = pc.rgb

        if pose_R is not None and pose_t is not None:
            xyz = transform_point_cloud(xyz, pose_R, pose_t)

        if max_points and xyz.shape[0] > max_points:
            idx = np.random.choice(xyz.shape[0], max_points, replace=False)
            xyz = xyz[idx]
            labels = labels[idx]
            rgb = rgb[idx] if rgb is not None else None

        for i in range(xyz.shape[0]):
            x, y, z = xyz[i]
            if not self._in_range(x, y, z):
                continue
            key = self._key(x, y, z)
            lab = int(labels[i])
            rgb_val = rgb[i] if rgb is not None else np.zeros(3, dtype=np.float64)
            if key not in self._voxels:
                self._voxels[key] = (lab, 1, rgb_val.astype(np.float64))
                self._n_points += 1
            else:
                old_lab, count, sum_rgb = self._voxels[key]
                if old_lab == lab:
                    self._voxels[key] = (lab, count + 1, sum_rgb + rgb_val)
                else:
                    # Majority vote: decrement; if tie broken, set to new label
                    count -= 1
                    if count <= 0:
                        self._voxels[key] = (lab, 1, rgb_val.astype(np.float64))
                    else:
                        self._voxels[key] = (old_lab, count, sum_rgb)

    def get_point_cloud(self) -> PointCloudFrame:
        """Export map as a single point cloud (one point per voxel, centroid)."""
        if not self._voxels:
            return PointCloudFrame(xyz=np.zeros((0, 3)))
        keys = list(self._voxels.keys())
        xyz = np.array(keys, dtype=np.float64) * self.voxel_size + self.voxel_size / 2
        labels = np.array([self._voxels[k][0] for k in keys], dtype=np.int32)
        rgb_list = []
        for k in keys:
            _, _, sum_rgb = self._voxels[k]
            count = self._voxels[k][1]
            rgb_list.append((sum_rgb / max(count, 1)).clip(0, 255).astype(np.uint8))
        rgb = np.array(rgb_list)
        return PointCloudFrame(xyz=xyz, rgb=rgb, labels=labels)

    def get_labels_in_region(
        self,
        xmin: float, xmax: float,
        ymin: float, ymax: float,
        zmin: float, zmax: float,
    ) -> list[tuple[np.ndarray, int]]:
        """Return list of (centroid_xyz, label) for voxels in the given AABB."""
        v = self.voxel_size
        kx_min, kx_max = int(np.floor(xmin / v)), int(np.ceil(xmax / v))
        ky_min, ky_max = int(np.floor(ymin / v)), int(np.ceil(ymax / v))
        kz_min, kz_max = int(np.floor(zmin / v)), int(np.ceil(zmax / v))
        out = []
        for kx in range(kx_min, kx_max + 1):
            for ky in range(ky_min, ky_max + 1):
                for kz in range(kz_min, kz_max + 1):
                    key = (kx, ky, kz)
                    if key not in self._voxels:
                        continue
                    lab = self._voxels[key][0]
                    if lab < 0:
                        continue
                    cx = (kx + 0.5) * v
                    cy = (ky + 0.5) * v
                    cz = (kz + 0.5) * v
                    out.append((np.array([cx, cy, cz]), lab))
        return out

    @property
    def num_voxels(self) -> int:
        return len(self._voxels)
