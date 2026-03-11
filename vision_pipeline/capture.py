"""
Stereo and depth capture for robot vision pipeline.
Supports ZED cameras (pyzed) and ROS/external depth (e.g. StereoNet on K1).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class FrameData:
    """Single captured frame: left image, optional right, depth, and intrinsics."""

    rgb_left: np.ndarray  # (H, W, 3) BGR
    depth: np.ndarray     # (H, W) float32 meters, or (H_d, W_d) if different resolution
    rgb_right: Optional[np.ndarray] = None
    timestamp: float = 0.0
    # Camera intrinsics (for left camera); used for point cloud back-projection
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    # Depth image size might differ from RGB
    depth_scale: float = 1.0   # raw_depth * depth_scale = meters
    depth_min: float = 0.1
    depth_max: float = 10.0

    @property
    def height(self) -> int:
        return self.rgb_left.shape[0]

    @property
    def width(self) -> int:
        return self.rgb_left.shape[1]


class CaptureBackend(ABC):
    """Abstract base for stereo/depth capture."""

    @abstractmethod
    def open(self) -> bool:
        """Open device/stream. Returns True on success."""
        pass

    @abstractmethod
    def grab(self) -> Optional[FrameData]:
        """Grab next frame. Returns None if not available."""
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @property
    @abstractmethod
    def is_opened(self) -> bool:
        pass


# ─── ZED (pyzed) backend ─────────────────────────────────────────────────────

class ZEDCapture(CaptureBackend):
    """Capture from Stereolabs ZED / ZED2 camera via pyzed."""

    def __init__(
        self,
        resolution: str = "HD720",
        fps: int = 30,
        depth_mode: str = "NEURAL",
        coordinate_units: str = "METER",
    ):
        self._resolution = resolution
        self._fps = fps
        self._depth_mode = depth_mode
        self._coordinate_units = coordinate_units
        self._cam = None
        self._opened = False

    def open(self) -> bool:
        try:
            import pyzed.sl as sl
        except ImportError:
            raise RuntimeError(
                "ZED backend requires pyzed. Install: pip install pyzed"
            )
        self._sl = sl
        self._cam = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = getattr(
            sl.RESOLUTION, self._resolution, sl.RESOLUTION.HD720
        )
        init_params.camera_fps = self._fps
        init_params.depth_mode = getattr(
            sl.DEPTH_MODE, self._depth_mode, sl.DEPTH_MODE.NEURAL
        )
        init_params.coordinate_units = getattr(
            sl.UNIT, self._coordinate_units, sl.UNIT.METER
        )
        init_params.depth_minimum_distance = 0.3
        init_params.depth_maximum_distance = 15.0
        err = self._cam.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            return False
        self._opened = True
        return True

    def grab(self) -> Optional[FrameData]:
        if not self._opened or self._cam is None:
            return None
        sl = self._sl
        if self._cam.grab() != sl.ERROR_CODE.SUCCESS:
            return None
        # Left image
        left = sl.Mat()
        self._cam.retrieve_image(left, sl.VIEW.LEFT)
        rgb_left = left.get_data()[:, :, :3]  # RGBA -> RGB

        # Depth (in meters if coordinate_units=METER)
        depth_mat = sl.Mat()
        self._cam.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
        depth = depth_mat.get_data().squeeze().astype(np.float32)
        # Invalid depth is often NaN or very large
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

        # Intrinsics
        calib = self._cam.get_camera_information().camera_configuration.calibration_parameters
        left_calib = calib.left_cam
        fx = left_calib.fx
        fy = left_calib.fy
        cx = left_calib.cx
        cy = left_calib.cy

        return FrameData(
            rgb_left=rgb_left,
            depth=depth,
            timestamp=time.time(),
            fx=fx, fy=fy, cx=cx, cy=cy,
            depth_scale=1.0,
            depth_min=0.3,
            depth_max=15.0,
        )

    def close(self) -> None:
        if self._cam is not None:
            self._cam.close()
            self._cam = None
        self._opened = False

    @property
    def is_opened(self) -> bool:
        return self._opened


# ─── ROS / external (callback) backend ───────────────────────────────────────

class ROSCapture(CaptureBackend):
    """
    Capture from ROS topics (e.g. /image_left_raw, /StereoNetNode/stereonet_depth).
    Requires passing in the latest image and depth arrays + intrinsics.
    Use set_frame() from your ROS subscriber.
    """

    def __init__(
        self,
        fx: float = 320.0,
        fy: float = 320.0,
        cx: float = 320.0,
        cy: float = 240.0,
        depth_scale: float = 0.001,
        depth_height: Optional[int] = None,
        depth_width: Optional[int] = None,
    ):
        self._fx, self._fy = fx, fy
        self._cx, self._cy = cx, cy
        self._depth_scale = depth_scale  # e.g. 0.001 if depth is in mm
        self._depth_height = depth_height
        self._depth_width = depth_width
        self._rgb: Optional[np.ndarray] = None
        self._depth: Optional[np.ndarray] = None
        self._lock = __import__("threading").Lock()
        self._opened = True

    def set_frame(self, rgb: np.ndarray, depth: np.ndarray) -> None:
        """Update latest frame from ROS callbacks."""
        with self._lock:
            self._rgb = rgb.copy() if rgb is not None else self._rgb
            self._depth = depth.copy() if depth is not None else self._depth

    def open(self) -> bool:
        return True

    def grab(self) -> Optional[FrameData]:
        with self._lock:
            rgb = self._rgb
            depth = self._depth
        if rgb is None or depth is None:
            return None
        # Depth might be uint16 (mm); convert to float meters
        if depth.dtype == np.uint16:
            depth_f = depth.astype(np.float32) * self._depth_scale
        else:
            depth_f = depth.astype(np.float32)
        # Optionally resize depth to match RGB for simpler pipeline
        if rgb.shape[:2] != depth_f.shape[:2]:
            import cv2
            depth_f = cv2.resize(
                depth_f, (rgb.shape[1], rgb.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
        return FrameData(
            rgb_left=rgb,
            depth=depth_f,
            timestamp=time.time(),
            fx=self._fx, fy=self._fy, cx=self._cx, cy=self._cy,
            depth_scale=1.0,
            depth_min=0.1,
            depth_max=10.0,
        )

    def close(self) -> None:
        self._opened = False

    @property
    def is_opened(self) -> bool:
        return self._opened


# ─── OAK-D (Luxonis DepthAI) backend ────────────────────────────────────────

class OAKCapture(CaptureBackend):
    """
    Capture from OAK-D / OAK-D Lite / OAK-D Pro via DepthAI SDK.
    Uses pipeline.start() + getDefaultDevice() (same pattern as testing/project) so
    the device is opened when starting the pipeline, not upfront.
    Install: pip install depthai>=2.24 opencv-python
    """

    def __init__(
        self,
        rgb_size: tuple[int, int] = (1280, 720),
        fps: int = 30,
        depth_align_to_rgb: bool = True,
    ):
        self._rgb_size = rgb_size
        self._fps = fps
        self._depth_align_to_rgb = depth_align_to_rgb
        self._device = None
        self._pipeline = None
        self._opened = False
        self._rgb_queue = None
        self._depth_queue = None
        self._calib = None
        self._fx = self._fy = self._cx = self._cy = 0.0
        self._baseline = 0.075  # fallback 7.5 cm

    def open(self) -> bool:
        try:
            import depthai as dai
        except ImportError:
            raise RuntimeError(
                "OAK-D backend requires depthai. Install: pip install depthai"
            )
        self._dai = dai

        def _handle_oak_error(e: Exception, context: str = "open device") -> None:
            msg = str(e)
            if "X_LINK_INSUFFICIENT_PERMISSIONS" in msg or "Insufficient permissions" in msg:
                raise RuntimeError(
                    "OAK-D: USB permission denied. On macOS: (1) Quit OAK Viewer and any other app using the camera, "
                    "unplug and replug the OAK, then try again. (2) Grant Terminal/Cursor access in System Settings > "
                    "Privacy & Security. (3) Run with sudo: ./run_oak.sh  or  sudo python run_vision_pipeline.py --backend oak"
                ) from e
            if "closed or disconnected" in msg or "Input/output error" in msg or "crashed" in msg.lower():
                raise RuntimeError(
                    "OAK-D: device disconnected or crashed. Unplug the camera, wait ~5 s, plug it back in (use a USB3 port and cable), "
                    "then try again. If it keeps happening, try a powered USB hub or different cable."
                ) from e
            raise RuntimeError(f"OAK-D: failed to {context}: {e}") from e

        w, h = self._rgb_size
        # Build pipeline first (no device opened yet); then pipeline.start() + getDefaultDevice()
        pipeline = dai.Pipeline()
        rgb_socket = getattr(dai.CameraBoardSocket, "CAM_A", dai.CameraBoardSocket.RGB)

        # RGB (CAM_A = color on OAK-D) — use preview at desired size (DepthAI v3 style)
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setBoardSocket(rgb_socket)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setPreviewSize(w, h)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setFps(self._fps)

        # Mono left/right for stereo
        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_right = pipeline.create(dai.node.MonoCamera)
        mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        mono_left.setFps(self._fps)
        mono_right.setFps(self._fps)

        stereo = pipeline.create(dai.node.StereoDepth)
        preset = getattr(dai.node.StereoDepth.PresetMode, "HIGH_ACCURACY", dai.node.StereoDepth.PresetMode.FAST_ACCURACY)
        stereo.setDefaultProfilePreset(preset)
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(True)
        if self._depth_align_to_rgb:
            stereo.setDepthAlign(rgb_socket)

        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)

        # DepthAI v3: create queues from node outputs (no XLinkOut); device opens at pipeline.start()
        self._rgb_queue = cam_rgb.preview.createOutputQueue()
        self._depth_queue = stereo.depth.createOutputQueue()

        try:
            pipeline.start()
        except RuntimeError as e:
            _handle_oak_error(e, "start pipeline")
        except Exception as e:
            raise RuntimeError(f"OAK-D: failed to start pipeline: {e}") from e

        self._pipeline = pipeline
        self._device = pipeline.getDefaultDevice()
        self._calib = self._device.readCalibration()

        # Intrinsics (RGB camera at output resolution)
        try:
            M = self._calib.getCameraIntrinsics(
                rgb_socket,
                w,
                h,
            )
            self._fx = float(M[0][0])
            self._fy = float(M[1][1])
            self._cx = float(M[0][2])
            self._cy = float(M[1][2])
        except Exception:
            self._fx = self._fy = max(w, h) * 1.2
            self._cx = w / 2.0
            self._cy = h / 2.0

        self._opened = True
        return True

    def grab(self) -> Optional[FrameData]:
        if not self._opened or self._device is None:
            return None
        rgb_frame = self._rgb_queue.tryGet()
        depth_frame = self._depth_queue.tryGet()
        if rgb_frame is None or depth_frame is None:
            return None
        rgb = rgb_frame.getCvFrame()
        depth = depth_frame.getFrame()
        if depth is None:
            return None
        # depth from OAK is in mm (uint16) when using stereo.depth
        depth_m = depth.astype(np.float32) / 1000.0
        return FrameData(
            rgb_left=rgb,
            depth=depth_m,
            timestamp=time.time(),
            fx=self._fx,
            fy=self._fy,
            cx=self._cx,
            cy=self._cy,
            depth_scale=1.0,
            depth_min=0.1,
            depth_max=10.0,
        )

    def close(self) -> None:
        if self._pipeline is not None:
            try:
                if getattr(self._pipeline, "isRunning", lambda: False)():
                    self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None
        self._device = None
        self._opened = False

    @property
    def is_opened(self) -> bool:
        return self._opened


# ─── OpenCV stereo / test backend ────────────────────────────────────────────

class OpenCVCapture(CaptureBackend):
    """
    OpenCV VideoCapture (single or stereo). Depth can come from:
    - StereoBM/StereoSGBM if two cameras, or
    - Filled with zeros / synthetic for testing.
    """

    def __init__(
        self,
        device: int = 0,
        width: int = 640,
        height: int = 480,
        use_stereo: bool = False,
        fx: float = 320.0,
        fy: float = 320.0,
        cx: float = 320.0,
        cy: float = 240.0,
    ):
        import cv2
        self._cv2 = cv2
        self._device = device
        self._width, self._height = width, height
        self._use_stereo = use_stereo
        self._fx, self._fy = fx, fy
        self._cx, self._cy = cx, cy
        self._cap = None
        self._opened = False

    def open(self) -> bool:
        self._cap = self._cv2.VideoCapture(self._device)
        if not self._cap.isOpened():
            return False
        self._cap.set(self._cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(self._cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._opened = True
        return True

    def grab(self) -> Optional[FrameData]:
        if not self._opened or self._cap is None:
            return None
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None
        # Synthetic depth for testing: inverse distance from center
        h, w = frame.shape[:2]
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - self._cx) ** 2 + (y - self._cy) ** 2)
        depth = np.zeros((h, w), dtype=np.float32)
        depth[dist > 0] = 2.0 + 3.0 * (dist[dist > 0] / np.sqrt(self._cx**2 + self._cy**2))
        depth = np.clip(depth, 0.2, 8.0)
        return FrameData(
            rgb_left=frame,
            depth=depth,
            timestamp=time.time(),
            fx=self._fx, fy=self._fy, cx=self._cx, cy=self._cy,
            depth_scale=1.0,
            depth_min=0.2,
            depth_max=8.0,
        )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._opened = False

    @property
    def is_opened(self) -> bool:
        return self._opened


def create_capture(
    backend: str = "opencv",
    device: int = 0,
    **kwargs,
) -> CaptureBackend:
    """
    Factory: 'zed' | 'ros' | 'opencv' | 'oak'.
    """
    if backend == "zed":
        return ZEDCapture(**kwargs)
    if backend == "ros":
        return ROSCapture(**kwargs)
    if backend == "oak":
        return OAKCapture(**kwargs)
    if backend == "opencv":
        return OpenCVCapture(device=device, **kwargs)
    raise ValueError(f"Unknown capture backend: {backend}")
