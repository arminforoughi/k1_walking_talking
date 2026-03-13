"""Ground-plane object localization from depth + segmentation masks.

Coordinate systems
------------------
**Camera frame (OpenCV convention)**:
    X = right, Y = down, Z = forward (out of lens).
    Origin at the camera optical center.

**Robot body frame**:
    X = forward (direction robot faces), Y = left, Z = up.
    Origin on the floor directly below the camera mount.

**World frame** (used by ObstacleMap / SceneReconstructor):
    Same axes as robot body frame but rotated by ``robot_theta`` and
    translated by ``(robot_x, robot_y)`` — i.e. a fixed odometry frame
    whose origin is where the robot started.

The camera is mounted at height ``cam_height_m`` above the ground and tilted
downward by ``cam_tilt_rad`` (positive = looking down).  The transform from
camera frame to robot body frame is:

    R_cam2body = Ry(-cam_tilt) · P       (P swaps OpenCV→body axes)
    t_cam2body = [0, 0, cam_height_m]

Pipeline
--------
1.  For each detected object, take the SAM2 mask interior (or bbox fallback)
    and compute a *robust depth* via the median of valid depth pixels inside
    the mask.
2.  Pick the *ground-contact pixel* — bottom-center of the YOLO bounding box.
3.  Back-project that pixel at the robust depth into 3D camera-frame coords.
4.  Optionally refine the ground plane via RANSAC on the full point cloud.
5.  Project the 3D point down onto the ground plane and convert to robot body
    frame → ``(x_floor, y_floor)`` in meters relative to the robot.
"""

import math
import numpy as np
import cv2
from typing import Optional, Tuple, List, Dict, Any


# ---------------------------------------------------------------------------
# Default camera parameters (K1 robot stereo rig — override via calibrate())
# ---------------------------------------------------------------------------
DEFAULT_K = np.array([
    [400.0,   0.0, 320.0],
    [  0.0, 400.0, 240.0],
    [  0.0,   0.0,   1.0],
], dtype=np.float64)

DEFAULT_CAM_HEIGHT_M = 0.45
DEFAULT_CAM_TILT_RAD = 0.18   # ~10 degrees down


# ── Ground Plane (RANSAC + fallback) ─────────────────────────────────────────

class GroundPlane:
    """Represents the ground as a plane ``n · p = d`` in camera frame."""

    def __init__(self, normal: np.ndarray, d: float):
        self.normal = normal / np.linalg.norm(normal)
        self.d = d

    def project_point(self, point_cam: np.ndarray) -> np.ndarray:
        """Project a 3D camera-frame point onto this plane."""
        dist = np.dot(self.normal, point_cam) - self.d
        return point_cam - dist * self.normal

    @staticmethod
    def from_camera_pose(cam_height_m: float, cam_tilt_rad: float) -> "GroundPlane":
        """Build a ground plane from known camera mounting geometry.

        The ground in camera frame is a plane whose normal points roughly in
        the -Y_cam direction (up in world = -Y in OpenCV), shifted by the
        camera height.
        """
        ct = math.cos(cam_tilt_rad)
        st = math.sin(cam_tilt_rad)
        normal = np.array([0.0, -ct, -st], dtype=np.float64)
        d = -cam_height_m * ct
        return GroundPlane(normal, d)


def fit_ground_plane_ransac(
    points_cam: np.ndarray,
    *,
    n_iterations: int = 200,
    distance_threshold: float = 0.03,
    min_inlier_ratio: float = 0.15,
    cam_height_m: float = DEFAULT_CAM_HEIGHT_M,
    cam_tilt_rad: float = DEFAULT_CAM_TILT_RAD,
) -> GroundPlane:
    """Fit a ground plane from a 3D point cloud using RANSAC.

    Parameters
    ----------
    points_cam : (N, 3) float — 3D points in camera frame.
    n_iterations : RANSAC iterations.
    distance_threshold : inlier distance to plane (meters).
    min_inlier_ratio : reject RANSAC if too few inliers and fall back.
    cam_height_m / cam_tilt_rad : fallback geometry.

    Returns
    -------
    GroundPlane in camera frame.
    """
    fallback = GroundPlane.from_camera_pose(cam_height_m, cam_tilt_rad)

    if points_cam is None or len(points_cam) < 50:
        return fallback

    N = len(points_cam)
    if N > 8000:
        idx = np.random.choice(N, 8000, replace=False)
        pts = points_cam[idx]
    else:
        pts = points_cam

    best_inliers = 0
    best_normal = None
    best_d = None
    rng = np.random.default_rng(42)

    for _ in range(n_iterations):
        sample = rng.choice(len(pts), 3, replace=False)
        p0, p1, p2 = pts[sample[0]], pts[sample[1]], pts[sample[2]]
        v1 = p1 - p0
        v2 = p2 - p0
        n = np.cross(v1, v2)
        norm = np.linalg.norm(n)
        if norm < 1e-8:
            continue
        n = n / norm
        d = np.dot(n, p0)

        dists = np.abs(pts @ n - d)
        inliers = int(np.sum(dists < distance_threshold))

        if inliers > best_inliers:
            best_inliers = inliers
            best_normal = n
            best_d = d

    if best_normal is None or best_inliers / len(pts) < min_inlier_ratio:
        return fallback

    # Ensure the normal points "upward" in world (roughly -Y_cam for a
    # forward-looking camera).  If it points the wrong way, flip it.
    if best_normal[1] > 0:
        best_normal = -best_normal
        best_d = -best_d

    # Refit on inliers for a tighter estimate
    dists = np.abs(pts @ best_normal - best_d)
    inlier_mask = dists < distance_threshold
    inlier_pts = pts[inlier_mask]
    if len(inlier_pts) >= 3:
        centroid = inlier_pts.mean(axis=0)
        cov = np.cov((inlier_pts - centroid).T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        refined_normal = eigvecs[:, 0]  # smallest eigval = plane normal
        if refined_normal[1] > 0:
            refined_normal = -refined_normal
        refined_d = np.dot(refined_normal, centroid)
        return GroundPlane(refined_normal, refined_d)

    return GroundPlane(best_normal, best_d)


# ── Coordinate Transforms ────────────────────────────────────────────────────

def _rotation_cam_to_body(cam_tilt_rad: float) -> np.ndarray:
    """3×3 rotation: camera frame (OpenCV) → robot body frame.

    Body: X-fwd, Y-left, Z-up.  Camera: X-right, Y-down, Z-fwd.
    First swap axes (P), then undo the tilt around the body-Y axis.
    """
    ct = math.cos(cam_tilt_rad)
    st = math.sin(cam_tilt_rad)
    # Axis swap: cam_Z→body_X, -cam_X→body_Y, -cam_Y→body_Z
    P = np.array([
        [0,  0, 1],
        [-1, 0, 0],
        [0, -1, 0],
    ], dtype=np.float64)
    # Undo tilt (rotation about body Y by -tilt)
    R_tilt = np.array([
        [ ct, 0, st],
        [  0, 1,  0],
        [-st, 0, ct],
    ], dtype=np.float64)
    return R_tilt @ P


def cam_point_to_body(
    point_cam: np.ndarray,
    cam_height_m: float = DEFAULT_CAM_HEIGHT_M,
    cam_tilt_rad: float = DEFAULT_CAM_TILT_RAD,
) -> np.ndarray:
    """Convert a 3D point from camera frame to robot body frame.

    Returns (3,) array [x_fwd, y_left, z_up] in meters, with the
    origin on the ground below the camera mount.
    """
    R = _rotation_cam_to_body(cam_tilt_rad)
    body = R @ point_cam
    body[2] += cam_height_m   # camera is cam_height above ground
    return body


def body_to_world(
    point_body: np.ndarray,
    robot_x: float,
    robot_y: float,
    robot_theta: float,
) -> np.ndarray:
    """Transform body-frame (x_fwd, y_left, z_up) to world frame."""
    c = math.cos(robot_theta)
    s = math.sin(robot_theta)
    bx, by = point_body[0], point_body[1]
    wx = robot_x + bx * c - by * s
    wy = robot_y + bx * s + by * c
    wz = point_body[2]
    return np.array([wx, wy, wz], dtype=np.float64)


# ── Core: pixel + depth → floor position ─────────────────────────────────────

def _backproject_pixel(u: float, v: float, depth_m: float, K: np.ndarray) -> np.ndarray:
    """Back-project a single pixel (u, v) at depth_m into camera-frame 3D."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (u - cx) * depth_m / fx
    y = (v - cy) * depth_m / fy
    z = depth_m
    return np.array([x, y, z], dtype=np.float64)


def _mask_median_depth(
    mask: Optional[np.ndarray],
    depth_map: np.ndarray,
    bbox: List[int],
    frame_shape: Tuple[int, int],
) -> Optional[float]:
    """Robust depth from a SAM2/YOLO mask interior.

    Uses median of valid depth pixels inside the mask.  Falls back to a
    central bbox patch if the mask is empty or None.

    Parameters
    ----------
    mask : (H, W) bool or None — segmentation mask at *frame* resolution.
    depth_map : (Hd, Wd) uint16 depth in mm.
    bbox : [x1, y1, x2, y2] at frame resolution.
    frame_shape : (H, W) of the detection frame.

    Returns
    -------
    Depth in meters, or None if no valid depth found.
    """
    dh, dw = depth_map.shape
    fh, fw = frame_shape
    sx = dw / fw
    sy = dh / fh

    if mask is not None and mask.any():
        if mask.shape != (dh, dw):
            mask_resized = cv2.resize(
                mask.astype(np.uint8), (dw, dh),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        else:
            mask_resized = mask
        depths = depth_map[mask_resized].astype(np.float32)
        valid = depths[(depths > 0) & (depths < 65535)]
        if len(valid) > 5:
            return float(np.median(valid)) / 1000.0

    # Fallback: central 50 % of bbox mapped to depth image
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    bw = (x2 - x1) * 0.25
    bh = (y2 - y1) * 0.25
    dx1 = max(0, int((cx - bw) * sx))
    dy1 = max(0, int((cy - bh) * sy))
    dx2 = min(dw, int((cx + bw) * sx))
    dy2 = min(dh, int((cy + bh) * sy))
    if dx2 <= dx1 or dy2 <= dy1:
        return None
    patch = depth_map[dy1:dy2, dx1:dx2].astype(np.float32)
    valid = patch[(patch > 0) & (patch < 65535)]
    if len(valid) == 0:
        return None
    return float(np.median(valid)) / 1000.0


def object_floor_position(
    bbox: List[int],
    depth_map: np.ndarray,
    frame_shape: Tuple[int, int],
    K: np.ndarray = DEFAULT_K,
    cam_height_m: float = DEFAULT_CAM_HEIGHT_M,
    cam_tilt_rad: float = DEFAULT_CAM_TILT_RAD,
    mask: Optional[np.ndarray] = None,
    ground_plane: Optional[GroundPlane] = None,
) -> Optional[Tuple[float, float]]:
    """Compute an object's (x, y) floor position in robot body frame.

    Parameters
    ----------
    bbox : [x1, y1, x2, y2] in frame-pixel coordinates.
    depth_map : (Hd, Wd) uint16 depth in mm.
    frame_shape : (H, W) of the RGB detection frame.
    K : 3×3 camera intrinsic matrix.
    cam_height_m : camera mount height above ground.
    cam_tilt_rad : camera downward tilt angle (positive = looking down).
    mask : optional (H, W) bool segmentation mask for robust depth.
    ground_plane : optional pre-fitted GroundPlane; if None, uses camera pose.

    Returns
    -------
    (x_fwd, y_left) in metres in robot body frame, or None on failure.
    """
    if depth_map is None:
        return None

    depth_m = _mask_median_depth(mask, depth_map, bbox, frame_shape)
    if depth_m is None or depth_m < 0.05 or depth_m > 15.0:
        return None

    x1, y1, x2, y2 = bbox
    # Ground-contact pixel: bottom-center of bbox
    u_contact = (x1 + x2) / 2.0
    v_contact = float(y2)
    # Clamp to image bounds
    fh, fw = frame_shape
    u_contact = max(0.0, min(u_contact, fw - 1.0))
    v_contact = max(0.0, min(v_contact, fh - 1.0))

    point_cam = _backproject_pixel(u_contact, v_contact, depth_m, K)

    if ground_plane is not None:
        point_cam = ground_plane.project_point(point_cam)

    body = cam_point_to_body(point_cam, cam_height_m, cam_tilt_rad)

    # Return the XY floor projection (discard Z / height)
    return (float(body[0]), float(body[1]))


# ── Batch helper for FrameProcessor ──────────────────────────────────────────

class GroundMapper:
    """Stateful mapper that fits the ground plane once per frame, then
    localizes each detection to the floor with temporal smoothing.

    Typical usage inside ``FrameProcessor._run_detection``::

        gm = GroundMapper(K=K, cam_height_m=0.45, cam_tilt_rad=0.18)
        gm.update(depth_map, frame_shape)
        for det, mask in zip(detections, masks):
            pos = gm.locate(det['bbox'], mask, label=det['class'])
            det['floor_pos_m'] = pos
    """

    def __init__(
        self,
        K: np.ndarray = DEFAULT_K,
        cam_height_m: float = DEFAULT_CAM_HEIGHT_M,
        cam_tilt_rad: float = DEFAULT_CAM_TILT_RAD,
        use_ransac: bool = True,
        ema_alpha: float = 0.4,
    ):
        self.K = K.astype(np.float64)
        self.cam_height_m = cam_height_m
        self.cam_tilt_rad = cam_tilt_rad
        self.use_ransac = use_ransac
        self.ema_alpha = ema_alpha

        self._ground_plane: Optional[GroundPlane] = None
        self._prev_ground_plane: Optional[GroundPlane] = None
        self._depth_map: Optional[np.ndarray] = None
        self._frame_shape: Optional[Tuple[int, int]] = None

        # Temporal smoothing: keyed by (label, grid_bucket) → (x, y, age)
        self._track_ema: Dict[str, Tuple[float, float, int]] = {}
        self._track_max_age = 15  # frames without update before eviction

        # Debug stats (readable from outside)
        self.debug_ransac_inliers = 0
        self.debug_ransac_used = False
        self.debug_plane_normal = np.zeros(3)
        self.debug_plane_d = 0.0

    def update(self, depth_map: Optional[np.ndarray], frame_shape: Tuple[int, int]):
        """Call once per frame with the current depth map.

        Fits the ground plane via RANSAC (if enabled and enough points exist)
        or falls back to the camera-pose prior.  The plane is temporally
        smoothed with the previous frame to reduce jitter.
        """
        self._depth_map = depth_map
        self._frame_shape = frame_shape

        if depth_map is None:
            self._ground_plane = GroundPlane.from_camera_pose(
                self.cam_height_m, self.cam_tilt_rad
            )
            self.debug_ransac_used = False
            return

        if self.use_ransac:
            points_cam = self._sparse_unproject(depth_map, frame_shape)
            new_plane = fit_ground_plane_ransac(
                points_cam,
                cam_height_m=self.cam_height_m,
                cam_tilt_rad=self.cam_tilt_rad,
            )
            # Smooth the plane normal + d with previous frame
            if self._prev_ground_plane is not None:
                a = 0.5
                blended_n = self._prev_ground_plane.normal * (1 - a) + new_plane.normal * a
                blended_d = self._prev_ground_plane.d * (1 - a) + new_plane.d * a
                new_plane = GroundPlane(blended_n, blended_d)
            self._ground_plane = new_plane
            self._prev_ground_plane = new_plane
            self.debug_ransac_used = True
        else:
            self._ground_plane = GroundPlane.from_camera_pose(
                self.cam_height_m, self.cam_tilt_rad
            )
            self.debug_ransac_used = False

        self.debug_plane_normal = self._ground_plane.normal.copy()
        self.debug_plane_d = self._ground_plane.d

        # Age out stale tracks
        stale = [k for k, (_, _, age) in self._track_ema.items() if age > self._track_max_age]
        for k in stale:
            del self._track_ema[k]

    def locate(
        self,
        bbox: List[int],
        mask: Optional[np.ndarray] = None,
        label: Optional[str] = None,
    ) -> Optional[Tuple[float, float]]:
        """Return ``(x_fwd, y_left)`` floor position in robot body frame.

        If ``label`` is provided, the result is temporally smoothed via EMA
        with previous observations of the same object in a similar position.
        """
        if self._depth_map is None or self._frame_shape is None:
            return None
        raw = object_floor_position(
            bbox=bbox,
            depth_map=self._depth_map,
            frame_shape=self._frame_shape,
            K=self.K,
            cam_height_m=self.cam_height_m,
            cam_tilt_rad=self.cam_tilt_rad,
            mask=mask,
            ground_plane=self._ground_plane,
        )
        if raw is None:
            return None

        # Temporal EMA smoothing per object identity
        if label:
            track_key = self._make_track_key(label, raw)
            prev = self._track_ema.get(track_key)
            if prev is not None:
                px, py, _ = prev
                a = self.ema_alpha
                sx = px * (1 - a) + raw[0] * a
                sy = py * (1 - a) + raw[1] * a
                self._track_ema[track_key] = (sx, sy, 0)
                return (sx, sy)
            else:
                self._track_ema[track_key] = (raw[0], raw[1], 0)

        return raw

    def locate_world(
        self,
        bbox: List[int],
        robot_x: float,
        robot_y: float,
        robot_theta: float,
        mask: Optional[np.ndarray] = None,
        label: Optional[str] = None,
    ) -> Optional[Tuple[float, float]]:
        """Return ``(world_x, world_y)`` floor position in world frame."""
        body = self.locate(bbox, mask, label=label)
        if body is None:
            return None
        world = body_to_world(
            np.array([body[0], body[1], 0.0]),
            robot_x, robot_y, robot_theta,
        )
        return (float(world[0]), float(world[1]))

    def _make_track_key(self, label: str, pos: Tuple[float, float]) -> str:
        """Bucket key: label + coarse spatial grid (0.5 m cells).

        This lets us track the *same chair* across frames without confusing
        it with a different chair on the other side of the room.
        """
        bx = int(round(pos[0] * 2))
        by = int(round(pos[1] * 2))
        return f"{label}:{bx},{by}"

    # ── internal helpers ──────────────────────────────────────────────────

    def _sparse_unproject(
        self,
        depth_map: np.ndarray,
        frame_shape: Tuple[int, int],
        step: int = 8,
    ) -> Optional[np.ndarray]:
        """Sparse back-projection for RANSAC ground plane fitting."""
        dh, dw = depth_map.shape
        fh, fw = frame_shape
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        # Scale intrinsics to depth map resolution if it differs from frame
        scale_x = dw / fw
        scale_y = dh / fh
        fx_d = fx * scale_x
        fy_d = fy * scale_y
        cx_d = cx * scale_x
        cy_d = cy * scale_y

        vs = np.arange(0, dh, step)
        us = np.arange(0, dw, step)
        uu, vv = np.meshgrid(us, vs)
        z_mm = depth_map[vv, uu].astype(np.float32)
        valid = (z_mm > 100) & (z_mm < 10000)
        if np.sum(valid) < 50:
            return None

        z_m = z_mm[valid] / 1000.0
        u_v = uu[valid].astype(np.float32)
        v_v = vv[valid].astype(np.float32)

        x_cam = (u_v - cx_d) * z_m / fx_d
        y_cam = (v_v - cy_d) * z_m / fy_d

        return np.column_stack([x_cam, y_cam, z_m]).astype(np.float64)


# ── Stereo Calibration Helper ────────────────────────────────────────────────

def calibrate_stereo_from_checkerboard(
    left_image: np.ndarray,
    right_image: np.ndarray,
    board_size: Tuple[int, int] = (9, 6),
    square_size_m: float = 0.025,
) -> Optional[Dict[str, Any]]:
    """Calibrate a stereo pair from a single checkerboard image pair.

    Parameters
    ----------
    left_image, right_image : BGR uint8 images.
    board_size : inner corners (columns, rows) of the checkerboard.
    square_size_m : physical size of one square in metres.

    Returns
    -------
    dict with keys:
        K_left, K_right : 3×3 intrinsic matrices
        dist_left, dist_right : distortion coefficients (1×5)
        R, T : rotation and translation between cameras
        baseline_m : scalar baseline in metres
        cam_height_estimate_m : estimated camera height from the board plane

    Returns None if checkerboard detection fails in either image.
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size_m

    gray_l = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY) if left_image.ndim == 3 else left_image
    gray_r = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY) if right_image.ndim == 3 else right_image

    ret_l, corners_l = cv2.findChessboardCorners(gray_l, board_size, None)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, board_size, None)
    if not ret_l or not ret_r:
        return None

    corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
    corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)

    h, w = gray_l.shape[:2]
    obj_points = [objp]
    img_points_l = [corners_l]
    img_points_r = [corners_r]

    flags = (
        cv2.CALIB_FIX_ASPECT_RATIO
        | cv2.CALIB_ZERO_TANGENT_DIST
        | cv2.CALIB_SAME_FOCAL_LENGTH
    )
    ret, K_l, dist_l, K_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
        obj_points, img_points_l, img_points_r,
        None, None, None, None,
        (w, h), criteria=criteria, flags=flags,
    )

    baseline_m = float(np.linalg.norm(T))

    # Estimate camera height: solve PnP on left image to get camera pose
    # relative to the checkerboard, then read off the height component.
    cam_height_m = None
    _, rvec, tvec = cv2.solvePnP(objp, corners_l, K_l, dist_l)
    if tvec is not None:
        R_board, _ = cv2.Rodrigues(rvec)
        cam_pos_in_board = -R_board.T @ tvec
        # Board is on the floor: camera height ≈ Z component of cam position
        cam_height_m = float(abs(cam_pos_in_board[2, 0]))
        if cam_height_m < 0.05 or cam_height_m > 3.0:
            cam_height_m = None

    return {
        "K_left": K_l,
        "K_right": K_r,
        "dist_left": dist_l,
        "dist_right": dist_r,
        "R": R,
        "T": T,
        "baseline_m": baseline_m,
        "cam_height_estimate_m": cam_height_m,
    }
