#!/usr/bin/env python3
"""
Robot Client — runs on the Booster K1 robot.
Streams camera + depth + audio to a remote server via WebSocket.
Receives and executes control commands.

Usage:
    # Default (K1 / StereoNet camera topics)
    python3 robot_client.py eth0 --server ws://YOUR_PC_IP:9090

    # ZED camera: use ZED ROS2 topic names
    python3 robot_client.py eth0 --server ws://YOUR_PC_IP:9090 \\
        --image-topic /zed2i/zed_node/left/image_rect_color \\
        --depth-topic /zed2i/zed_node/depth/depth_registered
"""

import os
import sys
import asyncio
import threading
import time
import argparse
import json
import struct
import zlib

import numpy as np
import cv2
try:
    import pyaudio
except ImportError:
    import pyaudio_compat as pyaudio
import websockets

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge

from booster_robotics_sdk_python import (
    B1LocoClient, ChannelFactory, RobotMode, B1HandIndex, B1HandAction,
    Position, Orientation, Posture,
)

SEND_SAMPLE_RATE = 16000
RECV_SAMPLE_RATE = 24000
AUDIO_CHANNELS = 1
AUDIO_FORMAT = pyaudio.paInt16
AUDIO_CHUNK = 1024

# Binary message type prefixes
MSG_VIDEO = 0x01
MSG_DEPTH = 0x02
MSG_AUDIO_IN = 0x03  # robot mic -> server
MSG_POSE = 0x04      # 16 x float32 row-major world_T_cam (camera in world)


def _quat_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Quaternion (x,y,z,w) -> 3x3 rotation matrix."""
    n = qx * qx + qy * qy + qz * qz + qw * qw
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    x, y, z, w = qx, qy, qz, qw
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ], dtype=np.float64)


# ── ROS2 Camera Streamer ────────────────────────────────────────────────────


class CameraStreamer(Node):
    """ROS2 node: subscribes to camera + depth, buffers latest frames."""

    def __init__(self, image_topic: str = None, depth_topic: str = None, pose_topic: str = None):
        super().__init__('robot_stream_client')
        self.bridge = CvBridge()
        self._lock = threading.Lock()
        self._frame_jpeg = None
        self._depth_compressed = None
        self._pose_payload = None  # 16*4 bytes float32 world_T_cam
        self._new_frame = threading.Event()
        self._new_depth = threading.Event()
        self._new_pose = threading.Event()
        self._raw_frame = None

        # Default: K1 / StereoNet; override with --image-topic / --depth-topic for ZED
        img1 = image_topic or '/image_left_raw'
        img2 = '/booster_video_stream' if not image_topic else None
        dep = depth_topic or '/StereoNetNode/stereonet_depth'

        self.create_subscription(Image, img1, self._on_image, 10)
        if img2:
            self.create_subscription(CompressedImage, img2, self._on_compressed, 10)
        self.create_subscription(Image, dep, self._on_depth, 10)

        if pose_topic:
            # Support common ROS2 pose messages without forcing deps if unused
            try:
                from nav_msgs.msg import Odometry
                self.create_subscription(Odometry, pose_topic, self._on_odom, 10)
                self.get_logger().info(f"Subscribed pose (Odometry): {pose_topic}")
            except Exception:
                try:
                    from geometry_msgs.msg import PoseStamped
                    self.create_subscription(PoseStamped, pose_topic, self._on_pose_stamped, 10)
                    self.get_logger().info(f"Subscribed pose (PoseStamped): {pose_topic}")
                except Exception as e:
                    self.get_logger().error(f"Pose topic subscription failed: {e}")

    def _on_image(self, msg):
        try:
            if msg.encoding == 'nv12':
                h, w = msg.height, msg.width
                yuv = np.frombuffer(msg.data, dtype=np.uint8).reshape((int(h * 1.5), w))
                frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
            else:
                frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self._encode_frame(frame)
        except Exception as e:
            self.get_logger().error(f'Image error: {e}')

    def _on_compressed(self, msg):
        if self._raw_frame is not None:
            return
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                self._encode_frame(frame)
        except Exception as e:
            self.get_logger().error(f'Compressed image error: {e}')

    def _encode_frame(self, frame):
        h, w = frame.shape[:2]
        if max(h, w) > 640:
            s = 640 / max(h, w)
            frame = cv2.resize(frame, (int(w * s), int(h * s)))
        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with self._lock:
            self._raw_frame = frame
            self._frame_jpeg = jpeg.tobytes()
        self._new_frame.set()

    def _on_depth(self, msg):
        try:
            if msg.encoding == 'mono16':
                depth = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width))
            elif msg.encoding == '32FC1':
                # ZED depth is often float32 meters
                depth_f = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))
                depth = (np.clip(depth_f, 0, 65.0) * 1000.0).astype(np.uint16)  # m -> mm
            else:
                depth = self.bridge.imgmsg_to_cv2(msg)
            h, w = depth.shape
            small = cv2.resize(depth, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
            sh, sw = small.shape
            header = struct.pack('<HH', sw, sh)
            compressed = zlib.compress(small.tobytes(), level=1)
            with self._lock:
                self._depth_compressed = header + compressed
            self._new_depth.set()
        except Exception as e:
            self.get_logger().error(f'Depth error: {e}')

    def take_frame(self):
        self._new_frame.clear()
        with self._lock:
            return self._frame_jpeg

    def take_depth(self):
        self._new_depth.clear()
        with self._lock:
            return self._depth_compressed

    def take_pose(self):
        self._new_pose.clear()
        with self._lock:
            return self._pose_payload

    def _pose_to_payload(self, px: float, py: float, pz: float, qx: float, qy: float, qz: float, qw: float) -> bytes:
        R = _quat_to_R(qx, qy, qz, qw)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R.astype(np.float32)
        T[:3, 3] = np.array([px, py, pz], dtype=np.float32)
        return T.tobytes(order='C')

    def _on_odom(self, msg):
        try:
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            payload = self._pose_to_payload(p.x, p.y, p.z, q.x, q.y, q.z, q.w)
            with self._lock:
                self._pose_payload = payload
            self._new_pose.set()
        except Exception as e:
            self.get_logger().error(f'Odom pose error: {e}')

    def _on_pose_stamped(self, msg):
        try:
            p = msg.pose.position
            q = msg.pose.orientation
            payload = self._pose_to_payload(p.x, p.y, p.z, q.x, q.y, q.z, q.w)
            with self._lock:
                self._pose_payload = payload
            self._new_pose.set()
        except Exception as e:
            self.get_logger().error(f'PoseStamped error: {e}')


# ── Robot Command Executor ──────────────────────────────────────────────────


class RobotExecutor:
    """Receives JSON commands from server and executes them on the robot SDK."""

    def __init__(self, client: B1LocoClient):
        self.client = client
        self.lock = threading.Lock()
        self.head_pitch = 0.0
        self.head_yaw = 0.0
        self.right_arm_pos = [0.35, -0.25, 0.1]
        self.left_arm_pos = [0.35, 0.25, 0.1]

    def handle(self, msg):
        cmd = msg.get('cmd')
        if not cmd:
            return
        try:
            handler = getattr(self, f'_cmd_{cmd}', None)
            if handler:
                handler(msg)
            else:
                print(f"[Exec] Unknown command: {cmd}")
        except Exception as e:
            print(f"[Exec] Error in {cmd}: {e}")

    # ── Low-level commands (called at high frequency by server tracking loops)

    def _cmd_move(self, m):
        with self.lock:
            self.client.Move(m.get('x', 0), m.get('y', 0), m.get('yaw', 0))

    def _cmd_rotate_head(self, m):
        p = max(-0.5, min(1.0, m.get('pitch', 0)))
        y = max(-0.785, min(0.785, m.get('yaw', 0)))
        self.head_pitch, self.head_yaw = p, y
        with self.lock:
            self.client.RotateHead(p, y)

    # ── Gesture commands

    def _cmd_wave(self, _):
        def _do():
            with self.lock:
                self.client.WaveHand(B1HandAction.kHandOpen)
        threading.Thread(target=_do, daemon=True).start()

    def _cmd_handshake(self, _):
        def _do():
            with self.lock:
                self.client.Handshake(B1HandAction.kHandOpen)
        threading.Thread(target=_do, daemon=True).start()

    def _cmd_nod(self, _):
        def _do():
            for _ in range(3):
                self._set_head(0.3, self.head_yaw)
                time.sleep(0.25)
                self._set_head(-0.1, self.head_yaw)
                time.sleep(0.25)
            self._set_head(0.0, 0.0)
        threading.Thread(target=_do, daemon=True).start()

    def _cmd_head_shake(self, _):
        def _do():
            for _ in range(3):
                self._set_head(self.head_pitch, 0.4)
                time.sleep(0.2)
                self._set_head(self.head_pitch, -0.4)
                time.sleep(0.2)
            self._set_head(0.0, 0.0)
        threading.Thread(target=_do, daemon=True).start()

    # ── Dance commands

    def _cmd_dance(self, m):
        name = (m.get('name') or 'robot').lower()
        threading.Thread(target=self._run_dance, args=(name,), daemon=True).start()

    def _cmd_dab(self, _):
        threading.Thread(target=self._dab, daemon=True).start()

    def _cmd_flex(self, _):
        threading.Thread(target=self._flex, daemon=True).start()

    # ── Arm commands (for server-driven choreography if needed)

    def _cmd_arm_to_side(self, m):
        self._arm_to_side(m.get('hand', 'right'))

    def _cmd_arm_move_inc(self, m):
        self._arm_inc(m.get('direction', 'up'), m.get('hand', 'right'))

    def _cmd_change_mode(self, m):
        mode_str = m.get('mode', 'walking')
        mode_map = {
            'prepare': RobotMode.kPrepare,
            'walking': RobotMode.kWalking,
        }
        mode = mode_map.get(mode_str, RobotMode.kWalking)
        with self.lock:
            self.client.ChangeMode(mode)

    # ── Internal helpers ────────────────────────────────────────────────────

    def _set_head(self, pitch, yaw):
        pitch = max(-0.5, min(1.0, pitch))
        yaw = max(-0.785, min(0.785, yaw))
        self.head_pitch, self.head_yaw = pitch, yaw
        with self.lock:
            self.client.RotateHead(pitch, yaw)

    def _arm_to_side(self, hand):
        is_left = hand == 'left'
        y_sign = 1 if is_left else -1
        hand_idx = B1HandIndex.kLeftHand if is_left else B1HandIndex.kRightHand
        posture = Posture()
        posture.position = Position(0.35, y_sign * 0.25, 0.1)
        posture.orientation = Orientation(-y_sign * 1.57, -1.57, 0.0)
        with self.lock:
            self.client.MoveHandEndEffectorV2(posture, 800, hand_idx)
        if is_left:
            self.left_arm_pos = [0.35, 0.25, 0.1]
        else:
            self.right_arm_pos = [0.35, -0.25, 0.1]

    def _arm_inc(self, direction, hand):
        STEP = 0.03
        is_left = hand == 'left'
        pos = self.left_arm_pos if is_left else self.right_arm_pos
        hand_idx = B1HandIndex.kLeftHand if is_left else B1HandIndex.kRightHand
        y_sign = 1 if is_left else -1

        if direction == 'up':
            pos[2] = min(pos[2] + STEP, 0.35)
        elif direction == 'down':
            pos[2] = max(pos[2] - STEP, -0.10)
        elif direction == 'forward':
            pos[0] = min(pos[0] + STEP, 0.40)
        elif direction == 'back':
            pos[0] = max(pos[0] - STEP, 0.20)
        elif direction == 'out':
            pos[1] += STEP * y_sign
        elif direction == 'in':
            pos[1] -= STEP * y_sign

        posture = Posture()
        posture.position = Position(pos[0], pos[1], pos[2])
        posture.orientation = Orientation(-y_sign * 1.57, -1.57, 0.0)
        with self.lock:
            self.client.MoveHandEndEffectorV2(posture, 300, hand_idx)

    # ── Dance routines (run locally for timing precision) ───────────────────

    def _run_dance(self, name):
        try:
            from booster_robotics_sdk_python import B1LocoApiId

            sdk_wholebody = {
                'arabic': 0, 'salsa': 0,
                'michael jackson': 1, 'michael': 1, 'mj': 1,
                'michael2': 2, 'moonwalk': 4,
                'kick': 5, 'boxing': 5,
                'roundhouse': 6, 'karate': 6,
            }
            sdk_upper = {
                'newyear': 0, 'new year': 0, 'nezha': 1, 'future': 2,
                'dab': 3, 'ultraman': 4, 'respect': 5,
                'cheer': 6, 'celebrate': 6, 'luckycat': 7, 'lucky cat': 7,
            }

            if name in sdk_wholebody:
                self.client.SendApiRequest(
                    B1LocoApiId(2029),
                    json.dumps({'dance_id': sdk_wholebody[name]})
                )
                return
            if name in sdk_upper:
                self.client.SendApiRequest(
                    B1LocoApiId.kDance,
                    json.dumps({'dance_id': sdk_upper[name]})
                )
                return
        except ImportError:
            pass

        custom = {
            'macarena': self._dance_macarena,
            'twist': self._dance_twist,
            'bow': self._dance_bow,
            'chicken': self._dance_chicken,
            'disco': self._dance_disco,
        }
        if name in custom:
            custom[name]()
        else:
            self._dance_default()

    def _dance_default(self):
        D = 0.2
        for _ in range(2):
            self._set_head(0.0, -0.5)
            self._arm_to_side('right')
            time.sleep(0.5)
            for _ in range(5):
                self._arm_inc('up', 'right'); time.sleep(D)
            self._set_head(0.0, 0.5)
            self._arm_to_side('left')
            time.sleep(0.5)
            for _ in range(5):
                self._arm_inc('up', 'left'); time.sleep(D)
        for _ in range(4):
            self._arm_inc('up', 'right'); self._arm_inc('up', 'left'); time.sleep(D)
        self._set_head(-0.2, 0.0)
        time.sleep(1.5)
        self._set_head(0.0, 0.0)
        self._arm_to_side('right'); self._arm_to_side('left')

    def _dance_macarena(self):
        D = 0.25
        self._arm_to_side('right'); self._arm_to_side('left'); time.sleep(0.5)
        for _ in range(5):
            self._arm_inc('forward', 'right'); self._arm_inc('forward', 'left'); time.sleep(D)
        time.sleep(0.3)
        for _ in range(5):
            self._arm_inc('up', 'right'); self._arm_inc('up', 'left'); time.sleep(D)
        time.sleep(0.3)
        for _ in range(4):
            self._arm_inc('in', 'right'); self._arm_inc('in', 'left')
            self._arm_inc('down', 'right'); self._arm_inc('down', 'left'); time.sleep(D)
        time.sleep(0.3)
        with self.lock:
            self.client.Move(0, 0, 0.5)
        time.sleep(1.0)
        with self.lock:
            self.client.Move(0, 0, 0)
        self._arm_to_side('right'); self._arm_to_side('left')

    def _dance_twist(self):
        D = 0.2
        self._arm_to_side('right'); self._arm_to_side('left'); time.sleep(0.5)
        for _ in range(5):
            self._arm_inc('up', 'right'); self._arm_inc('up', 'left'); time.sleep(D)
        for _ in range(3):
            with self.lock: self.client.Move(0, 0, 0.4)
            self._set_head(0.0, 0.4); time.sleep(0.6)
            with self.lock: self.client.Move(0, 0, -0.4)
            self._set_head(0.0, -0.4); time.sleep(0.6)
        with self.lock: self.client.Move(0, 0, 0)
        self._set_head(0.0, 0.0); time.sleep(0.3)
        self._arm_to_side('right'); self._arm_to_side('left')

    def _dance_bow(self):
        self._set_head(0.8, 0.0); time.sleep(2.0); self._set_head(0.0, 0.0)

    def _dance_chicken(self):
        D = 0.15
        self._arm_to_side('right'); self._arm_to_side('left'); time.sleep(0.5)
        for _ in range(5):
            for _ in range(3):
                self._arm_inc('out', 'right'); self._arm_inc('out', 'left'); time.sleep(D)
            self._set_head(0.3, 0.0)
            for _ in range(3):
                self._arm_inc('in', 'right'); self._arm_inc('in', 'left'); time.sleep(D)
            self._set_head(-0.1, 0.0)
        self._set_head(0.0, 0.0); time.sleep(0.3)
        self._arm_to_side('right'); self._arm_to_side('left')

    def _dance_disco(self):
        D = 0.2
        self._arm_to_side('right'); self._arm_to_side('left'); time.sleep(0.5)
        for _ in range(3):
            for _ in range(6):
                self._arm_inc('up', 'right'); self._arm_inc('out', 'right'); time.sleep(D)
            self._set_head(-0.2, -0.3)
            with self.lock: self.client.Move(0, -0.2, 0)
            time.sleep(0.5)
            self._arm_to_side('right')
            for _ in range(6):
                self._arm_inc('up', 'left'); self._arm_inc('out', 'left'); time.sleep(D)
            self._set_head(-0.2, 0.3)
            with self.lock: self.client.Move(0, 0.2, 0)
            time.sleep(0.5)
            self._arm_to_side('left')
        with self.lock: self.client.Move(0, 0, 0)
        self._set_head(0.0, 0.0)

    def _dab(self):
        D = 0.25
        self._arm_to_side('right'); self._arm_to_side('left'); time.sleep(0.6)
        for _ in range(5): self._arm_inc('back', 'right'); time.sleep(D)
        for _ in range(5): self._arm_inc('in', 'right'); time.sleep(D)
        for _ in range(6): self._arm_inc('up', 'right'); time.sleep(D)
        for _ in range(7): self._arm_inc('up', 'left'); time.sleep(D)
        for _ in range(4): self._arm_inc('out', 'left'); time.sleep(D)
        self._set_head(0.5, 0.5); time.sleep(2.5)
        self._set_head(0.0, 0.0); time.sleep(0.3)
        self._arm_to_side('right'); self._arm_to_side('left')

    def _flex(self):
        D = 0.25
        self._arm_to_side('right'); self._arm_to_side('left'); time.sleep(0.6)
        for _ in range(7):
            self._arm_inc('up', 'right'); self._arm_inc('up', 'left'); time.sleep(D)
        for _ in range(3):
            self._arm_inc('out', 'right'); self._arm_inc('out', 'left'); time.sleep(D)
        self._set_head(-0.3, 0.0); time.sleep(2.0)
        self._set_head(0.0, 0.0); time.sleep(0.3)
        self._arm_to_side('right'); self._arm_to_side('left')


# ── WebSocket Client ────────────────────────────────────────────────────────


def _amplify_audio(data, gain):
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    samples *= gain
    np.clip(samples, -32768, 32767, out=samples)
    return samples.astype(np.int16).tobytes()


async def run_client(args, camera: CameraStreamer, executor: RobotExecutor):
    pya = pyaudio.PyAudio()

    # Auto-detect iFlytek mic
    mic_device = args.mic_device
    if mic_device is None:
        for i in range(pya.get_device_count()):
            info = pya.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0 and 'xfm' in info['name'].lower():
                mic_device = i
                print(f"Auto-detected mic: [{i}] {info['name']}")
                break

    uri = args.server
    print(f"Connecting to server: {uri}")
    print(f"websockets version: {websockets.__version__}")

    while True:
        try:
            async with websockets.connect(
                uri,
                max_size=10 * 1024 * 1024,
                ping_interval=20,
                ping_timeout=60,
                open_timeout=10,
            ) as ws:
                print("Connected to server!")
                tasks = [
                    asyncio.create_task(_stream_video(ws, camera, args.fps)),
                    asyncio.create_task(_stream_depth(ws, camera, args.depth_fps)),
                    asyncio.create_task(_stream_audio(ws, pya, mic_device, args.mic_gain)),
                    asyncio.create_task(_stream_pose(ws, camera, args.pose_fps)) if args.pose_topic else None,
                    asyncio.create_task(_receive_commands(ws, executor, pya)),
                ]
                tasks = [t for t in tasks if t is not None]
                try:
                    await asyncio.gather(*tasks)
                except websockets.ConnectionClosed:
                    print("Connection lost")
                finally:
                    for t in tasks:
                        t.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            print(f"Connection error ({type(e).__name__}: {e}), retrying in 3s...")
        await asyncio.sleep(3)


async def _stream_video(ws, camera: CameraStreamer, fps):
    interval = 1.0 / fps
    try:
        while True:
            jpeg = camera.take_frame()
            if jpeg:
                await ws.send(bytes([MSG_VIDEO]) + jpeg)
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        pass


async def _stream_depth(ws, camera: CameraStreamer, fps):
    interval = 1.0 / fps
    try:
        while True:
            data = camera.take_depth()
            if data:
                await ws.send(bytes([MSG_DEPTH]) + data)
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        pass


async def _stream_audio(ws, pya, mic_device, mic_gain):
    kwargs = dict(
        format=AUDIO_FORMAT, channels=AUDIO_CHANNELS, rate=SEND_SAMPLE_RATE,
        input=True, frames_per_buffer=AUDIO_CHUNK,
    )
    if mic_device is not None:
        kwargs['input_device_index'] = mic_device
    stream = pya.open(**kwargs)
    apply_gain = mic_gain > 1.01
    loop = asyncio.get_event_loop()
    try:
        while True:
            data = await loop.run_in_executor(
                None, lambda: stream.read(AUDIO_CHUNK, exception_on_overflow=False)
            )
            if apply_gain:
                data = _amplify_audio(data, mic_gain)
            await ws.send(bytes([MSG_AUDIO_IN]) + data)
    except asyncio.CancelledError:
        pass
    finally:
        stream.stop_stream()
        stream.close()


async def _stream_pose(ws, camera: CameraStreamer, fps: float):
    interval = 1.0 / max(1e-6, float(fps))
    try:
        while True:
            payload = camera.take_pose()
            if payload and len(payload) == 16 * 4:
                await ws.send(bytes([MSG_POSE]) + payload)
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        pass


async def _receive_commands(ws, executor: RobotExecutor, pya):
    speaker = pya.open(
        format=AUDIO_FORMAT, channels=AUDIO_CHANNELS, rate=RECV_SAMPLE_RATE,
        output=True, frames_per_buffer=AUDIO_CHUNK,
    )
    loop = asyncio.get_event_loop()
    try:
        async for message in ws:
            if isinstance(message, bytes) and len(message) > 0:
                msg_type = message[0]
                payload = message[1:]
                if msg_type == 0x10:  # audio playback from Gemini
                    await loop.run_in_executor(None, speaker.write, payload)
            elif isinstance(message, str):
                try:
                    msg = json.loads(message)
                    executor.handle(msg)
                except json.JSONDecodeError:
                    pass
    except asyncio.CancelledError:
        pass
    finally:
        speaker.stop_stream()
        speaker.close()


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description='K1 Robot Client — streams to remote server')
    parser.add_argument('interface', help='Network interface for robot SDK (e.g. eth0)')
    parser.add_argument('--server', default='ws://localhost:9090',
                        help='WebSocket server URL (e.g. ws://192.168.1.100:9090)')
    parser.add_argument('--fps', type=int, default=10, help='Video stream FPS')
    parser.add_argument('--depth-fps', type=int, default=5, help='Depth stream FPS')
    parser.add_argument('--image-topic', type=str, default=None,
                        help='ROS2 image topic (e.g. /zed2i/zed_node/left/image_rect_color for ZED)')
    parser.add_argument('--depth-topic', type=str, default=None,
                        help='ROS2 depth topic (e.g. /zed2i/zed_node/depth/depth_registered for ZED)')
    parser.add_argument('--pose-topic', type=str, default=None,
                        help='ROS2 pose topic for camera pose (Odometry or PoseStamped). If set, server TSDF uses it.')
    parser.add_argument('--pose-fps', type=float, default=30.0, help='Pose stream FPS (only if --pose-topic is set)')
    parser.add_argument('--mic-gain', type=float, default=3.0, help='Mic gain multiplier')
    parser.add_argument('--mic-device', type=int, default=None, help='PyAudio mic device index')
    args = parser.parse_args()

    print("=" * 60)
    print("K1 Robot Client")
    print(f"  Streaming to: {args.server}")
    print(f"  Video: {args.fps} fps | Depth: {args.depth_fps} fps")
    print("=" * 60)

    # Robot SDK
    print(f"Connecting to robot via {args.interface}...")
    ChannelFactory.Instance().Init(0, args.interface)
    loco_client = B1LocoClient()
    loco_client.Init()
    time.sleep(1.0)

    print("Switching to walking mode...")
    loco_client.ChangeMode(RobotMode.kPrepare)
    time.sleep(2.0)
    loco_client.ChangeMode(RobotMode.kWalking)
    time.sleep(1.0)
    print("Robot ready")

    executor = RobotExecutor(loco_client)

    # ROS2
    rclpy.init()
    camera = CameraStreamer(
        image_topic=args.image_topic,
        depth_topic=args.depth_topic,
        pose_topic=args.pose_topic,
    )
    ros_thread = threading.Thread(target=rclpy.spin, args=(camera,), daemon=True)
    ros_thread.start()

    print("Waiting for camera frame...")
    deadline = time.time() + 10
    while camera._raw_frame is None and time.time() < deadline:
        time.sleep(0.1)
    if camera._raw_frame is not None:
        print("Camera ready!")
    else:
        print("Warning: no frame after 10s — continuing anyway")

    try:
        asyncio.run(run_client(args, camera, executor))
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        with executor.lock:
            executor.client.Move(0, 0, 0)
        camera.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
