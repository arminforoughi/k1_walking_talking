"""Remote robot controller with tracking, following, obstacle avoidance, and command dispatch."""

import asyncio
import json
import math
import re
import time
import threading

import numpy as np

_CMD_PATTERNS = [
    (re.compile(r"\b(?:i'll |i will |let me |okay,? |ok,? )?(?:follow|following)\b(?:\s+(?:you|him|her|them|that person|(\w+)))?", re.IGNORECASE), "follow"),
    (re.compile(r"\b(?:i'll |i will |let me )?stop(?:ping)?\b(?:\s+(?:follow|track|mov))?", re.IGNORECASE), "stop"),
    (re.compile(r"\b(?:i'll |let me |okay,? )?(?:go(?:ing)?|walk(?:ing)?|head(?:ing)?|mov(?:e|ing))\s+(?:to(?:ward)?|over to)\s+(?:(\d+(?:\.\d+)?)\s*(?:m(?:eters?)?|ft|feet)\s+(?:from|away from|near|of)\s+)?(?:the\s+|that\s+)?(\w+)", re.IGNORECASE), "go_to"),
    (re.compile(r"\b(?:i'll |i will |let me |here'?s? |okay,? |ok,? )?(?:do |doing |start )?(?:a |the )?(?:dance|dancing)\b(?:\s+(?:the\s+)?(\w+))?", re.IGNORECASE), "dance"),
    (re.compile(r"\b(?:i'll |let me |okay,? )?(?:wave|waving)\b", re.IGNORECASE), "wave"),
    (re.compile(r"\b(?:i'll |let me |okay,? )?(?:handshake|shake hands?|shaking hands?)\b", re.IGNORECASE), "handshake"),
    (re.compile(r"\b(?:i'll |let me |okay,? )?look(?:ing)?\s+(?:to\s+(?:the\s+)?)?left\b", re.IGNORECASE), "look_left"),
    (re.compile(r"\b(?:i'll |let me |okay,? )?look(?:ing)?\s+(?:to\s+(?:the\s+)?)?right\b", re.IGNORECASE), "look_right"),
    (re.compile(r"\b(?:i'll |let me |okay,? )?look(?:ing)?\s+up\b", re.IGNORECASE), "look_up"),
    (re.compile(r"\b(?:i'll |let me |okay,? )?look(?:ing)?\s+down\b", re.IGNORECASE), "look_down"),
    (re.compile(r"\b(?:i'll |let me |okay,? )?look(?:ing)?\s+(?:center|straight|forward|ahead)\b", re.IGNORECASE), "look_center"),
    (re.compile(r"\b(?:i'll |let me |okay,? )?(?:look(?:ing)? at|track(?:ing)?|watch(?:ing)?)\b(?:\s+(?:the\s+)?(\w+))?", re.IGNORECASE), "track"),
    (re.compile(r"\b(?:i'll |let me )?turn(?:ing)?\s+(?:to\s+(?:the\s+)?)?left\b", re.IGNORECASE), "turn_left"),
    (re.compile(r"\b(?:i'll |let me )?turn(?:ing)?\s+(?:to\s+(?:the\s+)?)?right\b", re.IGNORECASE), "turn_right"),
    (re.compile(r"\b(?:i'll |let me )?turn(?:ing)?\s+around\b", re.IGNORECASE), "turn_around"),
    (re.compile(r"\b(?:i'll |let me )?(?:walk(?:ing)?|mov(?:e|ing))\s+forward\b", re.IGNORECASE), "forward"),
    (re.compile(r"\b(?:i'll |let me )?(?:walk(?:ing)?|mov(?:e|ing))\s+backward\b", re.IGNORECASE), "backward"),
    (re.compile(r"\b(?:com(?:e|ing)\s+closer|approach(?:ing)?)\b", re.IGNORECASE), "approach"),
    (re.compile(r"\b(?:back(?:ing)?\s+up|step(?:ping)?\s+back|mov(?:e|ing)\s+back)\b", re.IGNORECASE), "back_up"),
    (re.compile(r"\b(?:i'll |let me )?(?:straf(?:e|ing)|sidestep(?:ping)?|mov(?:e|ing)\s+sideways)\s+(?:to\s+(?:the\s+)?)?left\b", re.IGNORECASE), "strafe_left"),
    (re.compile(r"\b(?:i'll |let me )?(?:straf(?:e|ing)|sidestep(?:ping)?|mov(?:e|ing)\s+sideways)\s+(?:to\s+(?:the\s+)?)?right\b", re.IGNORECASE), "strafe_right"),
    (re.compile(r"\b(?:i'll |let me |okay,? |here'?s? (?:a )?)?dab(?:bing)?\b", re.IGNORECASE), "dab"),
    (re.compile(r"\b(?:i'll |let me |okay,? |here'?s? (?:a )?)?flex(?:ing)?\b", re.IGNORECASE), "flex"),
    (re.compile(r"\b(?:i'll |let me |okay,? )?(?:do )?(?:the )?new\s*year(?:'?s?)?\s*dance\b", re.IGNORECASE), "dance_newyear"),
    (re.compile(r"\b(?:i'll |let me |okay,? )?(?:do )?(?:the )?nezha\s*dance\b", re.IGNORECASE), "dance_nezha"),
    (re.compile(r"\b(?:i'll |let me |okay,? )?(?:do )?(?:the )?future\s*dance\b", re.IGNORECASE), "dance_future"),
    (re.compile(r"\b(?:kick(?:ing)?|boxing\s*kick)\b", re.IGNORECASE), "dance_kick"),
    (re.compile(r"\b(?:moonwalk(?:ing)?)\b", re.IGNORECASE), "dance_moonwalk"),
    (re.compile(r"\b(?:michael\s*jackson)\b", re.IGNORECASE), "dance_michael jackson"),
    (re.compile(r"\b(?:roundhouse(?:\s*kick)?)\b", re.IGNORECASE), "dance_roundhouse"),
    (re.compile(r"\b(?:salsa|arabic\s*dance)\b", re.IGNORECASE), "dance_salsa"),
    (re.compile(r"\b(?:ultraman)\b", re.IGNORECASE), "dance_ultraman"),
    (re.compile(r"\b(?:respect)\b", re.IGNORECASE), "dance_respect"),
    (re.compile(r"\b(?:celebrat(?:e|ing|ion)|cheer(?:ing)?)\b", re.IGNORECASE), "dance_celebrate"),
    (re.compile(r"\b(?:lucky\s*cat)\b", re.IGNORECASE), "dance_luckycat"),
    (re.compile(r"\b(?:macarena)\b", re.IGNORECASE), "dance_macarena"),
    (re.compile(r"\b(?:twist(?:ing)?)\b", re.IGNORECASE), "dance_twist"),
    (re.compile(r"\b(?:take a |do a )?bow(?:ing)?\b", re.IGNORECASE), "dance_bow"),
    (re.compile(r"\b(?:chicken\s*dance|do(?:ing)?\s+(?:the\s+)?chicken)\b", re.IGNORECASE), "dance_chicken"),
    (re.compile(r"\b(?:disco)\b", re.IGNORECASE), "dance_disco"),
    (re.compile(r"\b(?:karate|kung\s*fu)\b", re.IGNORECASE), "dance_karate"),
    (re.compile(r"\b(?:nod(?:ding)?)\b", re.IGNORECASE), "nod"),
    (re.compile(r"\b(?:shak(?:e|ing)\s+(?:my\s+)?head)\b", re.IGNORECASE), "head_shake"),
]


class RobotController:
    """Controls the remote robot by sending commands over WebSocket.
    Runs tracking/follow loops locally (has access to detections)."""

    def __init__(self):
        self._ws = None
        self._loop = None
        self.lock = threading.Lock()
        self.head_pitch = 0.0
        self.head_yaw = 0.0

        self.tracking_active = False
        self.tracking_target = None
        self.tracking_thread = None

        self.follow_active = False
        self.follow_target = None
        self.follow_thread = None
        self.follow_target_distance = 1.0

        self.move_active = False
        self.move_thread = None

        self.frame_processor = None

    def set_connection(self, ws, loop):
        self._ws = ws
        self._loop = loop

    def set_frame_processor(self, fp: "FrameProcessor"):
        self.frame_processor = fp

    def _send(self, cmd_dict):
        ws, loop = self._ws, self._loop
        if ws and loop:
            asyncio.run_coroutine_threadsafe(ws.send(json.dumps(cmd_dict)), loop)

    # ── Head control ─────────────────────────────────────────────────────

    def rotate_head(self, pitch, yaw):
        pitch = max(-0.5, min(1.0, pitch))
        yaw = max(-0.785, min(0.785, yaw))
        self.head_pitch, self.head_yaw = pitch, yaw
        self._send({'cmd': 'rotate_head', 'pitch': pitch, 'yaw': yaw})

    def nod(self):
        self._send({'cmd': 'nod'})

    def head_shake(self):
        self._send({'cmd': 'head_shake'})

    # ── Movement ─────────────────────────────────────────────────────────

    def _move(self, x, y, yaw):
        self._send({'cmd': 'move', 'x': x, 'y': y, 'yaw': yaw})
        if self.frame_processor and self.frame_processor.scene_reconstructor:
            self.frame_processor.scene_reconstructor.update_pose(x, y, yaw, dt=0.05)

    def move_timed(self, x, y, yaw, duration):
        self.stop_movement()
        def _run():
            self.move_active = True
            start = time.time()
            while self.move_active and (time.time() - start) < duration:
                self._move(x, y, yaw)
                time.sleep(0.05)
            self._move(0, 0, 0)
            self.move_active = False
        self.move_thread = threading.Thread(target=_run, daemon=True)
        self.move_thread.start()

    def stop_movement(self):
        self.move_active = False
        if self.move_thread and self.move_thread.is_alive():
            self.move_thread.join(timeout=1.0)
        self._move(0, 0, 0)

    def turn_around(self):
        self.move_timed(0, 0, 0.5, 3.0)

    def approach(self):
        self.move_timed(0.4, 0, 0, 2.0)

    def back_up(self):
        self.move_timed(-0.2, 0, 0, 1.5)

    def turn_left(self):
        self.move_timed(0, 0, 0.5, 1.5)

    def turn_right(self):
        self.move_timed(0, 0, -0.5, 1.5)

    def forward(self):
        self.move_timed(0.5, 0, 0, 2.0)

    def backward(self):
        self.move_timed(-0.3, 0, 0, 2.0)

    def strafe_left(self):
        self.move_timed(0, 0.3, 0, 1.5)

    def strafe_right(self):
        self.move_timed(0, -0.3, 0, 1.5)

    # ── Dances / gestures (delegated to robot client) ────────────────────

    def do_dance(self, dance_name=None):
        self._send({'cmd': 'dance', 'name': dance_name or 'robot'})

    def do_wave(self):
        self._send({'cmd': 'wave'})

    def do_handshake(self):
        self._send({'cmd': 'handshake'})

    def do_dab(self):
        self._send({'cmd': 'dab'})

    def do_flex(self):
        self._send({'cmd': 'flex'})

    # ── Tracking ─────────────────────────────────────────────────────────

    def start_tracking(self, target=None):
        self.stop_tracking()
        self.tracking_active = True
        self.tracking_target = target
        self.tracking_thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.tracking_thread.start()
        print(f"[Robot] Head tracking started: {target or 'closest person'}")

    def stop_tracking(self):
        self.tracking_active = False
        if self.tracking_thread:
            self.tracking_thread.join(timeout=1.0)
            self.tracking_thread = None
        self._move(0, 0, 0)

    def _tracking_loop(self):
        YAW_BODY_TURN_THRESHOLD = 0.45
        BODY_TURN_SPEED = 0.35

        while self.tracking_active:
            if not self.frame_processor or self.frame_processor._raw_frame is None:
                time.sleep(0.1)
                continue

            det = self._find_target_detection()
            if det is None:
                if not self.follow_active:
                    self._move(0, 0, 0)
                time.sleep(0.1)
                continue

            shape = self.frame_processor._frame_shape
            if shape is None:
                time.sleep(0.1)
                continue

            h, w = shape
            cx, cy = det['center']
            err_x = (cx - w / 2) / (w / 2)
            err_y = (cy - h / 2) / (h / 2)

            kp_yaw, kp_pitch = 0.15, 0.1
            new_yaw = self.head_yaw - err_x * kp_yaw
            new_pitch = self.head_pitch + err_y * kp_pitch

            if abs(err_x) > 0.08 or abs(err_y) > 0.08:
                self.rotate_head(new_pitch, new_yaw)

            if not self.follow_active:
                if abs(self.head_yaw) > YAW_BODY_TURN_THRESHOLD:
                    body_rot = BODY_TURN_SPEED if self.head_yaw > 0 else -BODY_TURN_SPEED
                    self._move(0, 0, body_rot)
                else:
                    self._move(0, 0, 0)

            time.sleep(0.1)

    def _find_target_detection(self):
        if not self.frame_processor:
            return None
        with self.frame_processor._lock:
            dets = list(self.frame_processor.latest_detections)
        if not dets:
            return None

        target = self.tracking_target
        if target is None or target.lower() in ('person', 'people', 'someone', 'anyone'):
            persons = [d for d in dets if d['class'] == 'person']
            if not persons:
                return None
            with_dist = [p for p in persons if p.get('distance_m')]
            if with_dist:
                return min(with_dist, key=lambda p: p['distance_m'])
            return max(persons, key=lambda p: (p['bbox'][2] - p['bbox'][0]) * (p['bbox'][3] - p['bbox'][1]))

        named = [d for d in dets if d.get('name') and target.lower() in d['name'].lower()]
        if named:
            return named[0]

        classed = [d for d in dets if d['class'].lower() == target.lower()]
        if classed:
            with_dist = [c for c in classed if c.get('distance_m')]
            if with_dist:
                return min(with_dist, key=lambda c: c['distance_m'])
            return max(classed, key=lambda c: (c['bbox'][2] - c['bbox'][0]) * (c['bbox'][3] - c['bbox'][1]))
        return None

    # ── Follow ───────────────────────────────────────────────────────────

    def start_follow(self, target=None):
        self.stop_follow()
        self.follow_active = True
        self.follow_target = target
        self.start_tracking(target or 'person')
        self.follow_thread = threading.Thread(target=self._follow_loop, daemon=True)
        self.follow_thread.start()
        print(f"[Robot] Following started: {target or 'closest person'}")

    def stop_follow(self):
        self.follow_active = False
        self.stop_tracking()
        if self.follow_thread:
            self.follow_thread.join(timeout=1.0)
            self.follow_thread = None
        self._move(0, 0, 0)

    def _follow_loop(self):
        OBSTACLE_DIST = 0.9
        OBSTACLE_EMERGENCY = 0.5
        STRAFE_SPEED = 0.25
        MAX_FWD = 0.5

        while self.follow_active:
            fp = self.frame_processor
            if not fp or fp._raw_frame is None:
                time.sleep(0.1)
                continue

            det = self._find_target_detection()
            if det is None:
                self._move(0, 0, 0)
                time.sleep(0.2)
                continue

            shape = fp._frame_shape
            if shape is None:
                time.sleep(0.1)
                continue

            h, w = shape
            cx = det['center'][0]
            distance = det.get('distance_m')
            target_id = id(det)

            err_x = (cx - w / 2) / (w / 2)
            rot_speed = -err_x * 0.8 if abs(err_x) > 0.10 else 0.0

            fwd_speed = 0.0
            if distance is not None:
                dist_error = distance - self.follow_target_distance
                if dist_error > 0.3:
                    fwd_speed = min(MAX_FWD, dist_error * 0.4)
                elif dist_error < -0.3:
                    fwd_speed = max(-0.2, dist_error * 0.2)
            else:
                bbox_w = det['bbox'][2] - det['bbox'][0]
                bbox_ratio = bbox_w / w
                if bbox_ratio < 0.15:
                    fwd_speed = 0.3
                elif bbox_ratio > 0.4:
                    fwd_speed = -0.1

            if abs(err_x) > 0.35:
                fwd_speed *= 0.2

            strafe_speed = 0.0
            obstacle_ahead = False

            depth_map = fp._depth_map
            if depth_map is not None and fwd_speed > 0:
                dh, dw = depth_map.shape
                strip_y1, strip_y2 = int(dh * 0.35), int(dh * 0.75)
                strip = depth_map[strip_y1:strip_y2, :]
                n_zones = 5
                zone_w = dw // n_zones
                zone_depths = []
                for i in range(n_zones):
                    col1 = i * zone_w
                    col2 = col1 + zone_w if i < n_zones - 1 else dw
                    zone = strip[:, col1:col2]
                    valid = zone[(zone > 0) & (zone < 65535)].astype(np.float32)
                    d = float(np.median(valid)) / 1000.0 if len(valid) > 20 else 10.0
                    zone_depths.append(d)

                center_d = zone_depths[2]
                near_left_d, near_right_d = zone_depths[1], zone_depths[3]
                far_left_d, far_right_d = zone_depths[0], zone_depths[4]
                target_d = distance if distance else 5.0

                if center_d < OBSTACLE_DIST and center_d < target_d - 0.3:
                    obstacle_ahead = True
                    left_clear = (far_left_d + near_left_d) / 2
                    right_clear = (far_right_d + near_right_d) / 2

                    if center_d < OBSTACLE_EMERGENCY:
                        fwd_speed = min(fwd_speed, 0.05)
                        if left_clear > right_clear:
                            strafe_speed, rot_speed = STRAFE_SPEED, max(rot_speed, 0.3)
                        else:
                            strafe_speed, rot_speed = -STRAFE_SPEED, min(rot_speed, -0.3)
                    else:
                        fwd_speed = min(fwd_speed, 0.15)
                        if left_clear > right_clear:
                            rot_speed, strafe_speed = max(rot_speed, 0.25), 0.15
                        else:
                            rot_speed, strafe_speed = min(rot_speed, -0.25), -0.15

                elif near_left_d < OBSTACLE_EMERGENCY:
                    strafe_speed = -0.15
                elif near_right_d < OBSTACLE_EMERGENCY:
                    strafe_speed = 0.15

            if fwd_speed > 0 and not obstacle_ahead:
                with fp._lock:
                    all_dets = list(fp.latest_detections)
                for obj in all_dets:
                    if id(obj) == target_id:
                        continue
                    obj_dist = obj.get('distance_m')
                    if obj_dist is None or obj_dist > OBSTACLE_DIST:
                        continue
                    obj_cx = obj['center'][0]
                    obj_err = (obj_cx - w / 2) / (w / 2)
                    if abs(obj_err) < 0.4:
                        fwd_speed = min(fwd_speed, 0.1)
                        if obj_err <= 0:
                            strafe_speed = max(strafe_speed, -0.15)
                        else:
                            strafe_speed = min(strafe_speed, 0.15)
                        break

            self._move(fwd_speed, strafe_speed, rot_speed)
            time.sleep(0.05)

    # ── Go to object ─────────────────────────────────────────────────────

    def go_to_object(self, target, stop_distance=0.5):
        self.stop_follow()
        self.follow_active = True
        self.follow_target = target
        self.start_tracking(target)
        self.follow_thread = threading.Thread(
            target=self._go_to_loop, args=(target, stop_distance), daemon=True
        )
        self.follow_thread.start()
        print(f"[Robot] Going to: {target} (stop {stop_distance:.1f}m away)")

    def _go_to_loop(self, target, stop_distance=0.5):
        TIMEOUT = 30.0
        OBSTACLE_DIST = 0.8
        MIN_FWD, MAX_FWD = 0.15, 0.5
        STEER_SPEED = 0.35
        LOST_PATIENCE = 3.0

        start = time.time()
        lost_since = None

        while self.follow_active and (time.time() - start) < TIMEOUT:
            fp = self.frame_processor
            if not fp or fp._raw_frame is None:
                time.sleep(0.1)
                continue

            det = self._find_target_detection()
            if det is None:
                if lost_since is None:
                    lost_since = time.time()
                self._move(0, 0, 0)
                if time.time() - lost_since > LOST_PATIENCE:
                    print(f"[Robot] Lost target {target}, giving up")
                    break
                time.sleep(0.2)
                continue
            lost_since = None

            shape = fp._frame_shape
            if shape is None:
                time.sleep(0.1)
                continue

            h, w = shape
            cx = det['center'][0]
            distance = det.get('distance_m')

            if distance is not None and distance <= stop_distance:
                print(f"[Robot] Arrived at {target} ({distance:.1f}m)")
                break

            bbox_w = det['bbox'][2] - det['bbox'][0]
            bbox_ratio = bbox_w / w
            bbox_threshold = max(0.30, 0.55 - stop_distance * 0.15)
            if distance is None and bbox_ratio >= bbox_threshold:
                print(f"[Robot] Arrived at {target} (bbox {bbox_ratio:.2f})")
                break

            err_x = (cx - w / 2) / (w / 2)
            rot_speed = -err_x * 0.5 if abs(err_x) > 0.10 else 0.0

            if distance is not None:
                remaining = distance - stop_distance
                if remaining <= 0:
                    break
                fwd_speed = min(MAX_FWD, max(MIN_FWD, remaining * 0.35))
            else:
                fwd_speed = 0.3

            depth_map = fp._depth_map
            if depth_map is not None:
                dh, dw = depth_map.shape
                strip_y1, strip_y2 = int(dh * 0.4), int(dh * 0.7)
                strip = depth_map[strip_y1:strip_y2, :]
                third = dw // 3

                def _zone_depth(s):
                    valid = s[(s > 0) & (s < 65535)].astype(np.float32)
                    return float(np.median(valid)) / 1000.0 if len(valid) > 20 else 10.0

                left_d = _zone_depth(strip[:, :third])
                center_d = _zone_depth(strip[:, third:2*third])
                right_d = _zone_depth(strip[:, 2*third:])

                target_dist = distance if distance else 5.0
                if center_d < OBSTACLE_DIST and center_d < target_dist - 0.5:
                    rot_speed = STEER_SPEED if left_d > right_d else -STEER_SPEED
                    fwd_speed = min(fwd_speed, 0.2)

            self._move(fwd_speed, 0.0, rot_speed)
            time.sleep(0.05)

        self._move(0, 0, 0)
        self.follow_active = False
        print(f"[Robot] Go-to complete: {target}")

    # ── Stop all ─────────────────────────────────────────────────────────

    def stop_all(self):
        self.stop_follow()
        self.stop_tracking()
        self.stop_movement()
        self.rotate_head(0.0, 0.0)
        print("[Robot] All stopped")

    def shutdown(self):
        self.stop_all()


class CommandDispatcher:
    def __init__(self, robot: RobotController, on_transcript=None):
        self.robot = robot
        self._on_transcript = on_transcript
        self._last_cmd_time = 0
        self._cmd_cooldown = 2.0

    def check_transcript(self, text):
        now = time.time()
        if now - self._last_cmd_time < self._cmd_cooldown:
            return None
        text_lower = text.lower().strip()
        for pattern, cmd_name in _CMD_PATTERNS:
            match = pattern.search(text_lower)
            if match:
                self._last_cmd_time = now
                self._execute(cmd_name, match)
                return cmd_name
        return None

    def _execute(self, cmd, match):
        if cmd == "go_to":
            groups = match.groups()
            dist_str, go_target = groups[0], groups[1]
            stop_dist = float(dist_str) if dist_str else 0.5
            target = go_target or "person"
            print(f"[CMD] go_to {target} ({stop_dist}m)")
            if self._on_transcript:
                self._on_transcript("Action", f"go_to {target} ({stop_dist}m)")
            self.robot.go_to_object(target, stop_distance=stop_dist)
            return

        target = None
        for g in match.groups():
            if g:
                target = g.strip()
                break

        print(f"[CMD] {cmd}" + (f" ({target})" if target else ""))
        if self._on_transcript:
            self._on_transcript("Action", f"{cmd}" + (f" ({target})" if target else ""))

        actions = {
            "follow": lambda: self.robot.start_follow(target),
            "stop": self.robot.stop_all,
            "dance": lambda: self.robot.do_dance(target),
            "wave": self.robot.do_wave,
            "handshake": self.robot.do_handshake,
            "dab": self.robot.do_dab,
            "flex": self.robot.do_flex,
            "track": lambda: self.robot.start_tracking(target),
            "look_left": lambda: self.robot.rotate_head(0.0, 0.5),
            "look_right": lambda: self.robot.rotate_head(0.0, -0.5),
            "look_up": lambda: self.robot.rotate_head(-0.3, 0.0),
            "look_down": lambda: self.robot.rotate_head(0.5, 0.0),
            "look_center": lambda: self.robot.rotate_head(0.0, 0.0),
            "turn_left": self.robot.turn_left,
            "turn_right": self.robot.turn_right,
            "turn_around": self.robot.turn_around,
            "forward": self.robot.forward,
            "backward": self.robot.backward,
            "strafe_left": self.robot.strafe_left,
            "strafe_right": self.robot.strafe_right,
            "approach": self.robot.approach,
            "back_up": self.robot.back_up,
            "nod": self.robot.nod,
            "head_shake": self.robot.head_shake,
        }

        if cmd in actions:
            actions[cmd]()
        elif cmd.startswith("dance_"):
            self.robot.do_dance(cmd.replace("dance_", ""))
