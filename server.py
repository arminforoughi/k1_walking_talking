#!/usr/bin/env python3
"""
Remote Server — runs on YOUR machine (laptop/desktop with GPU).
Receives camera + depth + audio from the robot over WebSocket.
Runs YOLO, face recognition, Gemini Live API, tracking/follow logic.
Sends control commands back to the robot.

Usage:
    export GEMINI_API_KEY="your-key"
    python3 server.py
    python3 server.py --voice Charon --no-faces --port 8080
"""

import os
import sys
import asyncio
import threading
import time
import argparse
import json
from collections import deque
from http.server import HTTPServer, BaseHTTPRequestHandler

import numpy as np
import cv2
if sys.platform == 'darwin':
    import pyaudio_compat as pyaudio
else:
    try:
        import pyaudio
    except ImportError:
        import pyaudio_compat as pyaudio

try:
    import face_recognition
except (ImportError, OSError):
    face_recognition = None

from frame_processor import FaceCache, FrameProcessor
from robot_controller import RobotController, CommandDispatcher
from google import genai
from google.genai import types

# Audio config
SEND_SAMPLE_RATE = 16000
RECV_SAMPLE_RATE = 24000
AUDIO_CHANNELS = 1
AUDIO_FORMAT = pyaudio.paInt16
AUDIO_CHUNK = 1024

# Binary message type prefixes (robot -> server)
MSG_VIDEO = 0x01
MSG_DEPTH = 0x02
MSG_AUDIO_IN = 0x03
# server -> robot
MSG_AUDIO_OUT = 0x10


# ── Transcript ───────────────────────────────────────────────────────────────

transcript = deque(maxlen=200)
transcript_lock = threading.Lock()
_frame_processor_ref = None
_cmd_dispatcher_ref = None
_session_ref = None
_event_loop_ref = None


def add_transcript(role, text):
    with transcript_lock:
        transcript.append({"role": role, "text": text, "ts": time.time()})
    if role == "You" and _frame_processor_ref:
        _frame_processor_ref.try_learn_name_from_transcript(text)


def get_transcript():
    with transcript_lock:
        return list(transcript)


def send_text_to_gemini(text):
    session, loop = _session_ref, _event_loop_ref
    if not session or not loop:
        print("[Chat] No active Gemini session")
        return False

    async def _send():
        try:
            await session.send_client_content(
                turns=[types.Content(role="user", parts=[types.Part(text=text)])],
                turn_complete=True,
            )
        except Exception as e:
            print(f"[Chat] Send error: {e}")

    asyncio.run_coroutine_threadsafe(_send(), loop)
    return True


# ── Web UI ───────────────────────────────────────────────────────────────────

_TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')


def _load_html():
    with open(os.path.join(_TEMPLATE_DIR, 'index.html')) as f:
        return f.read()


HTML_PAGE = _load_html()


class WebHandler(BaseHTTPRequestHandler):
    frame_processor = None
    robot_controller = None

    def log_message(self, format, *args):
        pass

    def _read_body(self):
        length = int(self.headers.get('Content-Length', 0))
        return self.rfile.read(length) if length else b''

    def _json_response(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode())
        elif self.path.startswith('/frame'):
            fp = self.frame_processor
            if fp and fp.latest_frame is not None:
                with fp._lock:
                    frame = fp.latest_frame.copy()
                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()
                self.wfile.write(buf.tobytes())
            else:
                self.send_error(503, 'No frame')
        elif self.path.startswith('/depth_gradient'):
            fp = self.frame_processor
            if fp:
                view_frame = fp.get_depth_gradient_frame()
                if view_frame is not None:
                    _, buf = cv2.imencode('.jpg', view_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    self.send_response(200)
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Cache-Control', 'no-cache')
                    self.end_headers()
                    self.wfile.write(buf.tobytes())
                else:
                    self.send_error(503, 'No depth view')
            else:
                self.send_error(503, 'No frame processor')
        elif self.path == '/transcript':
            self._json_response(get_transcript())
        elif self.path == '/detections':
            fp = self.frame_processor
            dets = []
            if fp:
                with fp._lock:
                    dets = list(fp.latest_detections)
            self._json_response(dets)
        elif self.path == '/known_faces':
            fp = self.frame_processor
            faces = fp.face_cache.list_known() if (fp and fp.face_cache) else []
            self._json_response(faces)
        elif self.path == '/robot_status':
            robot = self.robot_controller
            connected = robot is not None and robot._ws is not None
            self._json_response({'connected': connected})
        elif self.path == '/scene_data':
            fp = self.frame_processor
            if fp and fp.scene_reconstructor:
                self._json_response(fp.scene_reconstructor.get_scene_data())
            else:
                self._json_response({'robot': {'x': 0, 'y': 0, 'z': 0, 'theta': 0},
                                     'trajectory': [], 'objects': [],
                                     'points': {'positions': [], 'colors': [], 'count': 0}})
        else:
            self.send_error(404)

    def do_POST(self):
        fp = self.frame_processor
        robot = self.robot_controller
        try:
            body = json.loads(self._read_body())
        except Exception:
            self.send_error(400, 'Invalid JSON')
            return

        if self.path == '/chat':
            text = body.get('text', '').strip()
            if not text:
                self._json_response({'error': 'need text'}, 400)
                return
            add_transcript("You", text)
            ok = send_text_to_gemini(text)
            self._json_response({'ok': ok, 'text': text})
        elif self.path == '/cmd':
            action = body.get('action', '')
            result = self._handle_cmd(action, robot, body)
            self._json_response(result)
        elif self.path == '/save_face':
            if not fp:
                self._json_response({'error': 'not ready'}, 503)
                return
            uid = body.get('unknown_id')
            name = body.get('name', '').strip()
            if not name or uid is None:
                self._json_response({'error': 'need unknown_id and name'}, 400)
                return
            ok = fp.save_unknown_face(int(uid), name)
            self._json_response({'ok': ok, 'name': name})
        elif self.path == '/delete_face':
            if not fp or not fp.face_cache:
                self._json_response({'error': 'not ready'}, 503)
                return
            name = body.get('name', '').strip()
            if not name:
                self._json_response({'error': 'need name'}, 400)
                return
            fp.face_cache.delete_face(name)
            self._json_response({'ok': True})
        elif self.path == '/scene_clear':
            if fp and fp.scene_reconstructor:
                fp.scene_reconstructor.clear()
            self._json_response({'ok': True})
        else:
            self.send_error(404)

    def _handle_cmd(self, action, robot, body=None):
        if not robot:
            return {'status': 'error', 'message': 'Robot not connected'}
        add_transcript("Action", action)
        target = body.get('target') if isinstance(body, dict) else None
        distance = body.get('distance') if isinstance(body, dict) else None

        if action == 'follow':
            robot.start_follow(target)
        elif action == 'track':
            robot.start_tracking(target)
        elif action == 'stop':
            robot.stop_all()
        elif action.startswith('go_to'):
            obj = target or action.replace('go_to_', '').replace('go_to', 'person')
            stop_dist = float(distance) if distance else 0.5
            robot.go_to_object(obj or 'person', stop_distance=stop_dist)
            return {'status': 'ok', 'action': 'go_to', 'target': obj, 'distance': stop_dist}
        elif action == 'dance':
            robot.do_dance()
        elif action.startswith('dance_'):
            robot.do_dance(action.replace('dance_', ''))
        elif action == 'wave':
            robot.do_wave()
        elif action == 'handshake':
            robot.do_handshake()
        elif action == 'dab':
            robot.do_dab()
        elif action == 'flex':
            robot.do_flex()
        elif action == 'nod':
            robot.nod()
        elif action == 'head_shake':
            robot.head_shake()
        elif action == 'look_up':
            robot.rotate_head(-0.3, 0.0)
        elif action == 'look_down':
            robot.rotate_head(0.5, 0.0)
        elif action == 'look_left':
            robot.rotate_head(0.0, 0.5)
        elif action == 'look_right':
            robot.rotate_head(0.0, -0.5)
        elif action == 'look_center':
            robot.rotate_head(0.0, 0.0)
        elif action == 'forward':
            robot.forward()
        elif action == 'backward':
            robot.backward()
        elif action == 'strafe_left':
            robot.strafe_left()
        elif action == 'strafe_right':
            robot.strafe_right()
        elif action == 'turn_left':
            robot.turn_left()
        elif action == 'turn_right':
            robot.turn_right()
        elif action == 'turn_around':
            robot.turn_around()
        else:
            return {'status': 'error', 'message': f'Unknown: {action}'}
        return {'status': 'ok', 'action': action}


def start_web_server(frame_processor, robot_controller, host, port):
    WebHandler.frame_processor = frame_processor
    WebHandler.robot_controller = robot_controller
    httpd = HTTPServer((host, port), WebHandler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd


# ── Gemini Session ───────────────────────────────────────────────────────────

SYSTEM_INSTRUCTION = """You are a Booster K1 humanoid robot with stereo vision cameras, face recognition, \
and full body control. Video frames are streamed to you with real-time object detection overlays — \
each detected object has a bounding box, class label, confidence score, and distance in meters. \
People you recognize are labeled with their name; unknown people are labeled 'Unknown #N'.

You can PHYSICALLY ACT by saying certain trigger phrases in your responses. When you decide to act, \
naturally include one of these phrases — your body will respond automatically:

FOLLOWING / TRACKING:
- "I'll follow you" or "Following you now" — walks toward and follows the person
- "I'll track that" or "Tracking the [object]" — moves head AND body to keep an object centered
- "Stopping now" or "I'll stop" — stops all movement and tracking

GO TO OBJECTS:
- "Going to the [object]" or "Walking to the [object]" — walks toward a detected object
- "Going to 0.5 meters from the [object]" — walks to a specific distance from the object
- "Moving over to the [person name]" — walks toward a specific named person

LOOKING / HEAD CONTROL:
- "Looking left/right/up/down" — head movement
- "Looking forward" or "Looking straight" — centers head

MOVEMENT:
- "Walking forward/backward" — walks briefly
- "Strafing left/right" — sidesteps
- "Turning left/right" — rotates body
- "Turning around" — turns 180 degrees
- "Coming closer" or "Backing up"

DANCES & GESTURES:
- "Let me dance" — does a robot dance
- "Moonwalk!" / "Michael Jackson dance!" / "Kick!" / "Roundhouse!" / "Salsa!" etc.
- "I'll wave" / "Let me shake hands" / "Dabbing!" / "Flexing!"
- "Nodding" / "Shaking my head"

IMPORTANT RULES:
- When someone says "follow me", respond with "I'll follow you!"
- When someone says "go to the chair", respond with "Going to the chair!"
- Keep responses short and conversational.
- Only trigger actions when explicitly asked or socially appropriate.
"""


async def gemini_send_video(session, frame_processor, interval):
    try:
        while True:
            b64 = frame_processor.get_frame_b64jpeg()
            if b64:
                await session.send_realtime_input(
                    video=types.Blob(data=b64, mime_type="image/jpeg")
                )
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        pass


async def gemini_send_audio(session, audio_queue):
    """Forward audio from robot mic (via audio_queue) to Gemini."""
    try:
        while True:
            data = await audio_queue.get()
            await session.send_realtime_input(
                audio=types.Blob(data=data, mime_type=f"audio/pcm;rate={SEND_SAMPLE_RATE}")
            )
    except asyncio.CancelledError:
        pass


async def gemini_send_local_audio(session, pya, mic_device=None, mic_gain=1.0):
    """Capture audio from local mic and send to Gemini."""
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
                samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                samples *= mic_gain
                np.clip(samples, -32768, 32767, out=samples)
                data = samples.astype(np.int16).tobytes()
            await session.send_realtime_input(
                audio=types.Blob(data=data, mime_type=f"audio/pcm;rate={SEND_SAMPLE_RATE}")
            )
    except asyncio.CancelledError:
        pass
    finally:
        stream.stop_stream()
        stream.close()


async def gemini_receive(session, pya, cmd_dispatcher, robot_ws_ref):
    """Receive Gemini responses: play audio locally + send to robot, parse commands."""
    stream = pya.open(
        format=AUDIO_FORMAT, channels=AUDIO_CHANNELS, rate=RECV_SAMPLE_RATE,
        output=True, frames_per_buffer=AUDIO_CHUNK,
    )
    loop = asyncio.get_event_loop()
    try:
        while True:
            async for msg in session.receive():
                if msg.data:
                    await loop.run_in_executor(None, stream.write, msg.data)
                    ws = robot_ws_ref.get('ws')
                    if ws:
                        try:
                            await ws.send(bytes([MSG_AUDIO_OUT]) + msg.data)
                        except Exception:
                            pass

                sc = msg.server_content
                if sc:
                    if sc.input_transcription and sc.input_transcription.text:
                        txt = sc.input_transcription.text
                        print(f"  You: {txt}")
                        add_transcript("You", txt)
                    if sc.output_transcription and sc.output_transcription.text:
                        txt = sc.output_transcription.text
                        print(f"Robot: {txt}")
                        add_transcript("Robot", txt)
                        cmd_dispatcher.check_transcript(txt)
    except asyncio.CancelledError:
        pass
    finally:
        stream.stop_stream()
        stream.close()


# ── WebSocket Server (robot connection) ──────────────────────────────────────


async def handle_robot_ws(websocket, frame_processor: FrameProcessor,
                          robot: RobotController, audio_queue: asyncio.Queue,
                          robot_ws_ref: dict):
    """Handle a single robot WebSocket connection."""
    print(f"[WS] Robot connected from {websocket.remote_address}")
    robot.set_connection(websocket, asyncio.get_event_loop())
    robot_ws_ref['ws'] = websocket

    try:
        async for message in websocket:
            if isinstance(message, bytes) and len(message) > 1:
                msg_type = message[0]
                payload = message[1:]
                if msg_type == MSG_VIDEO:
                    frame_processor.on_video_frame(payload)
                elif msg_type == MSG_DEPTH:
                    frame_processor.on_depth_frame(payload)
                elif msg_type == MSG_AUDIO_IN:
                    await audio_queue.put(payload)
    except Exception as e:
        print(f"[WS] Robot disconnected: {e}")
    finally:
        robot.set_connection(None, None)
        robot_ws_ref['ws'] = None
        print("[WS] Robot connection closed")


# ── Main ─────────────────────────────────────────────────────────────────────


async def run_server(args):
    global _frame_processor_ref, _cmd_dispatcher_ref, _session_ref, _event_loop_ref

    api_key = args.api_key or os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        print("Error: provide --api-key or set GEMINI_API_KEY env variable")
        sys.exit(1)

    os.environ.pop('GOOGLE_API_KEY', None)
    os.environ.pop('GEMINI_API_KEY', None)

    face_cache = None
    if not args.no_faces and face_recognition is not None:
        face_cache = FaceCache(tolerance=args.face_tolerance)

    enable_faces = (not args.no_faces) and (face_recognition is not None)
    if not enable_faces and face_recognition is None:
        print("Note: face_recognition/dlib not available; running without face recognition.")
    frame_processor = FrameProcessor(
        model_path=args.model, confidence=args.confidence,
        face_cache=face_cache, enable_faces=enable_faces,
        on_transcript=add_transcript,
    )
    _frame_processor_ref = frame_processor

    robot = RobotController()
    robot.set_frame_processor(frame_processor)
    robot.follow_target_distance = args.follow_distance

    cmd_dispatcher = CommandDispatcher(robot, on_transcript=add_transcript)
    _cmd_dispatcher_ref = cmd_dispatcher

    httpd = start_web_server(frame_processor, robot, '0.0.0.0', args.port)
    print(f"Web UI: http://0.0.0.0:{args.port}")

    audio_queue = asyncio.Queue(maxsize=100)
    robot_ws_ref = {'ws': None}

    import websockets
    print(f"websockets version: {websockets.__version__}")

    async def _ws_handler(websocket, path=None):
        await handle_robot_ws(websocket, frame_processor, robot, audio_queue, robot_ws_ref)

    ws_server = await websockets.serve(
        _ws_handler,
        '0.0.0.0', args.ws_port,
        max_size=10 * 1024 * 1024,
        ping_interval=20,
        ping_timeout=60,
    )
    print(f"Robot WebSocket: ws://0.0.0.0:{args.ws_port}")
    print(f"  Tell robot_client.py to connect to: ws://<THIS_IP>:{args.ws_port}")

    client = genai.Client(api_key=api_key)
    config = types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=args.voice)
            ),
        ),
        system_instruction=SYSTEM_INSTRUCTION,
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )

    pya = pyaudio.PyAudio()
    _event_loop_ref = asyncio.get_event_loop()

    print("Connecting to Gemini Live...")
    try:
        async with client.aio.live.connect(
            model="gemini-2.5-flash-native-audio-preview-12-2025",
            config=config,
        ) as session:
            _session_ref = session
            print("Connected to Gemini!")
            print("Waiting for robot to connect...")
            print("Press Ctrl+C to stop.\n")

            tasks = [
                asyncio.create_task(gemini_send_video(session, frame_processor, args.frame_interval)),
                asyncio.create_task(gemini_receive(session, pya, cmd_dispatcher, robot_ws_ref)),
            ]

            if args.audio_source == 'local':
                tasks.append(asyncio.create_task(
                    gemini_send_local_audio(session, pya, args.mic_device, args.mic_gain)
                ))
            else:
                tasks.append(asyncio.create_task(
                    gemini_send_audio(session, audio_queue)
                ))

            try:
                await asyncio.gather(*tasks)
            except asyncio.CancelledError:
                pass
            finally:
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            _session_ref = None
    finally:
        pya.terminate()
        ws_server.close()
        robot.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description='Remote Server — YOLO + Face + Gemini + Robot Control'
    )
    parser.add_argument('--api-key', type=str, default=None,
                        help='Gemini API key (or set GEMINI_API_KEY)')
    parser.add_argument('--voice', type=str, default='Puck',
                        choices=['Puck', 'Charon', 'Kore', 'Fenrir', 'Aoede'])
    parser.add_argument('--frame-interval', type=float, default=1.0,
                        help='Seconds between frames sent to Gemini')
    parser.add_argument('--port', type=int, default=8080, help='Web UI port')
    parser.add_argument('--ws-port', type=int, default=9090,
                        help='WebSocket port for robot connection')
    parser.add_argument('--model', type=str, default='yolov8n-seg.pt', help='YOLO model')
    parser.add_argument('--confidence', type=float, default=0.5)
    parser.add_argument('--face-tolerance', type=float, default=0.6)
    parser.add_argument('--no-faces', action='store_true')
    parser.add_argument('--follow-distance', type=float, default=1.0)
    parser.add_argument('--audio-source', choices=['local', 'robot'], default='robot',
                        help="'local' = use this machine's mic; 'robot' = stream from robot mic")
    parser.add_argument('--mic-gain', type=float, default=3.0,
                        help='Mic gain (only for --audio-source local)')
    parser.add_argument('--mic-device', type=int, default=None,
                        help='PyAudio mic device (only for --audio-source local)')
    args = parser.parse_args()

    print("=" * 60)
    print("Remote Robot Server")
    print("  YOLO + Face Recognition + Gemini Live + Robot Control")
    print(f"  Audio: {args.audio_source} mic")
    print("=" * 60)

    try:
        asyncio.run(run_server(args))
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == '__main__':
    main()
