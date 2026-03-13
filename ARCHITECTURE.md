# K1 Walking & Talking — Architecture

A voice-controlled robotics system for the **Booster K1 humanoid**.
The robot streams camera, depth, and audio to a GPU server over WebSocket.
The server runs vision (YOLO, SAM2, depth), talks to Gemini Live for
conversational AI, and sends movement commands back to the robot.

---

## High-Level System

```mermaid
graph LR
    subgraph Robot["🤖 Robot (robot_client.py)"]
        ROS2["ROS2\nCameras + Depth\n+ IMU"]
        SDK["K1 Loco SDK\nMove / Head / Dance"]
    end

    subgraph Server["💻 GPU Server (server.py)"]
        FP["FrameProcessor\nYOLO + SAM2 + Depth\n+ Faces + GroundMapper"]
        RC["RobotController\nFollow · Track · GoTo\nObstacle Avoidance"]
        SR["SceneReconstructor\nGaussian Splats\n+ Scene Graph"]
        GM["Gemini Live API\nVoice Conversation\n→ Command Dispatch"]
    end

    WebUI["🌐 Web UI\n3D Scene · Video\nTranscript · Controls"]

    ROS2 -- "WS binary\nvideo + depth + audio" --> FP
    FP -- detections --> RC
    FP -- detections + depth --> SR
    FP -- video frames --> GM
    GM -- "voice commands" --> RC
    RC -- "JSON cmds\nmove / head / dance" --> SDK
    SR -- "splats + objects" --> WebUI
    FP -- "annotated frame" --> WebUI
    GM -- "transcript" --> WebUI
    WebUI -- "POST /cmd" --> RC
```

---

## Vision Pipeline (per frame)

```mermaid
flowchart TD
    A["📷 Left Camera JPEG\n(from robot WebSocket)"] --> B["cv2.imdecode\n→ BGR frame"]
    D["📏 Depth Frame\n(uint16 mm, zlib)"] --> E["_depth_map\n(H×W uint16)"]

    B --> F["YOLO v8m-seg\nconf threshold"]
    F --> G["Bounding Boxes\n+ class + confidence"]
    F --> H["YOLO Seg Masks\n(coarse)"]

    G --> I{"SAM2\nenabled?"}
    I -- Yes --> J["SAM2 segment_from_boxes\n→ pixel-perfect masks"]
    I -- No --> H

    J --> K["Per-detection mask"]
    H --> K

    B --> L{"Depth Anything V2\nenabled?"}
    E --> L
    L -- "refine" --> M["Enhanced Depth\n(neural + stereo fused)"]
    L -- "no stereo" --> N["Monocular Depth\n(relative → metric)"]
    L -- "off" --> E

    K --> O["GroundMapper.update\n(RANSAC ground plane)"]
    E --> O
    M --> O
    O --> P["GroundMapper.locate\nper detection"]

    G --> P
    K --> P

    P --> Q["floor_pos_m: x, y\n(robot body frame)"]

    G --> R["Face Recognition\n(CNN + cache)"]
    B --> R
    R --> S["name / Unknown #N"]

    Q --> T["✅ Detection Dict\nclass · confidence · bbox\ndistance_m · floor_pos_m\nname · mask"]
```

---

## Ground Mapper — How Floor Localization Works

```mermaid
flowchart TD
    subgraph PerFrame["Once per frame"]
        D["Depth Map\n(uint16 mm)"] --> U["Sparse Unproject\n(every 8th pixel)\n→ 3D point cloud\nin camera frame"]
        U --> R["RANSAC\n200 iters, 3cm threshold\nfit plane n·p = d"]
        R --> GP{"Enough\ninliers?"}
        GP -- "≥15% inliers" --> RE["Refit on inliers\n(PCA smallest eigvec)"]
        GP -- "too few" --> FB["Fallback:\nplane from\ncam_height + tilt"]
        RE --> PLANE["Ground Plane\n(in camera frame)"]
        FB --> PLANE
    end

    subgraph PerObject["For each detection"]
        MASK["SAM2 Mask\n(or YOLO mask)"] --> MD["Median depth\nof valid pixels\ninside mask"]
        BBOX["YOLO BBox\nx1,y1,x2,y2"] --> BC["Bottom-center pixel\nu = (x1+x2)/2\nv = y2"]
        MD -- "fallback: central\nbbox patch" --> DEPTH["Robust depth\n(meters)"]
        BC --> BP["Back-project\npixel → 3D\ncamera frame"]
        DEPTH --> BP

        BP --> PROJ["Project onto\nground plane"]
        PLANE --> PROJ

        PROJ --> XFORM["Camera → Body frame\nR_tilt · axis_swap\n+ cam_height offset"]
        XFORM --> OUT["floor_pos_m\n(x_fwd, y_left)\nmeters from robot"]
    end

    style PerFrame fill:#1a1a2e,stroke:#e94560,color:#eee
    style PerObject fill:#16213e,stroke:#0f3460,color:#eee
```

---

## Coordinate Systems

```mermaid
graph LR
    subgraph CAM["Camera Frame (OpenCV)"]
        direction TB
        CX["X → right"]
        CY["Y → down"]
        CZ["Z → forward"]
    end

    subgraph BODY["Robot Body Frame"]
        direction TB
        BX["X → forward"]
        BY["Y → left"]
        BZ["Z → up"]
    end

    subgraph WORLD["World Frame (Odom)"]
        direction TB
        WX["X → initial forward"]
        WY["Y → initial left"]
        WT["θ → robot heading"]
    end

    CAM -- "R_tilt · axis_swap\n+ height offset" --> BODY
    BODY -- "rotate by θ\ntranslate by (rx, ry)" --> WORLD
```

| Transform | What it does |
|-----------|-------------|
| **Camera → Body** | Swap axes (Z→X, −X→Y, −Y→Z), undo tilt, add camera height |
| **Body → World** | Rotate by `robot_theta`, translate by `(robot_x, robot_y)` |

---

## Obstacle Map Integration

```mermaid
flowchart LR
    DET["Detections\nwith floor_pos_m"] --> OM["ObstacleMap\n.update_from_detections()"]
    OM --> BW["body → world\nfor each detection"]
    BW --> MERGE["Merge or add\nnamed object\n(EMA smoothing)"]
    BW --> GRID["Rasterize into\n400×400 grid\n(5cm cells)"]
    GRID --> RAY["query_ray()\nobstacle avoidance"]
    MERGE --> DIST["get_objects_with_distance()\nlabel + dist + bearing"]

    RC["RobotController\nfollow / go_to"] --> RAY
    RC --> DIST
```

---

## 3D Scene Reconstruction

```mermaid
flowchart TD
    DEPTH["Filtered Depth"] --> UNPROJ["Depth Unprojection\n→ 3D points\n(cam → body → world)"]
    DUST["DUSt3R Stereo\n(optional, async)"] --> DENSE["Dense Point Cloud\n(cam → world)"]
    SEG["Instance Seg Map\n+ class info"] --> COLOR["Semantic Coloring\nper-class palette"]

    UNPROJ --> SPLAT["Gaussian Splat Buffer\n80k max, spatial hash merge"]
    DENSE --> SPLAT
    COLOR --> SPLAT

    SPLAT --> MESH["Convex Hull Meshes\nper object instance"]
    SPLAT --> TRACK["Object Tracker\nEMA position + size"]
    TRACK --> SG["Scene Graph\nnodes + spatial edges\n(on_top_of, next_to, etc.)"]

    SPLAT --> UI["Web UI\nThree.js renderer\nsplats + meshes + trajectory"]
    SG --> UI
```

---

## Server Wiring

```mermaid
flowchart TD
    subgraph Connections
        WS["WebSocket :9090\nRobot ↔ Server"]
        HTTP["HTTP :8080\nWeb UI"]
    end

    subgraph Inbound["Robot → Server (binary)"]
        V1["0x01 MSG_VIDEO\nleft JPEG"]
        V2["0x04 MSG_VIDEO_RIGHT\nright JPEG"]
        DP["0x02 MSG_DEPTH\nzlib uint16"]
        AU["0x03 MSG_AUDIO_IN\nPCM from mic"]
    end

    subgraph Outbound["Server → Robot"]
        AO["0x10 MSG_AUDIO_OUT\nGemini speech"]
        CMD["JSON commands\nmove · rotate_head\ndance · wave · nod"]
    end

    WS --> Inbound
    Outbound --> WS

    V1 --> FP["FrameProcessor"]
    V2 --> FP
    DP --> FP
    AU --> GEM["Gemini Live\nbidirectional audio+video"]
    GEM -- transcript --> DISP["CommandDispatcher\nregex → actions"]
    DISP --> RC["RobotController"]
    RC --> CMD
    GEM --> AO

    HTTP --> UI["Web UI endpoints\n/frame /scene_data\n/detections /transcript\n/cmd"]
```

---

## Module Map

| Module | Purpose |
|--------|---------|
| `server.py` | Entry point — HTTP + WS server, Gemini session, wires everything |
| `robot_client.py` | Runs on robot — ROS2 subs, streams to server, executes commands |
| `frame_processor.py` | YOLO + SAM2 + faces + depth + ground mapper + scene recon |
| `ground_mapper.py` | RANSAC ground plane, floor localization, calibration helper |
| `robot_controller.py` | Follow, track, go-to, obstacle avoidance, movement commands |
| `scene_reconstructor.py` | Gaussian splat accumulation, object meshes, trajectory |
| `scene_graph.py` | Semantic graph — spatial relations between tracked objects |
| `obstacle_map.py` | 2D occupancy grid, ray queries, detection-based updates |
| `depth_model.py` | Depth Anything V2 neural depth (optional) |
| `sam2_segmenter.py` | SAM2 instance segmentation (optional) |
| `stereo_depth.py` | Stereo disparity, point clouds, gradient maps |
| `dust3r_reconstructor.py` | DUSt3R dense 3D from stereo pairs (optional) |
