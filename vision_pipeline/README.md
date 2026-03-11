# Robot Vision Pipeline

Real-time pipeline for stereo/depth capture, 3D point cloud, segmentation, incremental semantic map, and vision-language scene understanding.

## Components

1. **Capture** (`capture.py`) – Stereo and depth input
   - **ZED**: Stereolabs ZED/ZED2 via `pyzed` (install ZED SDK + `pip install pyzed`)
   - **OAK-D**: Luxonis OAK-D / OAK-D Lite / OAK-D Pro via `depthai` (`pip install depthai`)
   - **ROS**: Feed images from ROS topics (e.g. K1 robot with StereoNet depth) via `ROSCapture.set_frame()`
   - **OpenCV**: Webcam with synthetic depth for testing

2. **Point cloud** (`point_cloud.py`) – Back-project depth to 3D with optional RGB and labels.

3. **Segmenter** (`segmenter.py`) – YOLOv8-seg for lightweight instance segmentation (COCO classes); optional `min_area_pixels` to drop tiny blobs (&lt; ~2 in).

4. **Semantic map** (`semantic_map.py`) – Incremental global voxel map; fuses labeled point clouds as the robot moves (optional pose)

5. **Vision-language** (`vision_language.py`) – BLIP-2 for image captioning and “what do you see?” queries

## Quick start

```bash
# Install
pip install -r requirements-vision.txt

# Run with webcam (synthetic depth)
python run_vision_pipeline.py

# Run with ZED
python run_vision_pipeline.py --backend zed

# Run with OAK-D camera
python run_vision_pipeline.py --backend oak

# Enable VLM and run every 30 frames
python run_vision_pipeline.py --vlm --vlm-interval 30
```

## Using with ROS / K1 robot

1. Run your ROS node that publishes left image and depth (e.g. `/image_left_raw`, `/StereoNetNode/stereonet_depth`).
2. Create a `ROSCapture` and call `set_frame(rgb, depth)` from your image/depth callbacks.
3. In the pipeline, use `create_capture(backend="ros", fx=..., fy=..., cx=..., cy=..., depth_scale=0.001)` with your camera intrinsics.

## Pipeline API

```python
from vision_pipeline import VisionPipeline, create_capture

pipeline = VisionPipeline(
    capture_backend="opencv",
    segmenter_model="yolov8m-seg.pt",
    use_vlm=True,
)
pipeline.start()
result = pipeline.step()  # one frame: segment, point cloud, map update
# result["frame"], result["segmentation"], result["point_cloud"], result["semantic_map"]
# result["vlm_answer"] every N frames if use_vlm=True
map_pc = pipeline.get_map_point_cloud()
pipeline.stop()
```

## Simplified pipeline: CNN-only filtering

The pipeline uses the **YOLOv8-seg CNN** as the single filter: only points on detected segments (labels > 0) are added to the map. Small blobs are dropped via `min_area_pixels` (default 50). No classical depth or point-cloud filters.

- **`cnn_filter=True`** (default): only map points on segmented objects.
- **`cnn_filter=False`** or `--no-cnn-filter`: use all depth points.
- **`min_area_pixels`**: ignore segments smaller than this many pixels (~2 in at typical resolution).

## Optional: classical filters

For extra smoothing you can still use `filter_depth` and `filter_point_cloud_outliers` from `point_cloud.py` (bilateral depth, statistical outlier removal) in your own code; the default pipeline no longer uses them. For learned depth refinement you can plug in external CNNs (e.g. NLSPN, PENet) and pass cleaned depth into the pipeline. The default pipeline relies on the segmentation CNN only.

