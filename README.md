# InfernoGuard PC Inference Pipeline (`pc-infernoguard-inference`)

High-performance, fault-isolated multi-stream video ingestion and specialized multi-model YOLO inference pipeline designed for wildfire detection.

## Architecture Highlights
- **Multi-Camera Ingestion**: Resilient OpenCV-based capture with bounded buffers, frame-age filtering, automatic reconnection, and per-camera health telemetry.
- **Ultralytics `.streams` Compatibility**: Reads camera sources from a standard `.streams` file matching Ultralytics `LoadStreams` semantics.
- **Multi-Model Orchestration**: Executes multiple specialized YOLO models from a `models.yaml` manifest sequentially in the inference worker with model failure isolation.
- **Credential Protection**: Redacts sensitive credentials (`user:password@` and tokens) across all logs and telemetry.

---

## Streams File Specification (`.streams`)

> **IMPORTANT REQUIREMENT FOR `.streams` FILES**:
> - **One URI per line** (RTSP, HTTP, video file path, or numeric webcam device index).
> - **No comments** (lines beginning with `#` are not supported and will raise a validation error because Ultralytics `LoadStreams` treats each word as a stream URL).
> - **No spaces** (Ultralytics splits on whitespace; each URI must be a single whitespace-free token).
> - Empty lines are ignored.
> - UTF-8 with or without BOM (`utf-8-sig`) is supported.

### Example `cameras.streams`:
```
0
rtsp://camera1.local:554/live/stream1
rtsp://viewer:placeholder_secret@192.168.1.100:554/h264/ch1
```

---

## Models Manifest Specification (`models.yaml`)

Specialized models are registered via a YAML manifest specifying weights, device, confidence thresholds, and class mappings:

```yaml
models:
  - name: "primary_inferno"
    path: "models/inferno.pt"
    device: "auto"
    conf_threshold: 0.25
    iou_threshold: 0.45
    imgsz: 640
    half_precision: false
    max_det: 300
    warmup_iterations: 1
    fallback_to_cpu: true
    expected_classes:
      0: "fire"
      1: "smoke"
      2: "spark"
    strict_classes: false
```

---

## Running the Pipeline

### Ingestion & Multi-Model Inference:
```bash
python -m pc_infernoguard.main_capture_demo --streams-file cameras.streams --models-file models.yaml
```

### Quick Single-Source Test:
```bash
python -m pc_infernoguard.main_capture_demo --source 0 --model-path models/inferno.pt
```

### Ultralytics Native Streams Comparison:
```bash
python -m pc_infernoguard.main_capture_demo --streams-file cameras.streams --native-streams
```

### Running with Limited Duration:
```bash
python -m pc_infernoguard.main_capture_demo --source 0 --duration 5.0
```
