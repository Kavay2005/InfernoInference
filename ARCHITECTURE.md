# Architecture: `pc-infernoguard-inference`

## 1. System Overview

`pc-infernoguard-inference` is the canonical PC-targeted inference architecture for InfernoGuard. It provides low-latency ingestion, thread-safe bounded queuing, inference execution against YOLO models (`models/inferno.pt`), and structured detection output.

This document covers:
- **Component A**: Camera Input and Multiple Streams
- **Component B**: CPU and GPU Inference Runtime Execution

---

## 2. Ingestion & Inference Architecture

```
                        +---------------------------------------------+
                        |                 Video Sources               |
                        |   - Local Webcam (cv2.VideoCapture(index))  |
                        |   - IP/RTSP Streams (cv2.VideoCapture(url)) |
                        +---------------------------------------------+
                                               |
                                               v
                        +---------------------------------------------+
                        |      StreamCapture (Per-Camera Worker)       |
                        |   - Background daemon thread                |
                        |   - Bounded deque(maxlen=1) [Fresh frame]   |
                        |   - Independent health state machine        |
                        |   - Automatic exponential reconnect         |
                        +---------------------------------------------+
                                               |
                                               v
                        +---------------------------------------------+
                        |            MultiStreamManager               |
                        |   - Stream lifecycle orchestration          |
                        |   - Failure isolation across streams        |
                        |   - Per-stream telemetry (FPS, drop count)  |
                        +---------------------------------------------+
                                               |
                                               +------------------------+
                                               |                        |
                                               v                        v
                        +------------------------------+ +------------------------------+
                        | Batch Adapter for YOLO       | | Telemetry & Health Reporting |
                        | to_ultralytics_batch(batch)  | | get_health_report()          |
                        +------------------------------+ +------------------------------+
                                               |
                                               v
                        +---------------------------------------------+
                        | DeviceManager (pc_infernoguard/inference)   |
                        | - Resolves device ('cpu', 'cuda', 'auto')   |
                        | - Fallback policy if CUDA unavailable       |
                        | - DeviceInfo hardware capabilities          |
                        +---------------------------------------------+
                                               |
                                               v
                        +---------------------------------------------+
                        | InferenceEngine (pc_infernoguard/inference) |
                        | - Loads models/inferno.pt once at startup   |
                        | - Validates hazard classes: fire/smoke/spark|
                        | - Performs runtime warmup pass              |
                        | - Unified predict_batch() / predict_frame() |
                        +---------------------------------------------+
                                               |
                                               v
                        +---------------------------------------------+
                        | Detection Output Contracts                  |
                        | - List[InferenceResult]                     |
                        |   - source_id, frame_id, timestamps         |
                        |   - detections: Tuple[DetectionBox, ...]    |
                        |   - preprocess_ms, inference_ms, post_ms    |
                        |   - device_used                             |
                        +---------------------------------------------+
```

---

## 3. Component Ownership and Responsibilities

### 3.1 `StreamCapture` (`pc_infernoguard/capture/stream_capture.py`)
- **Ownership**: Owns a single video capture lifecycle and worker thread.
- **Worker Thread**: Continuously executes `cv2.VideoCapture.read()`.
- **Bounded Buffer**: Stores only up to `buffer_size` frames (default `1`) using `collections.deque(maxlen=buffer_size)`. Incoming new frames automatically replace older unread frames, preventing lag and memory leaks.
- **Failure Recovery**: On read failure or disconnection, transitions to `RECONNECTING`, calculates exponential backoff (`reconnect_delay_sec * (backoff_factor ** attempt)` capped at `max_reconnect_delay_sec`), releases the capture descriptor, and re-attempts connection.
- **Telemetry**: Tracks FPS, total frames read, dropped frames, reconnect attempts, and last active timestamp.

### 3.2 `MultiStreamManager` (`pc_infernoguard/capture/multi_stream_manager.py`)
- **Ownership**: Orchestrates multiple `StreamCapture` instances.
- **Fault Isolation**: Completely decouples stream failures. If stream A drops out or hangs, stream B continues operating with zero throughput penalty.
- **Batch Provider**: `read_batch()` returns a mapping `{source_id: FrameData}` of all presently available freshest frames.
- **Context Management**: Supports `with MultiStreamManager(cfg) as manager:` to ensure deterministic startup and teardown of all camera resources.

### 3.3 `DeviceManager` (`pc_infernoguard/inference/device_manager.py`)
- **Ownership**: Device detection, capability querying, and resolution.
- **Device Support**: Explicit selectable CPU option, CUDA options (`cuda`, `cuda:0`), and dynamic `auto` mode.
- **Fallback Policy**: If CUDA is requested but unavailable (`torch.cuda.is_available() == False`), either logs a descriptive warning and falls back to CPU (if `fallback_to_cpu=True`) or raises a clear `RuntimeError` (if `fallback_to_cpu=False`).
- **Telemetry**: Exposes `DeviceInfo` detailing device name, memory, CUDA version, and PyTorch version.

### 3.4 `InferenceEngine` (`pc_infernoguard/inference/inference_engine.py`)
- **Ownership**: Model weights lifecycle, runtime warmup, and batch inference execution.
- **Single Startup Load**: Loads `models/inferno.pt` once during startup.
- **Model Validation**: Confirms model task is `detect` and class mapping includes `{0: 'fire', 1: 'smoke', 2: 'spark'}`.
- **Runtime Warmup**: Performs dummy inference pass during startup to initialize PyTorch JIT execution graphs and eliminate first-frame latency spikes.
- **Unified API**: Exposes identical `predict_frame(frame)` and `predict_batch(images)` API across CPU and CUDA execution backends.

### 3.5 Detection Contracts (`pc_infernoguard/inference/inference_types.py`)
- `DetectionBox`: Immutable container with class index, class name, confidence, absolute pixel coordinates `(x1, y1, x2, y2)`, and normalized coordinates `(x1_norm, y1_norm, x2_norm, y2_norm)`.
- `InferenceResult`: Immutable container linking detection boxes to source stream ID, sequence frame ID, capture timestamps (UTC and monotonic), latency breakdown (preprocess, inference, postprocess ms), and the compute device used.

---

## 4. Official References

- PyTorch CUDA Semantics: https://docs.pytorch.org/docs/stable/notes/cuda.html
- Ultralytics Predict Mode: https://docs.ultralytics.com/modes/predict/
- Ultralytics Thread-Safe Inference: https://docs.ultralytics.com/guides/yolo-thread-safe-inference/
- OpenCV `VideoCapture`: https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html
