# Architecture Decision Log (`DECISIONS.md`)

This document records the architectural decisions, trade-offs, evaluated alternatives, and user approval for `pc-infernoguard-inference`.

---

## ADR 0001: Resilient Custom Multi-Stream Ingestion vs. Ultralytics `LoadStreams`

### Context
In Part A of `pc-infernoguard-inference`, the system must ingest video frames from multiple camera inputs (local webcams and RTSP/IP streams) concurrently, ensuring low latency, bounded memory, and zero frame lag.

Ultralytics YOLO provides a built-in `LoadStreams` loader (`ultralytics.data.loaders.LoadStreams`).
Inspection of `LoadStreams.__next__` revealed:
1. **Cascade Failure**: If one camera stream disconnects or halts, `LoadStreams.__next__` calls `self.close()` and raises `StopIteration`, terminating **all** other camera streams simultaneously.
2. **Lack of Per-Stream Telemetry**: `LoadStreams` does not expose per-stream health transitions (`INITIALIZING`, `ACTIVE`, `STALE`, `RECONNECTING`, `FAILED`), dropped-frame counts, or independent exponential reconnection backoff.
3. **Stale Frame Accumulation**: Without explicit bounded queue control, network hiccups lead to latency drift between streams.

Section 7 of `AGENTS.md` explicitly requires:
> "Every frame and result must retain a stable camera/source identifier and timestamp. A failure or reconnection of one camera must not crash the other camera pipelines. Use bounded buffers and define what happens when inference cannot keep up."

### Decision
Implement an independent, resilient ingestion architecture:
1. **`StreamCapture`**: Dedicated daemon worker thread per camera wrapping OpenCV `VideoCapture`, utilizing a bounded queue (`collections.deque(maxlen=1)` by default) to guarantee newest-frame delivery, explicit health telemetry, and isolated exponential reconnect backoff.
2. **`MultiStreamManager`**: Orchestrates streams, aggregates health reports, isolates stream failures, and yields synchronized batches.
3. **`to_ultralytics_batch` & `.streams` Generator**: Utility adapters that convert manager batches directly into `model.predict(source=images)` inputs and generate standard Ultralytics `.streams` files.

### Alternatives Considered
1. **Direct Ultralytics `LoadStreams`**: Rejected due to all-streams shutdown on any single stream disconnect.
2. **AsyncIO (`aiortsp` / GStreamer Python)**: Deferred as unnecessary complexity for PC target where OpenCV with native OS threads meets all latency requirements.

### Status
Approved by user on 2026-10-21.

---

## ADR 0002: Device-Agnostic Execution with Configurable Fallback Policy

### Context
In Part B of `pc-infernoguard-inference`, model execution must support CPU and NVIDIA GPU devices.
The host environment inspection revealed an AMD Radeon GPU with a CPU-only PyTorch build (`torch.cuda.is_available() == False`).
Section 5 of `AGENTS.md` states:
> "CPU inference must be an explicit selectable option. NVIDIA GPU inference may use a compatible CUDA-enabled PyTorch/Ultralytics installation. Use the same application-level detection interface across CPU and GPU backends where practical. Check GPU availability and runtime compatibility rather than assuming a CUDA GPU is present."

### Decision
Implement a decoupled device management and inference engine:
1. **`DeviceManager`**:
   - Explicit device selection: `"cpu"`, `"cuda"`, `"cuda:0"`, or `"auto"`.
   - Dynamic CUDA hardware inspection via `torch.cuda.is_available()` and `torch.cuda.device_count()`.
   - Configurable fallback policy (`fallback_to_cpu: bool = True`): If CUDA is requested on a host lacking CUDA hardware or drivers, log an informative warning with environment diagnostics and safely execute on CPU. If strict enforcement is desired, setting `fallback_to_cpu=False` raises an explicit `RuntimeError`.
2. **`InferenceEngine`**:
   - Loads `models/inferno.pt` once during startup.
   - Performs runtime warmup (`warmup_iterations >= 1`) using dummy frames to initialize JIT graphs.
   - Validates class mapping (`{0: 'fire', 1: 'smoke', 2: 'spark'}`).
   - Emits structured detection output (`InferenceResult` / `DetectionBox`) with latency profiling (preprocess, inference, postprocess ms).

### Status
Approved by user on 2026-10-21.

### Evidence & Validation
Verified with unit tests in `tests/test_device_manager.py`, `tests/test_inference_engine.py`, and `tests/test_pipeline_integration.py`. Device resolution, fallback handling, warmup, and batch prediction executed successfully across 25 automated tests.

---

## ADR 0003: Strict Elimination of Fabricated Runtime Values and Thread-Safe Pipeline Refactoring

### Context
To maintain industrial reliability and auditability, measured runtime facts (such as fps estimates, frame timestamps, latency metrics, device specifications, and model identities) must reflect actual measurements rather than invented placeholders (e.g. `fps = 30.0` or `device = "cpu"` when no model ran). Furthermore, live stream pacing must not insert artificial sleeps that cause RTSP/USB buffer accumulation, and concurrent data access across multi-stream dictionaries and model inference wrappers must be strictly serialized.

### Decision
1. **Measured Facts Over Placeholders**:
   - `FrameData.fps_estimate`, `StreamHealth.last_frame_timestamp`, and `StreamHealth.current_fps` default to `None` until real measurements occur.
   - Pacing logic applies only to local video files (`_is_file_source()`); live streams read continuously without artificial pacing sleeps.
   - `DetectionBox.model_name` and `DeviceInfo.torch_version` are strictly required parameters.
   - `InferenceResult.end_to_end_ms` measures wall-clock time from frame acquisition to result emission: `(time.monotonic() - timestamp_mono) * 1000`.
   - `MultiModelPipeline.predict_combined()` reports `device_used="none"` when no models run.
2. **Robust Multi-Threading and Invariant Enforcement**:
   - `MultiStreamManager` uses `threading.Lock()` to protect stream registry mutations and snapshots `list(self._streams.items())` across all iterations.
   - `InferenceConfig` is frozen (`frozen=True`) with optional strict class validation (`strict_classes: bool = False`, `expected_classes`).
   - `MultiModelPipeline.register_model()` forbids mutating model names, requiring registered names to match `engine.cfg.model_name`.
   - `ThreadSafeInferenceWrapper` adopts `threading.Lock()` to serialize YOLO model inference per Ultralytics concurrency guidance.
   - `FrameData` uses `eq=False` to prevent NumPy ambiguous boolean evaluation errors on dataclass equality checks.

### Status
Approved and implemented on 2026-10-22.

### Evidence & Validation
Validated across 41 unit and integration tests (`pytest tests -v` passing 100%), including specific tests for stale status derivation, live non-pacing, metadata validation in `predict_batch`, name mismatch detection in model registration, strict class validation, and ndarray equality safety.

---

## ADR 0004: Manifest-Driven Multi-Model Orchestration, Ultralytics `.streams` Compliance, and Failure Isolation

### Context
In real-world wildfire detection deployments, multiple specialized models (e.g. primary fire detector, smoke verifier, spark detector) run concurrently across video frames ingested from an Ultralytics `.streams` file. Ultralytics defines only the `.streams` camera format; no standard format exists for multi-model definitions. Cameras must be ingested without leaking network credentials into logs or telemetry, and runtime model exceptions (e.g. corrupt tensor, out of memory) must not take down the entire inference loop or discard results from healthy models.

### Decision
1. **Ultralytics `.streams` Specification Compliance**:
   - `parse_streams_file()` conforms strictly to Ultralytics LoadStreams: tolerates UTF-8 with/without BOM (`utf-8-sig`), splits on whitespace, treats `#` as token characters (no comment syntax), and raises `ValueError` on empty files.
   - `source_id_for(uri, index)` produces sanitized, stable camera IDs derived from hostname and path or device index, and strictly excludes credentials. Suffixes `_2`, `_3` resolve collisions.
   - `redact_uri(uri)` masks `user:password@` as `***:***@` in all log messages and telemetry reports.
2. **YAML Models Manifest (`models.yaml`)**:
   - `load_models_manifest()` validates model definitions matching `InferenceConfig` fields.
   - Validates unique model names, physical existence of weights files, and required fields (`name`, `path`).
3. **Multi-Model Batching & Failure Isolation**:
   - `MultiModelPipeline.from_manifest(path)` builds one `InferenceEngine` per entry.
   - `predict_batch(frames)` calls each model's `predict_batch` once per model (batched inference).
   - Runtime failures in any individual model are caught, logged with the model name, recorded in `pipeline.last_errors[name]`, and isolated so healthy model detections are preserved.
4. **Native Streams Baseline Runner (`NativeStreamsRunner`)**:
   - Implements baseline comparison via `model.predict(source=<file>, stream=True)`.
   - Documents edge limitations: no reconnection, no per-camera health, arrival-time timestamps only, black frames on dropped streams, and process-abort on startup if any stream is unreachable.

### Status
Approved and implemented on 2026-10-22.

### Evidence & Validation
Validated across 59 automated tests (`pytest tests -v` passing 100%), including manifest schema validation, duplicate detection, BOM and empty streams parsing, credential redaction, model failure isolation, native stream order mapping, and CLI smoke test with `--duration 3`.


