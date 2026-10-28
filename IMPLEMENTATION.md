# Implementation Roadmap & Status (`IMPLEMENTATION.md`)

This document tracks the approved implementation phases, current execution status, and next steps for the canonical PC architecture `pc-infernoguard-inference`.

---

## 1. Project Phase Overview

| Phase | Description | Status | Reference |
|---|---|---|---|
| **Part A** | Camera Input and Multiple Streams (Capture, bounded buffer, telemetry, failure isolation) | **COMPLETED** | ADR 0001 |
| **Part B** | CPU and GPU Inference Runtime Execution (`models/inferno.pt`, device selection, warmup, structured output) | **COMPLETED** | ADR 0002 |
| **Part C** | Detection Post-Processing & Output Formats (structured detections, camera-ID binding, coordinate mapping) | **READY FOR PLANNING** | `AGENTS.md` Sec. 4, 8 |
| **Part D** | Temporal Verification & Safety State Decisions (sparks, smoke persistence, safety state machine) | UPCOMING | `AGENTS.md` Sec. 4, 6 |
| **Part E** | Multi-Camera Benchmarking & System Resource Telemetry (CPU/RAM/GPU/VRAM profiling, latency profiling) | UPCOMING | `AGENTS.md` Sec. 5, 9 |

---

## 2. Completed Components

### Part A: Ingestion Subsystem
- [x] **`pc_infernoguard/config.py`**: `CameraConfig` and `MultiStreamConfig`.
- [x] **`pc_infernoguard/capture/stream_types.py`**: `StreamStatus`, `FrameData` (frozen), `StreamHealth`.
- [x] **`pc_infernoguard/capture/stream_capture.py`**: Threaded capture, rate pacing, reconnect backoff, bounded buffer.
- [x] **`pc_infernoguard/capture/multi_stream_manager.py`**: Multi-stream manager and failure isolation.
- [x] **`pc_infernoguard/utils/streams_file.py`**: `.streams` file parser, generator, batch adapter.

### Part B: Inference Subsystem
- [x] **`pc_infernoguard/inference/device_manager.py`**:
  - Device-agnostic resolution (`cpu`, `cuda`, `cuda:0`, `auto`).
  - Runtime CUDA inspection (`torch.cuda.is_available()`, `device_count()`).
  - Configurable CPU fallback policy when CUDA hardware or runtime is absent.
  - Device telemetry container (`DeviceInfo`).
- [x] **`pc_infernoguard/inference/inference_types.py`**:
  - `InferenceConfig`: Model path, device, thresholds (`conf`, `iou`), resolution (`imgsz`), FP16, warmup iterations.
  - `DetectionBox`: Structured immutable detection bounding box with pixel (`xyxy`) and normalized (`xyxy_norm`) bounds.
  - `InferenceResult`: Structured detection container with source stream ID, frame sequence ID, timestamps, and latency breakdown.
- [x] **`pc_infernoguard/inference/inference_engine.py`**:
  - Single startup model loading for `models/inferno.pt`.
  - Model verification for hazard classes: `{0: 'fire', 1: 'smoke', 2: 'spark'}`.
  - Runtime warmup pass to initialize execution graph before processing live frames.
  - Unified `predict_frame()` and `predict_batch()` APIs across devices.
- [x] **`pc_infernoguard/main_capture_demo.py`**:
  - Upgraded CLI runner with `--device`, `--conf`, `--imgsz`, `--half`, and `--warmup` flags.
  - Real-time logging of structured detections and latency breakdown.

---

## 3. Verification Summary

- Automated test suite: `pytest tests -v` (25 passed in 18.43s).
- Tests cover:
  - Device resolution, queries, and fallback behavior (both fallback and strict fail modes).
  - Model loading, class verification, and warmup forward passes.
  - Batch and single-frame inference with coordinate validation.
  - End-to-end integration: multi-stream ingestion feeding directly into `InferenceEngine`.
