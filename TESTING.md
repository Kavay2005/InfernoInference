# Testing & Validation Report (`TESTING.md`)

This document provides reproducible instructions and records test executions for `pc-infernoguard-inference`.

---

## 1. Automated Test Suite

### Environment
- Python: 3.12.1
- PyTorch: 2.11.0+cpu (CUDA: False)
- OpenCV: 5.0.0
- Ultralytics: 8.4.129
- Pytest: 9.1.1

### Command
```bash
python -m pytest tests -v
```

### Execution Results (Recorded: 2026-10-22)
```
tests/test_demo_smoke.py::test_demo_smoke_with_duration_on_local_video PASSED [  1%]
tests/test_demo_smoke.py::test_demo_error_when_both_source_and_streams_file_provided PASSED [  3%]
tests/test_device_manager.py::test_cpu_device_resolution PASSED          [  5%]
tests/test_device_manager.py::test_auto_device_resolution_on_current_host PASSED [  6%]
tests/test_device_manager.py::test_cuda_fallback_behavior_when_cuda_unavailable PASSED [  8%]
tests/test_device_manager.py::test_cuda_resolution_when_cuda_available PASSED [ 10%]
tests/test_device_manager.py::test_invalid_device_name_raises PASSED     [ 11%]
tests/test_device_manager.py::test_get_device_info_cpu PASSED            [ 13%]
tests/test_device_manager.py::test_list_available_devices_includes_cpu PASSED [ 15%]
tests/test_inference_engine.py::test_inference_config_validation PASSED  [ 16%]
tests/test_inference_engine.py::test_detection_box_validation PASSED     [ 18%]
tests/test_inference_engine.py::test_inference_engine_initialization_and_warmup PASSED [ 20%]
tests/test_inference_engine.py::test_inference_engine_predict_frame PASSED [ 22%]
tests/test_inference_engine.py::test_inference_engine_predict_batch PASSED [ 23%]
tests/test_inference_engine.py::test_inference_engine_predict_batch_rejecting_missing_metadata PASSED [ 25%]
tests/test_inference_engine.py::test_inference_engine_strict_classes_validation PASSED [ 27%]
tests/test_inference_engine.py::test_inference_engine_empty_batch PASSED [ 28%]
tests/test_models_manifest.py::test_load_models_manifest_valid PASSED    [ 30%]
tests/test_models_manifest.py::test_load_models_manifest_nonexistent_manifest_raises PASSED [ 32%]
tests/test_models_manifest.py::test_load_models_manifest_nonexistent_weights_raises PASSED [ 33%]
tests/test_models_manifest.py::test_load_models_manifest_duplicate_model_names_raises PASSED [ 35%]
tests/test_models_manifest.py::test_load_models_manifest_missing_name_raises PASSED [ 37%]
tests/test_models_manifest.py::test_load_models_manifest_empty_models_list_raises PASSED [ 38%]
tests/test_multi_model_pipeline.py::test_multi_model_pipeline_registration PASSED [ 40%]
tests/test_multi_model_pipeline.py::test_multi_model_pipeline_register_model_name_mismatch_raises PASSED [ 42%]
tests/test_multi_model_pipeline.py::test_multi_model_pipeline_predict_combined PASSED [ 44%]
tests/test_multi_model_pipeline.py::test_multi_model_pipeline_predict_combined_no_models_ran PASSED [ 45%]
tests/test_multi_model_pipeline.py::test_multi_model_pipeline_predict_batch PASSED [ 47%]
tests/test_multi_model_pipeline.py::test_multi_model_pipeline_failure_isolation PASSED [ 49%]
tests/test_multi_model_pipeline.py::test_multi_model_pipeline_from_manifest PASSED [ 50%]
tests/test_multi_stream_manager.py::test_multi_stream_manager_concurrent_reads PASSED [ 52%]
tests/test_multi_stream_manager.py::test_stream_isolation_failing_stream_does_not_crash_healthy_stream PASSED [ 54%]
tests/test_multi_stream_manager.py::test_multi_stream_manager_max_frame_age_sec PASSED [ 55%]
tests/test_multi_stream_manager.py::test_multi_stream_manager_thread_safe_add_remove_no_dict_mutation_error PASSED [ 57%]
tests/test_native_streams_runner.py::test_native_streams_runner_mapping_with_two_distinct_sources PASSED [ 59%]
tests/test_pipeline_integration.py::test_end_to_end_ingestion_and_inference_pipeline PASSED [ 61%]
tests/test_stream_capture.py::test_stream_capture_lifecycle_with_synthetic_video PASSED [ 62%]
tests/test_stream_capture.py::test_bounded_buffer_drops_old_frames PASSED [ 64%]
tests/test_stream_capture.py::test_invalid_source_isolation_and_no_crash PASSED [ 66%]
tests/test_stream_capture.py::test_stale_status_derivation_without_mutating_stored_state PASSED [ 67%]
tests/test_stream_capture.py::test_is_file_source_and_no_pacing_on_live_sources PASSED [ 69%]
tests/test_stream_capture.py::test_fps_smoothing_seeds_from_first_measurement_no_fake_30 PASSED [ 71%]
tests/test_stream_types.py::test_stream_status_enum_values PASSED        [ 72%]
tests/test_stream_types.py::test_frame_data_creation_and_immutability PASSED [ 74%]
tests/test_stream_types.py::test_frame_data_validation PASSED            [ 76%]
tests/test_stream_types.py::test_stream_health_defaults PASSED           [ 77%]
tests/test_stream_types.py::test_stream_health_is_stale_derivation PASSED [ 79%]
tests/test_stream_types.py::test_frame_data_comparison_does_not_raise_numpy_ambiguity PASSED [ 81%]
tests/test_thread_safe_engine.py::test_thread_safe_wrapper_delegation PASSED [ 83%]
tests/test_thread_safe_engine.py::test_concurrent_multi_threaded_inference PASSED [ 84%]
tests/test_thread_safe_engine.py::test_thread_safe_wrapper_predict_batch PASSED [ 86%]
tests/test_ultralytics_streams_file.py::test_generate_streams_file_round_trip PASSED [ 86%]
tests/test_ultralytics_streams_file.py::test_parse_streams_file_bom_clean PASSED [ 88%]
tests/test_ultralytics_streams_file.py::test_parse_streams_file_comment_line_raises_value_error PASSED [ 90%]
tests/test_ultralytics_streams_file.py::test_parse_streams_file_line_with_space_raises_value_error PASSED [ 91%]
tests/test_ultralytics_streams_file.py::test_parse_streams_file_empty_file_raises_value_error PASSED [ 93%]
tests/test_ultralytics_streams_file.py::test_parse_streams_file_duplicate_uri_raises_value_error PASSED [ 95%]
tests/test_ultralytics_streams_file.py::test_redact_uri_masks_credentials_and_query_tokens PASSED [ 96%]
tests/test_ultralytics_streams_file.py::test_source_id_for_stability_credentials_and_deduplication PASSED [ 98%]
tests/test_ultralytics_streams_file.py::test_to_ultralytics_batch_conversion PASSED [100%]

============================= 60 passed in 38.19s =============================
```

---

## 2. Test Coverage & Invariant Verification

| Test Module | Verified Invariants |
|---|---|
| `tests/test_device_manager.py` | - Explicit CPU resolution (`cpu`).<br/>- Auto resolution inspecting host hardware.<br/>- Fallback behavior when CUDA is requested on non-CUDA host (`fallback_to_cpu=True` vs strict `RuntimeError`).<br/>- Mock CUDA device detection and out-of-bounds index handling.<br/>- `DeviceInfo` metadata generation (strictly requiring `torch_version`). |
| `tests/test_inference_engine.py` | - Validation of `InferenceConfig` inputs (`conf`, `imgsz % 32 == 0`, `expected_classes`).<br/>- `DetectionBox` coordinate order `x1 <= x2, y1 <= y2`, confidence bounds, and required `model_name`.<br/>- Model loading once during startup and verification of fire/smoke/spark classes.<br/>- Warmup execution (`engine.is_warm == True`).<br/>- `predict_frame` and `predict_batch` output structure, latency timing, `end_to_end_ms` calculation, and empty batch handling.<br/>- Rejection of missing/invalid frame metadata (must be `Sequence[FrameData]`).<br/>- `strict_classes=True` raising `ValueError` on class mismatch. |
| `tests/test_models_manifest.py` | - Loading and validation of YAML manifests (`models.yaml`) into `List[InferenceConfig]`.<br/>- Validation of required fields (`name`, `path`).<br/>- Verification that referenced model weights exist on disk (`FileNotFoundError`).<br/>- Detection of duplicate model names (`ValueError`).<br/>- Rejection of empty model lists or malformed dictionaries. |
| `tests/test_multi_model_pipeline.py` | - Multiple model registration with enforced matching `model_name`.<br/>- Rejection of mismatched engine names with `ValueError`.<br/>- Construction from manifest (`MultiModelPipeline.from_manifest`).<br/>- Batched multi-model execution (`predict_batch`) calling each model engine once per frame batch.<br/>- Model failure isolation: runtime exceptions in one model are logged and captured in `last_errors`, preserving detections from sibling models.<br/>- Combined prediction aggregating detections and computing max `end_to_end_ms`.<br/>- Clean fallback to `device_used="none"` and `0.0ms` when no models run. |
| `tests/test_native_streams_runner.py` | - Baseline comparison runner invoking Ultralytics native stream prediction.<br/>- Verification that streamed Results map strictly to camera sources by file order across multiple streams. |
| `tests/test_demo_smoke.py` | - CLI runner integration smoke test with `--duration 3` on a local synthetic video.<br/>- Verification that mutually exclusive `--source` and `--streams-file` arguments are rejected with exit code != 0. |
| `tests/test_pipeline_integration.py` | - Complete end-to-end integration: `MultiStreamManager` concurrent ingestion feeding batches to `InferenceEngine.predict_batch()`.<br/>- Preserves source stream IDs, frame IDs, and timestamps through to inference results.<br/>- Verification that `end_to_end_ms` is strictly positive. |
| `tests/test_stream_types.py` | - `FrameData` immutability (`frozen=True`) preventing downstream corruption.<br/>- `FrameData` equality (`eq=False`) preventing NumPy ambiguous truth value exceptions.<br/>- Strict validation (rejects invalid types and non-3D arrays).<br/>- `StreamHealth` initialization defaults (`last_frame_timestamp=None`, `current_fps=None`).<br/>- `StreamHealth.is_stale()` derivation logic under timeout. |
| `tests/test_stream_capture.py` | - Full lifecycle (start, capture synthetic frames, graceful stop).<br/>- Bounded queue frame drop verification (`dropped_frames > 0` under consumer delay).<br/>- Fresh-frame retrieval (`read_latest()`).<br/>- Invalid source error isolation without throwing uncaught exceptions.<br/>- `get_health()` returns `STALE` status copy when frame delta exceeds timeout without mutating internal state.<br/>- `_is_file_source()` identification ensuring live RTSP/USB cameras are never rate-paced.<br/>- FPS smoothing seeds cleanly from the first inter-frame measurement with no fake 30.0 default.<br/>- Honors `cfg.loop_video` for EOF looping and `cfg.pace_fps` for rate pacing. |
| `tests/test_multi_stream_manager.py` | - Concurrent frame capture across multiple streams.<br/>- **Critical AGENTS.md Sec. 7 Rule**: Failure or disconnection of one camera does not crash sibling cameras or the manager.<br/>- `max_frame_age_sec` filters out stale frames from returned batches.<br/>- Dynamic concurrent addition and removal of streams without `dictionary changed size during iteration` errors. |
| `tests/test_thread_safe_engine.py` | - Thread-safe inference wrapper serializing calls across worker threads using `threading.Lock`.<br/>- Support for `Sequence[FrameData]` in `predict_batch()`. |
| `tests/test_ultralytics_streams_file.py` | - Parsing of `.streams` files matching Ultralytics LoadStreams: UTF-8 BOM tolerance (`utf-8-sig`), whitespace splitting, treatment of `#` as tokens (no comment syntax), and rejection of empty files (`ValueError`).<br/>- `redact_uri` credential masking (`***:***@`) preventing password/token leaks in logs.<br/>- `source_id_for` stable ID generation from host/path without credentials and collision resolution with numeric suffixes (`_2`, `_3`). |

---

## 3. Manual Live Verification & CLI Runner

### Running Multi-Model Pipeline with Manifest:
```bash
python -m pc_infernoguard.main_capture_demo --streams-file cameras.streams --models-file models.yaml
```

### Running Quick Single-Source Test:
```bash
python -m pc_infernoguard.main_capture_demo --source 0 --model-path models/inferno.pt
```

### Running Native Ultralytics Stream Comparison Mode:
```bash
python -m pc_infernoguard.main_capture_demo --streams-file cameras.streams --native-streams
```

### Running with Limited Duration:
```bash
python -m pc_infernoguard.main_capture_demo --source 0 --duration 5.0
```

