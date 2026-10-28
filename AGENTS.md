# AGENTS.md — InfernoGuard Inference Architecture

## 1. Purpose

This repository is for designing and implementing a new, maintainable InfernoGuard inference pipeline from the beginning. The goal is an architecture that is understandable, testable, extensible, and reliable to deploy over the long term.

The existing inference pipeline is a reference only. Do not modify, refactor, delete, or merge it into the new pipeline unless the user explicitly approves that specific migration.

The project has two distinct architecture targets:

- `pc-infernoguard-inference`: development and testing on a general-purpose PC using CPU or a compatible NVIDIA GPU.
- `jetson-infernoguard-inference`: edge deployment on NVIDIA Jetson, with hardware- and JetPack-compatible runtime choices.

Do not assume these are already implemented or that any files have been created for them. Inspect the repository before acting.

## 2. Non-negotiable working rules

1. **Inspect before changing.** Read the repository structure, existing documentation, configuration, dependencies, entry points, tests, and Git status before proposing or editing code.
2. **Do not take architectural control away from the user.** Explain the intended design and wait for approval before implementing major architecture choices.
3. **No broad “fix everything” changes.** Make small, reviewable changes with a clear purpose.
4. **Do not modify the existing pipeline.** Keep it untouched as a reference unless the user explicitly authorizes a particular change.
5. **Do not silently add frameworks.** Do not introduce DeepStream, Triton, TensorRT, ONNX Runtime, message brokers, databases, web frameworks, or other infrastructure without explaining the need and receiving approval.
6. **No duplicate implementations.** Do not create alternate files such as `*_new.py`, `*_v2.py`, `*_final.py`, or `*_production.py` as a workaround. Agree on the canonical file and update it deliberately.
7. **No invented interfaces.** Never invent API endpoints, DTO fields, database columns, camera capabilities, hardware-control signals, configuration values, model thresholds, or safety rules.
8. **Official documentation first.** For framework, hardware, runtime, and protocol decisions, use current official documentation. Record the relevant URL and the version/release that applies. Clearly distinguish documented facts from assumptions and recommendations.
9. **Preserve user data.** Do not delete models, datasets, configuration, evidence, or existing code without explicit permission.
10. **Keep the project understandable.** Prefer clear interfaces and few concepts over clever abstractions, unnecessary layers, and premature optimization.
11. **Writing the comments in the code is mandatory.** Whenever you create, update or change any file, piece of file/code/script, code snippet, configuration, model weight or any other asset; update the explanation of the code in a multi-line comment to explain all parameter and what that part of code does.

## 3. Required workflow before implementation

Before making code changes:

1. Inspect the repository and report the current state.
2. Identify the exact task, affected components, and anything that must remain unchanged.
3. craete implementation plan using `/plan` mode:
   - objective and scope;
   - proposed architecture and data flow;
   - component responsibilities and interfaces;
   - official documentation references;
   - assumptions and unresolved decisions;
   - failure cases and recovery behavior;
   - validation and acceptance criteria;
   - rollback plan.
4. Present major architectural decisions to the user for approval.
5. Implement only the approved scope.
6. Update documentation and tests with the implementation.
7. Report changed files, tests run, results, limitations, and rollback instructions.

If the user asks only for research, documentation, or planning, do not implement code.

## 4. Architecture boundaries

The target architecture should make these responsibilities explicit. This is a design guide, not a claim that every component already exists.

1. **Camera input** — opens a local webcam for initial PC testing and later supports RTSP/IP camera streams.
2. **Capture and frame handling** — validates frames, maintains timestamps and camera identity, and uses bounded buffering. Avoid unbounded queues and stale-frame buildup.
3. **Preprocessing** — performs the model's documented preprocessing consistently and records relevant configuration.
4. **Inference runtime** — runs the selected fire, smoke, and spark models using an explicitly configured backend.
5. **Detection output** — returns structured detections associated with the correct camera and frame timestamp.
6. **Later analysis layers** — tracking, temporal verification, safety-state decisions, local hardware response, evidence storage, and backend event delivery are separate concerns and must be designed explicitly before being added.

Do not make the capture loop perform long-running inference or network uploads. Do not let a failed camera silently become a SAFE condition. A camera's health and a hazard decision are different states. Each camera must have independent source identity, health, tracking history, and state where those features are implemented.

## 5. PC pipeline rules

Canonical architecture name: `pc-infernoguard-inference`.

- Begin with a local webcam for testing.
- Add RTSP/IP camera input through a documented interface after webcam capture is validated.
- CPU inference must be an explicit selectable option.
- NVIDIA GPU inference may use a compatible CUDA-enabled PyTorch/Ultralytics installation.
- Use the same application-level detection interface across CPU and GPU backends where practical.
- Check GPU availability and runtime compatibility rather than assuming a CUDA GPU is present.
- Do not assume CPU and GPU produce identical latency or bit-for-bit identical outputs.
- Measure throughput, end-to-end latency, dropped/stale frames, CPU/RAM/GPU/VRAM usage, and detection quality.
- Start with a simple, observable implementation. Add parallel model execution or batching only after profiling demonstrates a need.
- TensorRT or ONNX Runtime is optional; do not add it to the initial pipeline without a documented requirement and approved benchmark plan.

Official references:
- OpenCV `VideoCapture`: https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html
- Ultralytics Predict mode: https://docs.ultralytics.com/modes/predict/
- Ultralytics thread-safe inference: https://docs.ultralytics.com/guides/yolo-thread-safe-inference/
- PyTorch CUDA semantics: https://docs.pytorch.org/docs/stable/notes/cuda.html
- TensorRT Quick Start: https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html
- ONNX Runtime execution providers: https://onnxruntime.ai/docs/execution-providers/

## 6. Jetson pipeline rules

Canonical architecture name: `jetson-infernoguard-inference`.

- Treat Jetson as a separate deployment target with its own compatibility checks and acceptance tests.
- Confirm the exact JetPack, Jetson Linux, CUDA, cuDNN, TensorRT, Python, and Ultralytics versions before choosing or installing dependencies.
- Never assume a TensorRT engine exported on a PC is compatible with the Jetson runtime or target GPU.
- Validate the exported engine and model outputs on the actual target hardware.
- FP16 may be evaluated before INT8. INT8 requires representative calibration data and a measured detection-quality comparison.
- DeepStream and Triton are optional architectural choices, not default dependencies.
- Choose between a Python capture/inference pipeline and a DeepStream media pipeline only after documenting requirements, compatibility, complexity, measured performance, and custom integration needs.
- Keep local safety response independent of cloud or mobile connectivity when that layer is introduced.
- Document installation, startup, service supervision, logging, recovery, and rollback for the target JetPack release.

Official references:
- Ultralytics NVIDIA Jetson guide: https://docs.ultralytics.com/guides/nvidia-jetson/
- NVIDIA Jetson Orin Nano setup: https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/setup_jetpack.html
- DeepStream Developer Guide: https://docs.nvidia.com/metropolis/deepstream/dev-guide/index.html
- DeepStream multi-source sample applications: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_C_Sample_Apps.html
- `nvstreammux`: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvstreammux.html
- `nvurisrcbin`: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvurisrcbin.html
- `nvinfer`: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html
- TensorRT architecture and concurrency: https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/how-trt-works.html

Always verify that each URL and documentation version matches the actual installed release before implementation.

## 7. Multi-camera requirements

- Every frame and result must retain a stable camera/source identifier and timestamp.
- A failure or reconnection of one camera must not crash the other camera pipelines.
- Use bounded buffers and define what happens when inference cannot keep up. Do not allow unlimited frame queues.
- Define reconnect, timeout, stale-frame, and shutdown behavior explicitly.
- Do not assume all cameras have the same resolution, frame rate, codec, or network stability.
- Validate each stream independently before multi-stream testing.
- Measure per-camera frame rate, latency, dropped frames, and resource usage.
- If batching is used, preserve source identity and correctly associate each inference result with its source frame.
- Treat webcam testing and RTSP testing as separate acceptance stages.

## 8. Models and inference behavior

- Model paths, class mappings, input size, confidence settings, and runtime backend must be explicit configuration, not hidden constants.
- Do not change model weights, class names, preprocessing, thresholds, or inference resolution without approval and a recorded evaluation plan.
- Load models once during normal startup unless a documented requirement dictates otherwise.
- Warm up the runtime where applicable and report startup/runtime failures clearly.
- Keep model execution separate from post-processing and future safety decision logic.
- Validate output shape, class mapping, confidence values, and coordinate transformations.
- Compare model outputs and detection quality when changing runtime or precision.
- Do not claim production readiness based only on successful imports, syntax checks, or a single webcam test.

## 9. Testing and acceptance

At minimum, tests should cover the parts relevant to the current task:

- webcam opens, produces valid frames, and shuts down cleanly;
- invalid source and failed RTSP connection are reported clearly;
- stream interruption and reconnection do not crash unrelated cameras;
- frame timestamps and camera identities remain correct;
- CPU and GPU backend selection works as configured;
- inference output shape and class mapping are correct;
- buffers remain bounded under load;
- graceful shutdown releases camera and runtime resources;
- multi-camera load tests report throughput, latency, dropped frames, and system resource use.

For later safety layers, add tests for small/distant smoke, intermittent sparks, reflections, camera failure, inference failure, backend outage, event recovery, and local response. Do not implement or claim safety behavior before its requirements are explicitly defined and validated.

## 10. Documentation and change reporting

Maintain these documents as the project grows:

- `AGENTS.md` — enduring project rules and constraints.
- `IMPLEMENTATION.md` — current approved implementation plan and status.
- `ARCHITECTURE.md` — current architecture, component ownership, interfaces, and data flow.
- `DECISIONS.md` — important decisions, alternatives, rationale, evidence, and approval status.
- `TESTING.md` — reproducible validation procedures and recorded results.

After each approved change, report:

- what changed and why;
- files added, modified, or removed;
- the architectural decision that authorized it;
- official references consulted;
- tests run and actual outcomes;
- known limitations and untested cases;
- how to revert the change.

Do not label a component “industrial-grade,” “production-ready,” “safe,” or “certified” without the corresponding evidence and validation.

## 11. User approval required

Ask the user before:

- replacing the capture framework or inference runtime;
- introducing DeepStream, Triton, TensorRT, ONNX Runtime, or a new major dependency;
- changing the model execution strategy from sequential to concurrent/batched;
- changing model weights, preprocessing, thresholds, or input resolution;
- changing the safety state machine or hardware-control behavior;
- changing backend contracts, database schema, authentication, or mobile-app integration;
- deleting or migrating existing code;
- making a change that affects deployment, safety, or compatibility.

Small implementation details within an already approved design may be handled directly, provided they preserve these rules and are documented.

## 12. Definition of done

A task is complete only when:

- the implementation matches the approved design;
- the change is limited to the agreed scope;
- relevant tests have been run and their results reported honestly;
- configuration and interfaces are documented;
- failure and recovery behavior is addressed for the changed component;
- the user can understand what changed and why;
- the existing pipeline remains untouched unless explicitly authorized.
