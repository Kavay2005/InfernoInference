"""CLI demo for validating camera capture and multi-model inference on PC."""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

from pc_infernoguard.capture.multi_stream_manager import MultiStreamManager
from pc_infernoguard.capture.stream_types import StreamStatus
from pc_infernoguard.config import CameraConfig, MultiStreamConfig
from pc_infernoguard.inference import (
    InferenceConfig,
    InferenceEngine,
    MultiModelPipeline,
    NativeStreamsRunner,
)
from pc_infernoguard.utils.streams_file import (
    generate_streams_file,
    parse_streams_file,
    redact_uri,
    source_id_for,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pc_infernoguard.demo")


def build_parser() -> argparse.ArgumentParser:
    """Build and configure the command-line argument parser for the PC ingestion & inference demo."""
    parser = argparse.ArgumentParser(
        description="InfernoGuard PC Ingestion & Multi-Model Inference Demo"
    )
    parser.add_argument(
        "--streams-file",
        "-f",
        type=str,
        default=None,
        help="Path to an Ultralytics .streams file containing camera stream URIs",
    )
    parser.add_argument(
        "--source",
        "-s",
        nargs="+",
        default=None,
        help="One or more camera sources for quick testing (e.g. 0 1 rtsp://example.com/live)",
    )
    parser.add_argument(
        "--models-file",
        "-m",
        type=str,
        default=None,
        help="Path to models.yaml manifest file configuring specialized models",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Shortcut path to a single YOLO weights file (builds a one-entry pipeline)",
    )
    parser.add_argument(
        "--native-streams",
        action="store_true",
        help="Run baseline comparison using Ultralytics native stream inference mode",
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        default=0,
        help="Optional run duration in seconds (0 = run until interrupted)",
    )
    return parser


def run_native_streams(args, sources: List[str]) -> None:
    """Run baseline comparison using Ultralytics native stream inference mode."""
    streams_path = args.streams_file
    temp_created = False
    if not streams_path:
        streams_path = "temp_cameras.streams"
        generate_streams_file(sources, streams_path)
        temp_created = True

    model_weight = args.model_path or "models/inferno.pt"
    logger.info(
        "Starting NativeStreamsRunner comparison on %s with model %s",
        streams_path,
        model_weight,
    )

    try:
        runner = NativeStreamsRunner(
            model_path=model_weight,
            streams_file=streams_path,
        )
        start_time = time.time()
        for res in runner.stream():
            now = time.time()
            if res.detections:
                for det in res.detections:
                    logger.info(
                        "[%s | %s] Native Detection: %s (conf: %.2f) at [%.0f, %.0f, %.0f, %.0f] | Latency: %.1fms",
                        det.model_name,
                        res.source_id,
                        det.class_name,
                        det.confidence,
                        det.x1,
                        det.y1,
                        det.x2,
                        det.y2,
                        res.total_latency_ms,
                    )
            if args.duration > 0 and (now - start_time) >= args.duration:
                logger.info("Requested duration of %.1fs reached.", args.duration)
                break
    finally:
        if temp_created:
            try:
                Path(streams_path).unlink(missing_ok=True)
            except Exception:
                pass


def main():
    """Main CLI entry point for PC capture ingestion and multi-model inference pipeline."""
    parser = build_parser()
    args = parser.parse_args()

    # 1. Enforce mutually exclusive source specification
    if args.source is not None and args.streams_file is not None:
        parser.error("Cannot provide both --source and --streams-file.")

    # Determine camera stream sources
    sources: List[str] = []
    if args.streams_file:
        sources = parse_streams_file(args.streams_file)
        logger.info("Loaded %d stream(s) from .streams file: %s", len(sources), args.streams_file)
    elif args.source:
        sources = args.source
    else:
        sources = ["0"]

    # Redact URIs in all logs
    redacted_sources = [redact_uri(s) for s in sources]
    logger.info("Configuring ingestion for %d stream(s): %s", len(sources), redacted_sources)

    # 2. Check for native streams comparison mode
    if args.native_streams:
        run_native_streams(args, sources)
        return

    ids_so_far = set()
    cam_configs: List[CameraConfig] = []
    for idx, uri in enumerate(sources):
        source_id = source_id_for(uri, idx, taken=ids_so_far)
        ids_so_far.add(source_id)
        cam_configs.append(CameraConfig(source_id=source_id, uri=uri))


    multi_cfg = MultiStreamConfig(cameras=cam_configs)
    manager = MultiStreamManager(multi_cfg)

    # 4. Initialize MultiModelPipeline
    pipeline: Optional[MultiModelPipeline] = None
    if args.models_file:
        pipeline = MultiModelPipeline.from_manifest(args.models_file)
        logger.info(
            "Loaded MultiModelPipeline from manifest '%s' with %d model(s): %s",
            args.models_file,
            len(pipeline.list_models()),
            pipeline.list_models(),
        )
    elif args.model_path:
        name = Path(args.model_path).stem
        cfg = InferenceConfig(model_path=args.model_path, model_name=name)
        pipeline = MultiModelPipeline()
        pipeline.register_model(name, InferenceEngine(cfg))
        logger.info("Loaded single-model shortcut pipeline for '%s'", args.model_path)
    elif Path("models.yaml").is_file():
        pipeline = MultiModelPipeline.from_manifest("models.yaml")
        logger.info(
            "Loaded MultiModelPipeline from default 'models.yaml' with %d model(s): %s",
            len(pipeline.list_models()),
            pipeline.list_models(),
        )
    elif Path("models/inferno.pt").is_file():
        cfg = InferenceConfig(model_path="models/inferno.pt", model_name="inferno")
        pipeline = MultiModelPipeline()
        pipeline.register_model("inferno", InferenceEngine(cfg))
        logger.info("Loaded default single-model pipeline for 'models/inferno.pt'")

    print("\n" + "=" * 70)
    print(" INFERNOGUARD PC INGESTION & MULTI-MODEL PIPELINE DEMO")
    print(" Press Ctrl+C at any time to cleanly stop.")
    print("=" * 70 + "\n")

    start_time = time.time()
    last_report_time = time.time()
    total_batches = 0

    try:
        with manager:
            for batch in manager.iter_batches(poll_interval_sec=0.01):
                total_batches += 1
                now = time.time()

                # If models are configured, run batched multi-model inference
                if pipeline and batch:
                    frames = list(batch.values())
                    model_results = pipeline.predict_batch(frames)

                    for model_name, res_list in model_results.items():
                        for res in res_list:
                            if res.detections:
                                for det in res.detections:
                                    logger.info(
                                        "🚨 [%s | %s] Detection: %s (conf: %.2f) at [%.0f, %.0f, %.0f, %.0f] | Latency: %.1fms (e2e: %.1fms)",
                                        model_name,
                                        res.source_id,
                                        det.class_name,
                                        det.confidence,
                                        det.x1,
                                        det.y1,
                                        det.x2,
                                        det.y2,
                                        res.total_latency_ms,
                                        res.end_to_end_ms,
                                    )

                # Periodic diagnostic telemetry report (every 2 seconds)
                if now - last_report_time >= 2.0:
                    last_report_time = now
                    health_report = manager.get_health_report()
                    print("-" * 70)
                    print(f" Elapsed: {now - start_time:.1f}s | Batches yielded: {total_batches}")
                    for sid, health in health_report.items():
                        status_symbol = "🟢" if health.status == StreamStatus.ACTIVE else "🔴"
                        fps_str = f"{health.current_fps:>5.1f}" if health.current_fps is not None else "  n/a"
                        print(
                            f"  {status_symbol} [{sid}] Status: {health.status.value:<12} | "
                            f"FPS: {fps_str} | "
                            f"Frames: {health.total_frames_read:>6} | "
                            f"Dropped: {health.dropped_frames:>4} | "
                            f"Reconn: {health.reconnect_count}"
                        )
                        if health.error_message:
                            print(f"     ⚠️ Error: {health.error_message}")
                    print("-" * 70)

                if args.duration > 0 and (now - start_time) >= args.duration:
                    logger.info("Requested duration of %.1fs reached.", args.duration)
                    break

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Stopping...")
    finally:
        manager.stop_all()
        print("\nShutdown complete. Resources released cleanly.")


if __name__ == "__main__":
    main()
