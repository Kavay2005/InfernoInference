"""Unit tests for NativeStreamsRunner mapping and stream conversion."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest

from pc_infernoguard.inference.native_streams_runner import NativeStreamsRunner
from pc_infernoguard.utils.streams_file import generate_streams_file


def test_native_streams_runner_mapping_with_two_distinct_sources():
    """Verify that NativeStreamsRunner maps Results to cameras strictly in file order."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        streams_file = tmp_path / "two_cameras.streams"
        # Two distinct video sources in file order
        sources = ["warehouse_cam_a.mp4", "warehouse_cam_b.mp4"]
        generate_streams_file(sources, streams_file)

        runner = NativeStreamsRunner(
            model_path="models/inferno.pt",
            streams_file=streams_file,
            model_name="test_native_model",
        )

        assert runner.num_cameras == 2
        assert runner.source_ids == ["warehouse_cam_a", "warehouse_cam_b"]

        # Mock YOLO model and predict generator
        mock_results = []
        for i in range(4):
            mock_res = MagicMock()
            mock_res.orig_shape = (240, 320)
            mock_res.boxes = None  # no detections for simple test
            mock_res.speed = {"preprocess": 1.5, "inference": 12.0, "postprocess": 0.5}
            mock_res.device = "cpu"
            mock_results.append(mock_res)

        mock_model = MagicMock()
        mock_model.predict.return_value = iter(mock_results)
        runner._model = mock_model

        # Collect streamed results
        streamed = list(runner.stream(max_batches=4))
        assert len(streamed) == 4

        # Verify strict round-robin order mapping
        assert streamed[0].source_id == "warehouse_cam_a"
        assert streamed[0].frame_id == 1
        assert streamed[1].source_id == "warehouse_cam_b"
        assert streamed[1].frame_id == 2
        assert streamed[2].source_id == "warehouse_cam_a"
        assert streamed[2].frame_id == 3
        assert streamed[3].source_id == "warehouse_cam_b"
        assert streamed[3].frame_id == 4

        # Verify latency components
        for res in streamed:
            assert res.preprocess_ms == 1.5
            assert res.inference_ms == 12.0
            assert res.postprocess_ms == 0.5
            assert res.total_latency_ms == 14.0
            assert res.device_used == "cpu"
