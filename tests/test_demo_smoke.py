"""Smoke tests for CLI entry point main_capture_demo."""

import subprocess
import sys
from pathlib import Path
import pytest

from tests.synthetic_stream_fixture import create_synthetic_video_file


@pytest.fixture
def synthetic_video():
    path = create_synthetic_video_file(duration_sec=4.0, fps=25, width=320, height=240)
    yield path
    try:
        Path(path).unlink(missing_ok=True)
    except Exception:
        pass


def test_demo_smoke_with_duration_on_local_video(synthetic_video):
    """Smoke test running main_capture_demo with --duration 3 on a local video file."""
    cmd = [
        sys.executable,
        "-m",
        "pc_infernoguard.main_capture_demo",
        "--source",
        str(synthetic_video),
        "--duration",
        "3",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    assert proc.returncode == 0, f"Demo process failed with code {proc.returncode}. Stderr: {proc.stderr}"
    assert "Requested duration of 3.0s reached" in proc.stderr or "Requested duration of 3.0s reached" in proc.stdout
    assert "Shutdown complete" in proc.stdout


def test_demo_error_when_both_source_and_streams_file_provided():
    """Verify CLI immediately errors if both --source and --streams-file are given."""
    cmd = [
        sys.executable,
        "-m",
        "pc_infernoguard.main_capture_demo",
        "--source",
        "0",
        "--streams-file",
        "cameras.streams",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    assert proc.returncode != 0
    assert "Cannot provide both --source and --streams-file" in proc.stderr
