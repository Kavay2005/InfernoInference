"""Tests for Ultralytics .streams file generator, parser, URI redaction, and source ID derivation."""

import tempfile
import time
from pathlib import Path
import numpy as np
import pytest

from pc_infernoguard.capture.stream_types import FrameData
from pc_infernoguard.utils.streams_file import (
    generate_streams_file,
    parse_streams_file,
    redact_uri,
    source_id_for,
    to_ultralytics_batch,
)


def test_generate_streams_file_round_trip():
    """Verify generate_streams_file output round-trips cleanly through parse_streams_file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        streams_path = Path(tmpdir) / "sources.streams"
        sources = ["0", "rtsp://192.168.1.100:554/ch0", "1"]

        output_file = generate_streams_file(sources, streams_path)
        assert output_file.is_file()

        parsed = parse_streams_file(output_file)
        assert parsed == ["0", "rtsp://192.168.1.100:554/ch0", "1"]


def test_parse_streams_file_bom_clean():
    """Verify a UTF-8 file with BOM parses cleanly without leaving BOM in URLs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        streams_path = Path(tmpdir) / "bom.streams"
        # Write UTF-8 with BOM (\ufeff)
        content = "\ufeff0\nrtsp://192.168.1.100:554/live\n"
        streams_path.write_bytes(content.encode("utf-8"))

        parsed = parse_streams_file(streams_path)
        assert parsed == ["0", "rtsp://192.168.1.100:554/live"]
        assert not parsed[0].startswith("\ufeff")


def test_parse_streams_file_comment_line_raises_value_error():
    """Verify '#' comment line raises ValueError because Ultralytics does not support comments."""
    with tempfile.TemporaryDirectory() as tmpdir:
        streams_path = Path(tmpdir) / "comments.streams"
        streams_path.write_text("0\n# this is a comment line\nrtsp://cam/live\n", encoding="utf-8")

        with pytest.raises(ValueError, match="comments are not supported"):
            parse_streams_file(streams_path)


def test_parse_streams_file_line_with_space_raises_value_error():
    """Verify line containing whitespace raises ValueError because Ultralytics splits on whitespace."""
    with tempfile.TemporaryDirectory() as tmpdir:
        streams_path = Path(tmpdir) / "spaces.streams"
        streams_path.write_text("0\nrtsp://cam1/live extra_param\n", encoding="utf-8")

        with pytest.raises(ValueError, match="expected one URI per line with no whitespace"):
            parse_streams_file(streams_path)


def test_parse_streams_file_empty_file_raises_value_error():
    """Verify empty file or whitespace-only file raises ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        streams_path = Path(tmpdir) / "empty.streams"
        streams_path.write_text("   \n\t  \n", encoding="utf-8")

        with pytest.raises(ValueError, match="no stream sources found"):
            parse_streams_file(streams_path)


def test_parse_streams_file_duplicate_uri_raises_value_error():
    """Verify duplicate URIs raise ValueError to prevent duplicate connections."""
    with tempfile.TemporaryDirectory() as tmpdir:
        streams_path = Path(tmpdir) / "dupes.streams"
        streams_path.write_text("rtsp://cam1/live\nrtsp://cam1/live\n", encoding="utf-8")

        with pytest.raises(ValueError, match="duplicate sources would open duplicate connections"):
            parse_streams_file(streams_path)


def test_redact_uri_masks_credentials_and_query_tokens():
    """Verify redact_uri masks user:password@ (including passwords with @) and query tokens."""
    # Password containing '@' and query string token
    uri_with_at_password = "rtsp://user:p@ssword@host:554/live?token=secret123"
    redacted = redact_uri(uri_with_at_password)
    assert "p@ssword" not in redacted
    assert "user" not in redacted
    assert "secret123" not in redacted
    assert redacted == "rtsp://***@host:554/live?***"

    # Standard RTSP URL
    rtsp_url = "rtsp://admin:super_secret@192.168.1.100:554/live"
    assert redact_uri(rtsp_url) == "rtsp://***@192.168.1.100:554/live"

    # Clean URL without credentials or query
    clean_url = "rtsp://192.168.1.100:554/live"
    assert redact_uri(clean_url) == clean_url

    # Webcams
    assert redact_uri(0) == "0"
    assert redact_uri("1") == "1"


def test_source_id_for_stability_credentials_and_deduplication():
    """Verify source_id_for stability across file order, credential absence, and collision de-duplication."""
    uri_north = "rtsp://user:secret@192.168.1.50:554/live/north_cam"
    uri_south = "rtsp://user:secret@192.168.1.51:554/live/south_cam"

    # 1. Stability independent of file order / sequence index
    id_north_idx0 = source_id_for(uri_north, index=0)
    id_north_idx5 = source_id_for(uri_north, index=5)
    assert id_north_idx0 == id_north_idx5 == "192.168.1.50_554_live_north_cam"

    # 2. Contains no credentials
    assert "secret" not in id_north_idx0
    assert "user" not in id_north_idx0

    # 3. De-duplicates collisions when taken set is passed
    taken = set()
    id1 = source_id_for(uri_north, 0, taken=taken)
    taken.add(id1)
    id2 = source_id_for(uri_north, 1, taken=taken)
    taken.add(id2)
    id3 = source_id_for(uri_north, 2, taken=taken)
    taken.add(id3)

    assert id1 == "192.168.1.50_554_live_north_cam"
    assert id2 == "192.168.1.50_554_live_north_cam_2"
    assert id3 == "192.168.1.50_554_live_north_cam_3"


def test_to_ultralytics_batch_conversion():
    """Verify to_ultralytics_batch extracts keys and image arrays."""
    img1 = np.ones((240, 320, 3), dtype=np.uint8) * 10
    img2 = np.ones((240, 320, 3), dtype=np.uint8) * 20

    batch = {
        "cam_0": FrameData(
            source_id="cam_0",
            frame_id=1,
            timestamp_utc=time.time(),
            timestamp_mono=time.monotonic(),
            image=img1,
            width=320,
            height=240,
        ),
        "cam_1": FrameData(
            source_id="cam_1",
            frame_id=1,
            timestamp_utc=time.time(),
            timestamp_mono=time.monotonic(),
            image=img2,
            width=320,
            height=240,
        ),
    }

    source_ids, images = to_ultralytics_batch(batch)
    assert source_ids == ["cam_0", "cam_1"]
    assert len(images) == 2
    assert images[0].shape == (240, 320, 3)
    assert images[1].shape == (240, 320, 3)
