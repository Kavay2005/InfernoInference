"""Synthetic video generator fixtures for headless testing."""

import uuid
from pathlib import Path
import tempfile
import cv2
import numpy as np


def create_synthetic_video_file(
    duration_sec: float = 2.0,
    fps: int = 30,
    width: int = 320,
    height: int = 240,
    prefix: str = "synthetic_stream_",
) -> str:
    """Create a temporary MP4 video file with color patterns and frame counter text.
    
    Returns:
        Absolute string path to the created temporary video file.
    """
    temp_dir = tempfile.gettempdir()
    unique_id = uuid.uuid4().hex[:8]
    file_path = str(Path(temp_dir) / f"{prefix}_{unique_id}_{int(duration_sec*1000)}ms.avi")

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(file_path, fourcc, float(fps), (width, height))

    total_frames = int(duration_sec * fps)
    for frame_idx in range(total_frames):
        # Create a dynamic test image: gradient + text
        img = np.zeros((height, width, 3), dtype=np.uint8)
        color_val = (frame_idx * 5) % 255
        img[:, :] = [color_val, 128, 255 - color_val]
        cv2.putText(
            img,
            f"F:{frame_idx}",
            (20, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )
        out.write(img)

    out.release()
    return file_path
