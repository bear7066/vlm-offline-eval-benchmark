from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


VIDEO_EXTENSIONS = (".mp4", ".mkv")


def get_video_duration(video_reader: Any) -> float | None:
    """Compute video duration in seconds from a decord VideoReader.

    Args:
        video_reader: A decord VideoReader (or any object exposing
            ``get_avg_fps()`` and ``__len__``).

    Returns:
        Duration in seconds, or ``None`` if FPS is unavailable or zero.
    """
    try:
        fps = video_reader.get_avg_fps()
        total_frames = len(video_reader)
        if fps and fps > 0:
            return total_frames / fps
    except Exception:
        pass
    return None


def sample_frames(video_path: Path, num_frames: int = 8):
    """Sample frames evenly from a video file.

    Uses decord for I/O and numpy to compute uniformly spaced indices
    across the full duration of the video.

    Args:
        video_path: Path to the video file.
        num_frames: Number of frames to sample. Must be > 0.

    Returns:
        A 4-tuple of ``(pil_frames, video_duration_sec, total_frames,
        original_fps)``.  All elements are ``None`` when the video
        cannot be opened, contains no frames, or *num_frames* is invalid.

        - **pil_frames** (list[PIL.Image.Image] | None): Sampled frames
          as PIL images.
        - **video_duration_sec** (float | None): Total duration in seconds.
        - **total_frames** (int | None): Frame count reported by decord.
        - **original_fps** (float | None): Average FPS reported by decord.
    """
    import decord
    import numpy as np
    from PIL import Image

    try:
        video_reader = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
    except Exception as exc:
        logging.error("Could not read video %s: %s", video_path, exc)
        return None, None, None, None

    total_frames = len(video_reader)
    if total_frames == 0:
        return None, None, None, None

    try:
        original_fps = video_reader.get_avg_fps()
    except Exception:
        original_fps = None

    video_duration_sec = get_video_duration(video_reader)

    if num_frames <= 0:
        logging.error("num_frames must be > 0, got %s", num_frames)
        return None, None, None, None

    # Pick num_frames positions spread evenly from frame 0 to the last frame.
    # e.g. a 100-frame video with num_frames=4 → [0, 33, 66, 99]
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    # Guard against floating-point rounding pushing an index out of bounds.
    indices = np.clip(indices, 0, total_frames - 1)
    # When num_frames > total_frames, linspace produces duplicates; drop them
    # so we never decode the same frame twice.
    indices = np.unique(indices)

    # Shouldn't happen with valid inputs, but fall back to the first frame
    # rather than crashing.
    if len(indices) == 0:
        indices = np.array([0], dtype=int)

    # Decode all selected frames in one batched read (more efficient than
    # calling get_batch per frame), then convert to PIL for model processors.
    frames = video_reader.get_batch(indices).asnumpy()
    pil_frames = [Image.fromarray(frame) for frame in frames]
    return pil_frames, video_duration_sec, total_frames, original_fps
