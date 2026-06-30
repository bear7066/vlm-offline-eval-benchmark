from __future__ import annotations

import logging
from pathlib import Path


VIDEO_EXTENSIONS = (".mp4", ".mkv")


def get_video_duration(total_frames: int, fps: float | None) -> float | None:
    """Compute video duration in seconds from a frame count and FPS.

    Args:
        total_frames: Number of frames in the video.
        fps: Average frames per second, or ``None``/0 if unavailable.

    Returns:
        Duration in seconds, or ``None`` if FPS is unavailable or zero.
    """
    if fps and fps > 0:
        return total_frames / fps
    return None


def sample_frames(video_path: Path, num_frames: int = 8):
    """Sample frames evenly from a video file.

    Uses OpenCV for I/O and numpy to compute uniformly spaced indices
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
        - **total_frames** (int | None): Frame count reported by OpenCV.
        - **original_fps** (float | None): Average FPS reported by OpenCV.
    """
    import cv2
    import numpy as np
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.error("Could not read video %s", video_path)
        cap.release()
        return None, None, None, None

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS) or None
        video_duration_sec = get_video_duration(total_frames, original_fps)

        if total_frames == 0:
            return None, None, None, None

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

        # Seek to each index and decode. OpenCV reads BGR, so convert to RGB
        # before handing PIL images to the model processors.
        pil_frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                logging.warning("Failed to read frame %s from %s", idx, video_path)
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frames.append(Image.fromarray(rgb))

        if not pil_frames:
            return None, None, None, None
    finally:
        cap.release()

    return pil_frames, video_duration_sec, total_frames, original_fps


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sample frames from a video and save them as images for inspection."
    )
    parser.add_argument("video_path", type=Path, help="Path to the video file.")
    parser.add_argument("--num_frames", "-n", type=int, default=8, help="Number of frames to sample.")
    parser.add_argument(
        "--output_dir",
        "-o",
        type=Path,
        default=None,
        help="Directory to write sampled frames to (default: <video_stem>_frames next to the video).",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.video_path.with_name(f"{args.video_path.stem}_frames")
    output_dir.mkdir(parents=True, exist_ok=True)

    pil_frames, duration_sec, total_frames, fps = sample_frames(args.video_path, args.num_frames)
    if pil_frames is None:
        raise SystemExit(f"Failed to sample frames from {args.video_path}")

    print(f"video: {args.video_path}")
    print(f"total_frames: {total_frames}, fps: {fps}, duration_sec: {duration_sec}")
    print(f"sampled {len(pil_frames)} frame(s) -> {output_dir}")

    for i, frame in enumerate(pil_frames):
        frame_path = output_dir / f"frame_{i:03d}.png"
        frame.save(frame_path)
        print(f"  saved {frame_path}")
