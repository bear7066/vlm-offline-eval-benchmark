from __future__ import annotations

from pathlib import Path

from vlm_eval.video import VIDEO_EXTENSIONS


def discover_videos(root: Path, limit: int | None = None) -> list[tuple[Path, str]]:
    """Find labeled videos under a directory tree.

    The label for each clip is its parent directory name (the project's
    convention, e.g. ``.../face_planting/clip.mp4`` -> ``"face_planting"``).
    A single video file may also be passed directly, in which case its parent
    directory name is used as the label.

    Args:
        root: Directory to scan recursively, or a single video file.
        limit: Optional cap on the number of videos returned (per call, after
            sorting for determinism).

    Returns:
        A list of ``(video_path, label)`` pairs sorted by path.

    Raises:
        FileNotFoundError: If ``root`` does not exist.
        ValueError: If no videos are found.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Video root does not exist: {root}")

    if root.is_file():
        pairs = [(root, root.parent.name or "unknown")]
    else:
        pairs = [
            (path, path.parent.name or "unknown")
            for path in sorted(root.rglob("*"))
            if path.suffix.lower() in VIDEO_EXTENSIONS
        ]

    if not pairs:
        raise ValueError(f"No videos ({', '.join(VIDEO_EXTENSIONS)}) found under {root}")

    if limit is not None:
        pairs = pairs[:limit]
    return pairs
