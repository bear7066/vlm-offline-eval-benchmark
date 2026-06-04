from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

HF_REPO_ID = "gnitoahc/vlm-eval-videos"


def model_name_from_id(model_id: str) -> str:
    """Extract a short model name from a HuggingFace model ID.

    Strips the org prefix and the ``-it`` instruction-tuned suffix so
    ``"meta-llama/Llama-3-8B-it"`` becomes ``"Llama-3-8B"``.

    Args:
        model_id: Full HuggingFace model ID, e.g. ``"org/model-name-it"``.

    Returns:
        The bare model name without org prefix or ``-it`` suffix.
    """
    return model_id.split("/")[-1].replace("-it", "")


def label_from_video_dir(video_dir: Path) -> str:
    """Derive a label string from the name of a video directory.

    Retained for legacy-log parsing only.

    Args:
        video_dir: Path whose final component is used as the label.

    Returns:
        The directory name, or ``"default_ground_truth"`` when the name
        is empty or ``"."``.
    """
    name = Path(video_dir).resolve().name
    return name if name and name != "." else "default_ground_truth"


def display_label_from_video_path(video_path: str | Path) -> str:
    """Return a human-readable action label derived from a video file path.

    Assumes the video's parent directory name encodes the action class,
    e.g. ``".../push_up/clip.mp4"`` → ``"push up"``.

    Args:
        video_path: Path to the video file.

    Returns:
        Parent directory name with underscores replaced by spaces, or
        ``"Unknown Action"`` when the parent cannot be determined.
    """
    label = Path(video_path).parent.name
    if not label or label in {".", ".."}:
        return "Unknown Action"
    return label.replace("_", " ")


def slugify(value: str) -> str:
    """Convert an arbitrary string into a filesystem-safe slug.

    Replaces any run of characters outside ``[A-Za-z0-9._-]`` with a
    single underscore, then strips leading/trailing punctuation.

    Args:
        value: Raw string to slugify.

    Returns:
        A non-empty slug string; falls back to ``"run"`` if the result
        would otherwise be empty.
    """
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._-")
    return value or "run"


def build_run_id(
    model_id: str,
    dataset: str,
    num_frames: int,
    now: datetime | None = None,
) -> str:
    """Build a unique, human-readable run identifier.

    Combines the short model name, frame count, dataset label, and a
    timestamp so output directories sort chronologically and are easy to
    identify at a glance.

    Args:
        model_id: Full HuggingFace model ID.
        dataset: Dataset name or path label.
        num_frames: Number of frames sampled per video.
        now: Timestamp to embed; defaults to ``datetime.now()``.

    Returns:
        A slug of the form ``"<model>_<N>frames_<dataset>_<YYYYMMDD-HHMMSS>"``.
    """
    timestamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    model_name = slugify(model_name_from_id(model_id))
    label = slugify(dataset)
    return f"{model_name}_{num_frames}frames_{label}_{timestamp}"


def ensure_run_dir(output_root: Path, run_id: str) -> Path:
    """Create and return the output directory for a run.

    Args:
        output_root: Parent directory under which the run directory is created.
        run_id: Raw run identifier; passed through :func:`slugify` before use.

    Returns:
        Path to the newly created run directory.

    Raises:
        FileExistsError: If the directory already exists (``exist_ok=False``).
    """
    run_dir = output_root / slugify(run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def find_latest_run(
    output_root: Path,
    model_id: str | None = None,
    dataset: str | None = None,
    sample_fps: float | None = None,
    num_frames: int | None = None,
) -> Path | None:
    """Find the most recently modified run directory matching the given filters.

    Scans ``output_root`` for subdirectories containing a ``config.json``
    and returns the one with the latest modification time among those that
    satisfy all supplied filter criteria.

    Args:
        output_root: Root directory containing run subdirectories.
        model_id: If given, only consider runs whose ``config.json`` has a
            matching ``model_id`` field.
        dataset: If given, only consider runs whose ``config.json`` has a
            matching ``ground_truth_name`` field.
        sample_fps: If given, only consider runs whose ``config.json`` has a
            matching ``sample_fps`` field.
        num_frames: If given, only consider runs whose ``config.json`` has a
            matching ``num_frames`` field.

    Returns:
        Path to the matching run directory with the latest mtime, or ``None``
        if no matching run is found or ``output_root`` does not exist.
    """
    if not output_root.exists():
        return None

    candidates: list[Path] = []

    for config_path in output_root.glob("*/config.json"):
        try:
            with config_path.open("r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception:
            continue

        if model_id is not None and config.get("model_id") != model_id:
            continue
        if dataset is not None and config.get("ground_truth_name") != dataset:
            continue
        if sample_fps is not None and float(config.get("sample_fps", -1)) != sample_fps:
            continue
        if num_frames is not None and int(config.get("num_frames", -1)) != num_frames:
            continue

        candidates.append(config_path.parent)

    if not candidates:
        return None

    return max(candidates, key=lambda path: path.stat().st_mtime)
