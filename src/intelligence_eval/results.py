from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class EvalResult:
    """One video's evaluation outcome.

    Attributes:
        video: Path to the evaluated video.
        label: Action label.
        status: ``"success"`` or ``"error"``.
        ground_truth_prompt: The Sora prompt scored against.
        response: The VLM's description.
        similarity: Cosine similarity in ``[-1, 1]``.
        query_latency_ms: VLM generation latency.
        error: Error message when ``status == "error"``.
    """

    video: str
    label: str
    status: str
    ground_truth_prompt: str | None = None
    response: str | None = None
    similarity: float | None = None
    query_latency_ms: float | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def summarize(results: list[EvalResult]) -> dict:
    """Aggregate per-video results into run-level metrics.

    Args:
        results: Per-video :class:`EvalResult` records.

    Returns:
        A dict with counts and the mean similarity over successful videos.
    """
    scored = [r.similarity for r in results if r.status == "success" and r.similarity is not None]
    return {
        "total_videos": len(results),
        "successful_videos": sum(1 for r in results if r.status == "success"),
        "scored_videos": len(scored),
        "mean_similarity": sum(scored) / len(scored) if scored else None,
    }


def write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def append_jsonl(path: Path, data: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        f.write("\n")
