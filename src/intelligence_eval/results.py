from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
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
        scores: Per-scorer score keyed by scorer name; a value is ``None`` when
            that scorer failed for this video.
        query_latency_ms: VLM generation latency.
        error: Error message when ``status == "error"``.
    """

    video: str
    label: str
    status: str
    ground_truth_prompt: str | None = None
    response: str | None = None
    scores: dict[str, float | None] = field(default_factory=dict)
    query_latency_ms: float | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def summarize(results: list[EvalResult], scorer_names: list[str]) -> dict:
    """Aggregate per-video results into run-level metrics.

    Args:
        results: Per-video :class:`EvalResult` records.
        scorer_names: Names of the scorers that ran, in report order.

    Returns:
        A dict with counts plus ``mean_<name>`` and ``scored_<name>`` entries
        for each scorer, averaged over the videos that scorer succeeded on.
    """
    summary: dict = {
        "total_videos": len(results),
        "successful_videos": sum(1 for r in results if r.status == "success"),
    }
    for name in scorer_names:
        values = [
            r.scores.get(name)
            for r in results
            if r.status == "success" and r.scores.get(name) is not None
        ]
        summary[f"scored_{name}"] = len(values)
        summary[f"mean_{name}"] = sum(values) / len(values) if values else None
    return summary


def write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def append_jsonl(path: Path, data: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        f.write("\n")
