from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class RealtimeResult:
    """One timed inference plus derived real-time metrics.

    Extends the idea of ``vlm_eval.metrics.VideoResult`` with the fields a
    real-time decision needs: a prefill/decode split, a per-frame prefill
    cost, and the real-time factor ``rtf_inv``.

    Attributes:
        video: Path to the source clip.
        label: Ground-truth action label.
        model_id: HuggingFace model ID used.
        num_frames: Frames sampled and fed to the model.
        max_new_tokens: Generation cap for this run.
        repeat_index: Zero-based index of the timed repeat.
        query_latency_ms: Total wall-clock generation time.
        ttft_ms: Time to first token (prefill cost proxy).
        decode_ms: ``query_latency_ms - ttft_ms`` (token generation time).
        prefill_ms_per_frame: ``ttft_ms / num_frames``; predicts latency at
            other frame counts.
        rtf_inv: ``query_latency_sec / video_duration_sec``; <= 1.0 is real time.
        meets_realtime: Whether ``rtf_inv <= 1.0`` for this run.
        video_duration_sec: Source clip duration.
        tokens: Number of generated tokens.
        throughput_tps: Generated tokens per wall-second.
        mean_power_watts: Mean GPU power over the inference.
        peak_power_watts: Peak GPU power over the inference.
        peak_vram_gb: Peak VRAM allocated during the inference.
        response: Model output text.
        correct: Whether the response matched the label (heuristic; may be None).
        status: ``"success"`` or ``"error"``.
        error: Error message when ``status == "error"``.
    """

    video: str
    label: str
    model_id: str
    num_frames: int
    max_new_tokens: int
    repeat_index: int = 0
    query_latency_ms: float | None = None
    ttft_ms: float | None = None
    decode_ms: float | None = None
    prefill_ms_per_frame: float | None = None
    rtf_inv: float | None = None
    meets_realtime: bool | None = None
    video_duration_sec: float | None = None
    tokens: int | None = None
    throughput_tps: float | None = None
    mean_power_watts: float | None = None
    peak_power_watts: float | None = None
    peak_vram_gb: float | None = None
    response: str = ""
    correct: bool | None = None
    status: str = "success"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the result as a plain dict for JSON serialization."""
        return asdict(self)


def percentile(values: list[float], q: float) -> float | None:
    """Compute the ``q``-th percentile via linear interpolation.

    Args:
        values: Sample values; need not be sorted. Empty returns ``None``.
        q: Percentile in ``[0, 100]``.

    Returns:
        The interpolated percentile, or ``None`` for an empty input.
    """
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (q / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * frac


_WORD_RE = re.compile(r"[a-z0-9]+")


def naive_correct(response: str, label: str) -> bool:
    """Heuristic accuracy: does the response share label content words?

    This is a placeholder so the sweep is runnable without an LLM judge.
    It lowercases both strings, drops short stopword-like tokens, and checks
    whether any label content word appears in the response. Replace with the
    ``vlm_eval.judge`` pipeline before trusting the accuracy axis.

    Args:
        response: Model output text.
        label: Ground-truth label (underscores treated as spaces).

    Returns:
        True if any label content word is present in the response.
    """
    resp_words = set(_WORD_RE.findall(response.lower()))
    label_words = {
        word
        for word in _WORD_RE.findall(label.replace("_", " ").lower())
        if len(word) > 2
    }
    if not label_words:
        return False
    return bool(resp_words & label_words)


@dataclass
class ConfigSummary:
    """Aggregated metrics for one ``(model, frames, tokens)`` config."""

    model_id: str
    num_frames: int
    max_new_tokens: int
    n_runs: int
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    max_latency_ms: float | None
    p50_rtf_inv: float | None
    p95_rtf_inv: float | None
    mean_prefill_ms_per_frame: float | None
    mean_decode_ms: float | None
    mean_peak_vram_gb: float | None
    mean_power_watts: float | None
    accuracy: float | None
    meets_realtime_p95: bool | None

    def to_dict(self) -> dict[str, Any]:
        """Return the summary as a plain dict for JSON serialization."""
        return asdict(self)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def aggregate(results: list[RealtimeResult], threshold: float = 0.8) -> list[ConfigSummary]:
    """Collapse per-run results into one summary per config.

    Groups successful results by ``(model_id, num_frames, max_new_tokens)`` and
    computes latency percentiles, the p95 real-time factor, and accuracy.

    Args:
        results: All per-run results from a sweep.
        threshold: p95 ``rtf_inv`` cutoff used to set ``meets_realtime_p95``.

    Returns:
        One :class:`ConfigSummary` per config, ordered by model then frames.
    """
    groups: dict[tuple[str, int, int], list[RealtimeResult]] = {}
    for result in results:
        if result.status != "success":
            continue
        key = (result.model_id, result.num_frames, result.max_new_tokens)
        groups.setdefault(key, []).append(result)

    summaries: list[ConfigSummary] = []
    for (model_id, num_frames, max_new_tokens), items in sorted(groups.items()):
        latencies = [r.query_latency_ms for r in items if r.query_latency_ms is not None]
        rtfs = [r.rtf_inv for r in items if r.rtf_inv is not None]
        correct = [r.correct for r in items if r.correct is not None]
        p95_rtf = percentile(rtfs, 95)
        summaries.append(
            ConfigSummary(
                model_id=model_id,
                num_frames=num_frames,
                max_new_tokens=max_new_tokens,
                n_runs=len(items),
                p50_latency_ms=percentile(latencies, 50),
                p95_latency_ms=percentile(latencies, 95),
                max_latency_ms=max(latencies) if latencies else None,
                p50_rtf_inv=percentile(rtfs, 50),
                p95_rtf_inv=p95_rtf,
                mean_prefill_ms_per_frame=_mean(
                    [r.prefill_ms_per_frame for r in items if r.prefill_ms_per_frame is not None]
                ),
                mean_decode_ms=_mean([r.decode_ms for r in items if r.decode_ms is not None]),
                mean_peak_vram_gb=_mean(
                    [r.peak_vram_gb for r in items if r.peak_vram_gb is not None]
                ),
                mean_power_watts=_mean(
                    [r.mean_power_watts for r in items if r.mean_power_watts is not None]
                ),
                accuracy=(sum(correct) / len(correct)) if correct else None,
                meets_realtime_p95=(p95_rtf <= threshold) if p95_rtf is not None else None,
            )
        )
    return summaries
