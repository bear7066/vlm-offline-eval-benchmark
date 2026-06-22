from __future__ import annotations

import json
from pathlib import Path

from vlm_eval.paths import model_name_from_id

from realtime_eval.core.metrics import ConfigSummary, RealtimeResult, aggregate


def load_results(run_dir: Path) -> list[RealtimeResult]:
    """Load per-run results from a sweep's ``results.jsonl``.

    Args:
        run_dir: Sweep run directory containing ``results.jsonl``.

    Returns:
        The list of :class:`RealtimeResult` records.

    Raises:
        FileNotFoundError: If ``results.jsonl`` is missing.
    """
    results_path = Path(run_dir) / "results.jsonl"
    if not results_path.exists():
        raise FileNotFoundError(f"No results.jsonl in {run_dir}")

    results: list[RealtimeResult] = []
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(RealtimeResult(**json.loads(line)))
    return results


def _fmt(value: float | None, spec: str = ".2f") -> str:
    return format(value, spec) if value is not None else "-"


def format_table(summaries: list[ConfigSummary]) -> str:
    """Render per-config summaries as a fixed-width text table.

    Args:
        summaries: Aggregated config summaries.

    Returns:
        A multi-line string sorted by model, then frame count.
    """
    header = (
        f"{'model':<18}{'frames':>7}{'tok':>5}{'p50_lat_ms':>12}"
        f"{'p95_lat_ms':>12}{'p95_rtf':>9}{'ms/frame':>10}{'acc':>7}{'RT?':>5}"
    )
    lines = [header, "-" * len(header)]
    for s in sorted(summaries, key=lambda x: (x.model_id, x.num_frames, x.max_new_tokens)):
        rt = "yes" if s.meets_realtime_p95 else "no"
        lines.append(
            f"{model_name_from_id(s.model_id):<18}"
            f"{s.num_frames:>7}{s.max_new_tokens:>5}"
            f"{_fmt(s.p50_latency_ms, '.0f'):>12}"
            f"{_fmt(s.p95_latency_ms, '.0f'):>12}"
            f"{_fmt(s.p95_rtf_inv):>9}"
            f"{_fmt(s.mean_prefill_ms_per_frame, '.0f'):>10}"
            f"{_fmt(s.accuracy):>7}"
            f"{rt:>5}"
        )
    return "\n".join(lines)


def best_config(summaries: list[ConfigSummary]) -> ConfigSummary | None:
    """Pick the highest-accuracy config that meets the real-time threshold.

    Among configs with ``meets_realtime_p95`` True, returns the one with the
    highest accuracy, breaking ties toward more frames (more capacity) then
    lower p95 latency.

    Args:
        summaries: Aggregated config summaries.

    Returns:
        The recommended :class:`ConfigSummary`, or ``None`` if none are real time.
    """
    realtime = [s for s in summaries if s.meets_realtime_p95]
    if not realtime:
        return None
    return max(
        realtime,
        key=lambda s: (
            s.accuracy if s.accuracy is not None else -1.0,
            s.num_frames,
            -(s.p95_latency_ms or 0.0),
        ),
    )


def analyze(run_dir: Path, threshold: float = 0.8) -> str:
    """Build a human-readable analysis report for a sweep run.

    Args:
        run_dir: Sweep run directory.
        threshold: p95 ``rtf_inv`` cutoff for the real-time decision.

    Returns:
        A report string with the per-config table and the recommended pick.
    """
    results = load_results(run_dir)
    summaries = aggregate(results, threshold=threshold)
    table = format_table(summaries)

    pick = best_config(summaries)
    if pick is None:
        verdict = (
            f"\nNo config meets p95 rtf_inv <= {threshold}. "
            "Reduce frames/resolution or consider a faster runtime (vLLM/quantization)."
        )
    else:
        verdict = (
            f"\nRecommended: {model_name_from_id(pick.model_id)} | "
            f"{pick.num_frames} frames | max_new_tokens={pick.max_new_tokens}\n"
            f"  p95 rtf_inv={_fmt(pick.p95_rtf_inv)} (<= {threshold}), "
            f"accuracy={_fmt(pick.accuracy)}, p95 latency={_fmt(pick.p95_latency_ms, '.0f')} ms"
        )
    return f"{table}\n{verdict}"
