# Real-Time VLM Evaluation (`realtime_eval`)

Goal: find the **sweet spot** — the largest-capacity fine-tuned VLM config that still
runs in **real time** on the deployment GPU (V100), where real time means
*processing 1 second of video takes ≤ 1 second of wall-clock time*.

This module does **not** edit `src/vlm_eval`. It imports the stable, pure helpers
from there and re-implements the timing loop with the changes needed for a
real-time decision.

---

## 0. The metric

The project already exposes the right idea in `vlm_eval.metrics.summarize_results`
as `equivalent_real_time_latency`. We keep the same `latency / duration` direction
and name it clearly:

```
rtf_inv       = query_latency_sec / video_duration_sec   # wall-seconds per video-second
meets_realtime = rtf_inv <= 1.0                           # 1s of video in <= 1s
```

- Example baseline run: `4.53 / 4.0 = 1.13` → **not** real time.
- Deployment target: **p95 `rtf_inv` <= 0.8** (headroom for tail spikes).

Key empirical fact from the baseline: **prefill dominates** — `ttft_ms` was 82% of
total latency. Latency scales ~linearly with `num_frames` (image tokens), so the
frame count is the primary lever and `prefill_ms_per_frame` is the number that lets
us predict latency at any frame count without re-running.

---

## 1. Benchmark improvements over `vlm_eval`

| # | Change | Why the current benchmark is insufficient |
|---|--------|-------------------------------------------|
| 1 | **Warmup** iterations (discarded) | `inspect.py` times a single cold run incl. CUDA autotune. |
| 2 | **Repeats + percentiles** (p50/p95/max) | Real time is a tail-latency property; a single mean hides spikes. |
| 3 | **Prefill/decode split** (`decode_ms`, `prefill_ms_per_frame`) | Lets us predict latency vs. frame count; isolates the dominant cost. |
| 4 | **`rtf_inv` + `meets_realtime`** as first-class fields | Makes the benchmark answer the question directly. |
| 5 | **Deployment-matched `max_new_tokens`** (sweep axis) | Default 150 inflates decode vs. a one-sentence production cap. |
| 6 | **Continuous power sampling** (background thread) | `average_power_watts` was only a 2-sample start/end average. |
| 7 | **Labeled video set**, accuracy aggregated across it | A single clip cannot support a model-selection decision. |

---

## 2. Model / config search

Treat selection as a **Pareto search: accuracy vs. `rtf_inv`** — not "pick a model".

**Grid** (`SweepConfig`):
- `model_ids`: `bear7011/gemma4-e2b-webvid4K_FT`, `bear7011/gemma4-e4b-webvid4K_FT`
- `num_frames_grid`: `4, 8, 12, 16` (primary lever)
- `max_new_tokens_grid`: `20, 40`

**Flow:** `sweep.py` loads each model once → iterates its frame/token configs over the
labeled set → `results.jsonl`. `analyze.py` collapses to per-config p95 `rtf_inv` +
accuracy and prints the Pareto-relevant table.

**Decision rule:** pick the **highest-accuracy config with p95 `rtf_inv` <= 0.8**.
Expect E2B at 4–8 frames to clear the bar; the sweep's real output is how much
accuracy is forfeited shrinking 16→8→4, and whether E4B ever fits on V100.

---

## 3. V100 notes (bench GPU == deploy GPU)

- Stick to the E2B/E4B fine-tunes. Project memory notes `gemma-3-4b` **hangs on V100**
  and has a double-BOS issue — excluded from the sweep.
- `HuggingFaceVLM` already runs bf16; V100 bf16 throughput is weak, so prefill cost is
  real. If nothing clears 0.8, the next step (separate spike, deferred) is a runtime
  swap (vLLM / quantization).

---

## 4. Layout

```
src/realtime_eval/
  PLAN.md            # this file
  __init__.py
  __main__.py        # CLI: `uv run python -m realtime_eval sweep|analyze`
  core/              # reusable building blocks (no orchestration)
    __init__.py
    config.py        # SweepConfig + default prompt
    metrics.py       # RealtimeResult dataclass + percentile aggregation
    power.py         # PowerSampler: background NVML power thread
    dataset.py       # discover labeled videos -> [(path, label)]
  pipeline/          # orchestration (depends on core)
    __init__.py
    runner.py        # warm, repeated, timed inference loop (one config)
    sweep.py         # cartesian grid -> results.jsonl + summary.json
    analyze.py       # load results -> per-config p95 table + best pick
```

**Reused from `vlm_eval` (imported, never edited):**
`video.sample_frames`, `hardware.{get_gpu_power_watts,get_peak_vram_gb,reset_peak_memory_stats,get_hardware_name}`,
`inference.gemma.HuggingFaceVLM`, `paths.{slugify,model_name_from_id}`.

**Re-implemented here:** the timing loop (warmup + repeats + power thread + percentile
aggregation) and an extended result dataclass — the originals load the model per call,
time a single cold run, and use a fixed `VideoResult` we cannot edit.

---

## 5. Status / next steps

Accuracy is currently a **naive token-overlap heuristic** (`metrics.naive_correct`) so
the sweep is runnable end-to-end without an LLM judge. Replace with the real
`vlm_eval.judge` pipeline (or a held-out labeled benchmark) before trusting the
accuracy axis for the final pick.

Run:
```
uv run python -m realtime_eval config init <config.json>   # write a default config to edit
uv run python -m realtime_eval sweep   <config.json>
uv run python -m realtime_eval analyze <run_dir>
```
The sweep is fully described by a JSON config (`config init` writes the defaults;
see also `configs/sweep.example.json`): a required `videos` path, an optional
`limit`, and any `SweepConfig` field (omitted fields keep their defaults).
Unknown keys fail loudly to catch typos.
If `realtime_eval` is not importable, re-register the new package: `uv sync`.
