# VLM Offline Eval Benchmark

Tooling to evaluate video vision-language models (VLMs) three ways:

| Package | Question it answers |
|---|---|
| `vlm_eval` | How accurate are a model's descriptions? (inference + LLM-as-a-judge + BLEU/ROUGE/CIDEr) |
| `realtime_eval` | What's the largest model that still runs in real time? (latency/throughput/power sweep) |
| `intelligence_eval` | How close are descriptions to the Sora prompt that generated the clip? (semantic similarity) |

Built with [`uv`](https://docs.astral.sh/uv/). Run anything with `uv run <command>`; append `--help` for full flags.

## Setup

```bash
uv sync
```

Put API/HF credentials in a `.env` file (auto-loaded): `HF_TOKEN` for gated models/datasets, `OPENAI_API_KEY` for the LLM judge.

## `vlm_eval` — accuracy benchmark + judge

Run a model over a HuggingFace dataset, then score the predictions.

```bash
# 1. inference: writes predictions to a run dir under ./runs
uv run vlm-benchmark --model_id google/gemma-3-4b-it --dataset default --num_frames 8

# 2. judge the run (LLM-as-a-judge + text metrics)
uv run vlm-judge --run_dir runs/<run_id> --judge_model gpt-4o

# text metrics only, no LLM
uv run vlm-judge --run_dir runs/<run_id> --skip_llm_judge

# benchmark + judge across many datasets/models in one go
uv run vlm-batch --datasets default climbing_ladder --model_ids google/gemma-3-4b-it

# check the GPU/torch install
uv run vlm-gpu-test
```

## `realtime_eval` — real-time capability sweep

Find the largest config that holds real time on your hardware.

```bash
# smoke-test one model on one video
uv run realtime-bench single path/to/video.mp4 -m google/gemma-3-4b-it

# write a starter sweep config, edit it, then run the sweep
uv run realtime-bench config init sweep-conf.json
uv run realtime-bench sweep sweep-conf.json

# summarize results (p95 real-time-factor cutoff)
uv run realtime-bench analyze runs/<run_id> --threshold 0.8
```

The sweep config (`sweep-conf.json`) sets `videos`, `model_ids`, `num_frames_grid`, `max_new_tokens_grid`, `repeats`, etc. Omitted fields use defaults.

### TensorRT-LLM backend (`sweep-rt`)

`sweep-rt` runs the same sweep through TensorRT-LLM and tags `config.json` with `backend: "tensorrt"` (the default `sweep` tags `backend: "python-transformers"`); output is otherwise identical. TensorRT-LLM's prebuilt wheels are ABI-locked to NVIDIA's private NGC PyTorch build, so it can't be pip-installed alongside the base project — it runs inside NVIDIA's TensorRT-LLM container:

```bash
# On GB10 (DGX Spark) use a Spark-specific image; generic release tags lack the
# sm_121 attention cubins and fail with "Unsupported architecture".
export TRT_IMAGE=nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev
scripts/run_trt_container.sh sweep-conf.json
```

The script mounts the repo, installs this project (`--no-deps`), and runs `realtime-bench sweep-rt`; input processing (chat template + image preprocessing) is delegated to TensorRT-LLM's own multimodal loader, so it doesn't depend on the container's transformers version. Browse image tags at <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt-llm>.

Attention backend defaults to `TRT_ATTN_BACKEND=TRTLLM` (the fused fast path; works on the Spark image). Fall back to `FLASHINFER`, or `VANILLA` (pure-torch, slower, arch-independent) on a container that lacks trtllm-gen kernels for your GPU. If the input loader can't infer the model type, set `TRT_MODEL_TYPE` (e.g. `gemma3`).

The image must be new enough for your model on three axes at once: **GB10 attention kernels** (Spark image), the model's architecture in **TensorRT-LLM's modeling**, and the model's architecture in the container's **transformers** (e.g. `google/gemma-4-*` needs `transformers>=5.5.0` — the checkpoint was saved with 5.5.0.dev0). If only the transformers version is behind, patch it in-container without editing the image: `TRT_PIP_INSTALL="transformers>=5.5" scripts/run_trt_container.sh ...`.

## `intelligence_eval` — semantic similarity vs. Sora prompts

Scores a model's video description against the prompt that generated the clip on the [Sora accidents dataset](sora-accidents-dataset.md). Defaults to the HuggingFace dataset `gnitoahc/sora-accidents-copy`.

```bash
# evaluate a model (downloads videos lazily; sample_size caps the count)
uv run intelli-bench -m google/gemma-3-4b-it --sample_size 20 --seed 0

# point at a local dataset dir instead of the HF repo
uv run intelli-bench -m google/gemma-3-4b-it -d ./my_dataset
```

Outputs `predictions.jsonl` and a `summary.json` (mean cosine similarity) under `./intelligence_runs/<run_id>`.

## Layout

```
src/
  vlm_eval/          accuracy benchmark, judge, shared inference/video/paths helpers
  realtime_eval/     real-time sweep + analysis
  intelligence_eval/ semantic-similarity eval on the Sora dataset
```
