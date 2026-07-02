#!/usr/bin/env bash
# Run `realtime-bench sweep-rt` inside NVIDIA's TensorRT-LLM container.
#
# TensorRT-LLM's prebuilt wheels are ABI-locked to NVIDIA's private NGC PyTorch
# build, and no public pip torch wheel matches it (verified: public 2.9.x and
# 2.10.x each miss a symbol trt-llm needs). So the TRT backend runs inside
# NVIDIA's container -- which ships the matched torch + trt-llm + CUDA + MPI --
# rather than a local venv. The base `realtime-bench sweep` still runs normally
# in the project's own .venv; only sweep-rt needs this.
#
# On GB10 (DGX Spark) use a Spark-specific TensorRT-LLM image -- the generic
# `release:<ver>` tags lack the trtllm-gen attention cubins for sm_121 and fail
# with "Unsupported architecture". Use the Spark image NVIDIA ships, e.g.
# `nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev`.
#
# Usage:
#   export TRT_IMAGE=nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev
#   scripts/run_trt_container.sh sweep-conf.json
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${TRT_IMAGE:-}" ]; then
  echo "Set TRT_IMAGE to an NVIDIA TensorRT-LLM release image for your platform:" >&2
  echo "  export TRT_IMAGE=nvcr.io/nvidia/tensorrt-llm/release:<tag>" >&2
  echo "Browse tags: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt-llm" >&2
  exit 1
fi

CONFIG="${1:-sweep-conf.json}"

# Attention backend. TRTLLM is the fused fast path and works on the Spark image
# (which ships GB10 trtllm-gen cubins). Fall back to FLASHINFER, or VANILLA
# (pure-torch, slow) if a container lacks kernels for your arch.
ATTN="${TRT_ATTN_BACKEND:-TRTLLM}"

# Install our code WITHOUT deps (the image already has torch/trt-llm/transformers/
# opencv); only python-dotenv is missing. TRT_PIP_INSTALL injects extra pip
# packages into the container before the run -- use it when the image's
# transformers predates your model's architecture, e.g.
#   TRT_PIP_INSTALL="transformers>=5.5"   # google/gemma-4-* needs transformers>=5.5.0
# (only works if the image's TensorRT-LLM also has that model's support).
INNER="pip install --no-deps -e . && pip install -q python-dotenv"
if [ -n "${TRT_PIP_INSTALL:-}" ]; then
  INNER="$INNER && pip install -q ${TRT_PIP_INSTALL}"
fi
INNER="$INNER && realtime-bench sweep-rt '$CONFIG'"

# Flags (--ipc=host, memlock/stack ulimits) follow NVIDIA's TRT-LLM container
# guidance. HF_TOKEN is forwarded for gated Gemma models.
docker run --gpus all --rm -it \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$PWD:/work" -w /work \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e TRT_ATTN_BACKEND="$ATTN" \
  "$TRT_IMAGE" \
  bash -lc "$INNER"
