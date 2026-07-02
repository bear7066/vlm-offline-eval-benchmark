"""TensorRT-LLM backend for the real-time VLM sweep.

This module mirrors the HuggingFace ``sweep`` path but runs generation through
NVIDIA `TensorRT-LLM <https://github.com/NVIDIA/TensorRT-LLM>`_. It is isolated
in its own file because ``tensorrt-llm`` is an optional backend that is
unavailable on most machines: its prebuilt wheels are ABI-locked to NVIDIA's
private NGC PyTorch build, so sweep-rt runs inside NVIDIA's TensorRT-LLM
container (see ``scripts/run_trt_container.sh``), not the project's base venv.

The only thing TensorRT changes is *how a model generates*. Everything else --
frame sampling, timing, power/VRAM measurement, metrics, and the on-disk output
format -- is reused verbatim from :mod:`realtime_eval.pipeline.sweep` via
:func:`run_sweep`, which accepts a model-loader callable and a backend label.

So this file supplies exactly two things:

* :class:`TensorRTLLMVLM` -- a model object exposing the same
  ``generate_from_frames`` contract as
  :class:`vlm_eval.inference.gemma.HuggingFaceVLM`.
* :func:`run_sweep_rt` -- a thin wrapper that hands that loader to
  :func:`run_sweep` with ``backend="tensorrt"``.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from vlm_eval.hardware import get_gpu_power_watts

from realtime_eval.core.config import SweepConfig
from realtime_eval.pipeline.sweep import run_sweep

logger = logging.getLogger(__name__)


def _assemble_generation(
    response: str,
    start_time: float,
    first_token_time: float | None,
    end_time: float,
    num_tokens: int,
    power_watts: float | None,
) -> dict[str, Any]:
    """Pack timed-generation measurements into the ``generate_from_frames`` dict.

    This is the hardware-independent half of the TensorRT path: given the raw
    timestamps and token count, it computes the latency/TTFT/throughput fields
    that :func:`realtime_eval.pipeline.runner.run_config` consumes. Keeping it
    pure makes it testable without a GPU (see ``tests/test_sweep_rt.py``).

    Args:
        response: Decoded model output text.
        start_time: ``time.time()`` captured immediately before generation.
        first_token_time: ``time.time()`` at the first streamed token, or
            ``None`` if nothing was streamed.
        end_time: ``time.time()`` captured immediately after generation.
        num_tokens: Number of generated tokens.
        power_watts: Mean GPU power over the inference, or ``None``.

    Returns:
        A dict with the same keys as
        :meth:`vlm_eval.inference.gemma.HuggingFaceVLM.generate_from_frames`.
    """
    elapsed_sec = end_time - start_time
    ttft_sec = first_token_time - start_time if first_token_time is not None else None
    return {
        "response": response,
        "elapsed_sec": elapsed_sec,
        "elapsed_ms": elapsed_sec * 1000.0,
        "ttft_ms": ttft_sec * 1000.0 if ttft_sec is not None else None,
        "tokens": num_tokens,
        "throughput_tps": num_tokens / elapsed_sec if elapsed_sec > 0 else 0.0,
        "average_power_watts": power_watts,
    }


class TensorRTLLMVLM:
    """A Gemma VLM served by TensorRT-LLM, drop-in for :class:`HuggingFaceVLM`.

    The model is loaded once by TensorRT-LLM's high-level ``LLM`` API. All input
    processing (chat template + image preprocessing) is delegated to TensorRT-LLM
    via :func:`default_multimodal_input_loader`, rather than a raw transformers
    ``AutoProcessor`` -- the container's transformers may be too old to
    instantiate a newer model's processor class, whereas TensorRT-LLM's loader is
    version-matched to the container and knows the model.

    Attributes:
        model_id: HuggingFace model ID being served.
        model_type: HF ``model_type`` (e.g. ``gemma3``) the input loader needs.
        llm: The TensorRT-LLM ``LLM`` instance.
    """

    def __init__(self, model_id: str, hf_token: str | None = None) -> None:
        """Load ``model_id`` with TensorRT-LLM.

        Args:
            model_id: HuggingFace model ID to serve.
            hf_token: Optional HuggingFace access token for gated models.

        Raises:
            RuntimeError: If TensorRT-LLM can't be imported. Its prebuilt wheels
                are ABI-locked to NVIDIA's private NGC PyTorch build, so sweep-rt
                is meant to run inside NVIDIA's TensorRT-LLM container (see
                ``scripts/run_trt_container.sh``), not the project's base venv.
        """
        try:
            from tensorrt_llm import LLM  # type: ignore
        except Exception as exc:  # pragma: no cover - runs only in the TRT container
            raise RuntimeError(
                "Failed to import tensorrt-llm. sweep-rt must run inside NVIDIA's "
                "TensorRT-LLM container (its torch/trt-llm/CUDA/MPI are matched "
                "there); launch it with `scripts/run_trt_container.sh`. "
                f"Original error: {exc}"
            ) from exc

        self.model_id = model_id

        # Attention backend is hardware-sensitive. On GB10 (DGX Spark, sm_121)
        # the TRTLLM backend needs a container built with the Spark's trtllm-gen
        # cubins (e.g. the spark-single-gpu image); the generic release tag lacks
        # them and dies with "Unsupported architecture". TRTLLM is the fast fused
        # path and the default; set TRT_ATTN_BACKEND=FLASHINFER or VANILLA
        # (pure-torch, slow) to fall back. ponytail: GB10 calibration knob.
        attn_backend = os.environ.get("TRT_ATTN_BACKEND", "TRTLLM")
        logger.info("Loading TensorRT-LLM model: %s (attn_backend=%s)", model_id, attn_backend)
        # ponytail: torch backend loads weights each run (no serializable engine
        # to cache like the old TRT-C++ flow); HF weights are cached by hub.
        self.llm = LLM(model=model_id, attn_backend=attn_backend)

        # model_type drives the multimodal input loader. Read it from the HF
        # config (robust) or TRT_MODEL_TYPE if a container's transformers can't
        # even parse the config.
        self.model_type = os.environ.get("TRT_MODEL_TYPE")
        if not self.model_type:
            from transformers import AutoConfig

            self.model_type = AutoConfig.from_pretrained(
                model_id, token=hf_token, trust_remote_code=True
            ).model_type

    def generate_from_frames(
        self,
        frames: list[Any],
        prompt_text: str,
        max_new_tokens: int = 150,
    ) -> dict[str, Any]:
        """Run one multimodal generation, returning the same dict as the HF path.

        Frames + prompt are turned into model inputs by TensorRT-LLM's
        :func:`default_multimodal_input_loader` (it applies the chat template and
        image preprocessing), then run through a blocking ``generate``.

        Args:
            frames: Sampled PIL frames for one clip.
            prompt_text: Instruction text sent with the frames.
            max_new_tokens: Generation cap.

        Returns:
            A dict with keys ``response``, ``elapsed_sec``, ``elapsed_ms``,
            ``ttft_ms``, ``tokens``, ``throughput_tps``, ``average_power_watts``
            -- matching
            :meth:`vlm_eval.inference.gemma.HuggingFaceVLM.generate_from_frames`.
            ``ttft_ms`` is ``None`` here (blocking generate has no first-token
            timestamp).
        """
        from tensorrt_llm import SamplingParams  # type: ignore
        from tensorrt_llm.inputs import default_multimodal_input_loader  # type: ignore

        # The loader inserts image placeholders and preprocesses frames the way
        # this model expects -- one prompt, one list of images.
        inputs = default_multimodal_input_loader(
            tokenizer=self.llm.tokenizer,
            model_dir=self.model_id,
            model_type=self.model_type,
            modality="image",
            prompts=[prompt_text],
            media=[list(frames)],
            image_data_format="pt",
            num_frames=len(frames),
            device="cuda",
        )
        sampling = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)

        start_power = get_gpu_power_watts()
        start_time = time.time()
        outputs = self.llm.generate(inputs, sampling)
        end_time = time.time()
        end_power = get_gpu_power_watts()

        completion = outputs[0].outputs[0]
        # ponytail: blocking generate -> no TTFT; switch to generate_async with
        # streaming=True to recover the prefill/decode split if a run needs it.
        return _assemble_generation(
            response=completion.text.strip(),
            start_time=start_time,
            first_token_time=None,
            end_time=end_time,
            num_tokens=len(completion.token_ids),
            power_watts=_mean_power(start_power, end_power),
        )


def _mean_power(start: float | None, end: float | None) -> float | None:
    """Average two GPU power readings, tolerating missing endpoints.

    Args:
        start: Power before inference, or ``None``.
        end: Power after inference, or ``None``.

    Returns:
        Their mean, whichever single reading exists, or ``None``.
    """
    if start is not None and end is not None:
        return (start + end) / 2.0
    return start if start is not None else end


def load_tensorrt_model(model_id: str, hf_token: str | None = None) -> TensorRTLLMVLM:
    """Load a TensorRT-LLM-served VLM (the loader injected into :func:`run_sweep`).

    Matches the signature of :func:`realtime_eval.pipeline.runner.load_model` so
    it can be passed straight through as ``load_model_fn``.

    Args:
        model_id: HuggingFace model ID.
        hf_token: Optional HuggingFace access token.

    Returns:
        A ready :class:`TensorRTLLMVLM`.
    """
    return TensorRTLLMVLM(model_id, hf_token=hf_token)


def run_sweep_rt(
    videos_root: Path,
    config: SweepConfig,
    video_limit: int | None = None,
) -> Path:
    """Run the real-time sweep with the TensorRT-LLM backend.

    Identical to :func:`run_sweep` in every respect -- same grid, timing, output
    files, and metrics -- except generation runs through TensorRT-LLM and
    ``config.json`` records ``"backend": "tensorrt"``.

    Args:
        videos_root: Directory of labeled videos (or a single video file).
        config: Sweep grid and timing parameters.
        video_limit: Optional cap on number of videos used.

    Returns:
        Path to the created run directory.
    """
    return run_sweep(
        videos_root,
        config,
        video_limit=video_limit,
        backend="tensorrt",
        load_model_fn=load_tensorrt_model,
    )
