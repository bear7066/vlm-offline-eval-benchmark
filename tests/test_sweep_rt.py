"""Assert-based checks for the TensorRT-LLM sweep path (no GPU needed).

Run directly: ``uv run python tests/test_sweep_rt.py``. Covers the two pieces
that are testable without TensorRT or video files: the pure timing-math helper
and the ``backend`` field wiring in ``config.json``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from realtime_eval.pipeline.sweep_rt import _assemble_generation, run_sweep_rt  # noqa: E402


def test_assemble_generation() -> None:
    """Timestamps and token count map onto the generate_from_frames contract."""
    out = _assemble_generation(
        response="a crash happens",
        start_time=100.0,
        first_token_time=100.2,
        end_time=101.0,
        num_tokens=4,
        power_watts=50.0,
    )
    assert out["response"] == "a crash happens"
    assert out["elapsed_sec"] == 1.0
    assert out["elapsed_ms"] == 1000.0
    assert abs(out["ttft_ms"] - 200.0) < 1e-6
    assert out["tokens"] == 4
    assert abs(out["throughput_tps"] - 4.0) < 1e-6
    assert out["average_power_watts"] == 50.0
    # Same keys as HuggingFaceVLM.generate_from_frames.
    assert set(out) == {
        "response",
        "elapsed_sec",
        "elapsed_ms",
        "ttft_ms",
        "tokens",
        "throughput_tps",
        "average_power_watts",
    }


def test_assemble_generation_no_tokens() -> None:
    """Empty stream: TTFT is None and throughput is 0, never divides by zero."""
    out = _assemble_generation("", 5.0, None, 5.0, 0, None)
    assert out["ttft_ms"] is None
    assert out["throughput_tps"] == 0.0
    assert out["average_power_watts"] is None


def test_run_sweep_rt_tags_backend(tmp_path: Path | None = None) -> None:
    """run_sweep_rt routes through run_sweep with backend='tensorrt'.

    A fake model + monkeypatched loader avoids needing TensorRT or real videos,
    proving the config.json backend field and output format are reused intact.
    """
    import json

    from realtime_eval.core import metrics
    from realtime_eval.pipeline import sweep as sweep_mod
    from realtime_eval.core.config import SweepConfig

    out_root = Path(
        tmp_path or Path(__file__).resolve().parent / "_tmp_run"
    )

    class _FakeModel:
        def generate_from_frames(self, frames, prompt_text, max_new_tokens=150):
            return {
                "response": "everything is normal",
                "elapsed_sec": 0.5,
                "elapsed_ms": 500.0,
                "ttft_ms": 100.0,
                "tokens": 3,
                "throughput_tps": 6.0,
                "average_power_watts": 40.0,
            }

    # Stub the two external touch-points: video discovery and frame sampling.
    orig_discover = sweep_mod.discover_videos
    orig_run_config_sample = __import__(
        "realtime_eval.pipeline.runner", fromlist=["_sample_cache"]
    )
    fake_video = Path("clip/normal.mp4")
    sweep_mod.discover_videos = lambda root, limit=None: [(fake_video, "normal")]
    orig_run_config_sample._sample_cache = lambda videos, num_frames: {
        fake_video: ([object()], 2.0)
    }

    try:
        from realtime_eval.pipeline import sweep_rt
        sweep_rt.load_tensorrt_model = lambda model_id, hf_token=None: _FakeModel()

        cfg = SweepConfig(
            model_ids=("fake/model",),
            num_frames_grid=(4,),
            max_new_tokens_grid=(20,),
            repeats=1,
            warmup=0,
            output_root=out_root,
        )
        run_dir = run_sweep_rt(Path("videos"), cfg, video_limit=1)
        config_json = json.loads((run_dir / "config.json").read_text())
        assert config_json["backend"] == "tensorrt", config_json
        assert (run_dir / "results.jsonl").exists()
        assert (run_dir / "summary.json").exists()
    finally:
        sweep_mod.discover_videos = orig_discover
        # _sample_cache restored implicitly by reimport on next run; tmp cleanup:
        import shutil

        if out_root.exists():
            shutil.rmtree(out_root, ignore_errors=True)


if __name__ == "__main__":
    test_assemble_generation()
    test_assemble_generation_no_tokens()
    test_run_sweep_rt_tags_backend()
    print("ok: sweep_rt checks passed")
