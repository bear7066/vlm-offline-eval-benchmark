"""CLI for the intelligence (semantic-similarity) evaluation.

Run ``uv run python -m intelligence_eval --help`` for usage.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from intelligence_eval.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_PROMPT,
    IntelligenceConfig,
)
from intelligence_eval.dataset import DEFAULT_DATASET
from intelligence_eval.runner import run_intelligence_eval


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m intelligence_eval``."""
    parser = argparse.ArgumentParser(
        prog="intelligence_eval",
        description="Score a VLM's video descriptions against the Sora prompts "
        "that generated the videos, using semantic similarity.",
    )
    parser.add_argument(
        "--dataset", "-d", default=DEFAULT_DATASET,
        help=f"HuggingFace dataset repo id or local dir (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument("--model_id", "-m", required=True, help="HuggingFace VLM ID to evaluate.")
    parser.add_argument("--prompt", "-p", default=DEFAULT_PROMPT, help="Prompt sent to the VLM.")
    parser.add_argument("--num_frames", "-n", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=150)
    parser.add_argument("--sample_size", type=int, default=None, help="Cap on videos (default: all).")
    parser.add_argument("--seed", type=int, default=None, help="Shuffle seed before capping.")
    parser.add_argument("--embedding_model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--output_root", type=Path, default=Path("intelligence_runs"))
    parser.add_argument("--run_id", default=None)
    args = parser.parse_args(argv)

    config = IntelligenceConfig(
        dataset=args.dataset,
        model_id=args.model_id,
        prompt=args.prompt,
        num_frames=args.num_frames,
        max_new_tokens=args.max_new_tokens,
        sample_size=args.sample_size,
        seed=args.seed,
        embedding_model=args.embedding_model,
        output_root=args.output_root,
        run_id=args.run_id,
    )
    return 0 if run_intelligence_eval(config) is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
