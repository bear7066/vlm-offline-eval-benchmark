from __future__ import annotations

import logging
import os
import random
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from vlm_eval.inference.gemma import HuggingFaceVLM
from vlm_eval.logging_utils import configure_logging, quiet_third_party_loggers
from vlm_eval.paths import build_run_id, ensure_run_dir, slugify
from vlm_eval.video import sample_frames

from intelligence_eval.config import IntelligenceConfig
from intelligence_eval.dataset import DatasetItem, DatasetSource, MetadataRow
from intelligence_eval.results import EvalResult, append_jsonl, summarize, write_json
from intelligence_eval.scoring import (
    BERTScore,
    LLMJudge,
    NLIEntailment,
    Scorer,
    SemanticSimilarity,
)

logger = logging.getLogger(__name__)


def _select_rows(rows: list[MetadataRow], config: IntelligenceConfig) -> list[MetadataRow]:
    """Optionally shuffle and cap the rows before any videos are downloaded."""
    if config.seed is not None:
        rows = list(rows)
        random.Random(config.seed).shuffle(rows)
    if config.sample_size is not None:
        rows = rows[: config.sample_size]
    return rows


def _build_scorers(config: IntelligenceConfig, hf_token: str | None) -> list[Scorer]:
    """Load every scorer, skipping any that fails to initialize.

    A scorer can legitimately be unavailable (e.g. the LLM judge with no API
    key), so a load failure is logged and that scorer is dropped rather than
    aborting the whole run.

    Args:
        config: The evaluation configuration.
        hf_token: HuggingFace token for gated model downloads.

    Returns:
        The successfully loaded scorers, in report order.
    """
    factories = [
        lambda: SemanticSimilarity(config.embedding_model, hf_token=hf_token),
        lambda: NLIEntailment(config.nli_model, hf_token=hf_token),
        lambda: BERTScore(config.bert_model, hf_token=hf_token),
        lambda: LLMJudge(config.judge_model, backend=config.judge_backend),
    ]
    scorers: list[Scorer] = []
    for factory in factories:
        try:
            scorers.append(factory())
        except Exception as exc:
            logger.error("Skipping scorer (failed to load): %s", exc)
    return scorers


def run_intelligence_eval(config: IntelligenceConfig) -> Path | None:
    """Run the semantic-similarity evaluation end to end.

    For each dataset item: sample frames, run the VLM with the configured
    prompt, then score the VLM's description against the ground-truth Sora
    prompt with embedding cosine similarity. Per-video records stream to
    ``predictions.jsonl`` and aggregate metrics land in ``summary.json``.

    Args:
        config: The evaluation configuration.

    Returns:
        The run directory, or ``None`` if model/embedder loading failed.
    """
    load_dotenv()
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    quiet_third_party_loggers()
    hf_token = os.environ.get("HF_TOKEN")

    dataset_label = config.dataset.replace("/", "_")
    run_id = (
        slugify(config.run_id)
        if config.run_id
        else build_run_id(config.model_id, dataset_label, config.num_frames)
    )
    try:
        run_dir = ensure_run_dir(Path(config.output_root), run_id)
    except FileExistsError:
        run_dir = Path(config.output_root) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(run_dir / "intelligence.log", mode="a")

    config_data = asdict(config)
    config_data["output_root"] = str(config.output_root)
    config_data["run_id"] = run_id
    write_json(run_dir / "config.json", config_data)

    source = DatasetSource(config.dataset, hf_token=hf_token)
    rows = _select_rows(source.metadata_rows(), config)
    logger.info("Selected %d scorable videos from %s", len(rows), config.dataset)
    if not rows:
        logger.error("No videos with ground-truth records found.")
        return run_dir

    try:
        model = HuggingFaceVLM(config.model_id, hf_token=hf_token)
    except Exception as exc:
        logger.error("Failed to load VLM: %s", exc)
        return None

    scorers = _build_scorers(config, hf_token)
    if not scorers:
        logger.error("No scorers could be loaded.")
        return None

    predictions_path = run_dir / "predictions.jsonl"
    predictions_path.unlink(missing_ok=True)

    results: list[EvalResult] = []
    for index, row in enumerate(rows, start=1):
        logger.info("[%d/%d] %s", index, len(rows), row.file_name)
        try:
            item = source.resolve(row)
        except Exception as exc:
            logger.error("  could not load %s: %s", row.file_name, exc)
            results.append(EvalResult(video=row.file_name, label=row.label,
                                      status="error", error=f"Load failed: {exc}"))
            append_jsonl(predictions_path, results[-1].to_dict())
            continue

        result = _evaluate_item(item, model, scorers, config)
        results.append(result)
        append_jsonl(predictions_path, result.to_dict())
        if result.status == "success":
            logger.info("  scores=%s response=%s", result.scores, result.response)
        else:
            logger.error("  %s", result.error)

    scorer_names = [s.name for s in scorers]
    summary = summarize(results, scorer_names)
    summary.update(model_id=config.model_id, dataset=config.dataset,
                   embedding_model=config.embedding_model)
    write_json(run_dir / "summary.json", summary)

    for name in scorer_names:
        logger.info("Mean %s: %s over %d videos",
                    name, summary[f"mean_{name}"], summary[f"scored_{name}"])
    logger.info("Predictions: %s", predictions_path)
    return run_dir


def _evaluate_item(
    item: DatasetItem,
    model: HuggingFaceVLM,
    scorers: list[Scorer],
    config: IntelligenceConfig,
) -> EvalResult:
    """Run inference and every scorer for one item, capturing any failure.

    A failure in one scorer records ``None`` for that scorer rather than
    failing the whole video; only a VLM/sampling failure marks the item errored.
    """
    frames, _duration, _total, _fps = sample_frames(item.video_path, num_frames=config.num_frames)
    if frames is None:
        return EvalResult(
            video=str(item.video_path), label=item.label, status="error",
            ground_truth_prompt=item.ground_truth_prompt, error="Could not sample frames",
        )
    try:
        generated = model.generate_from_frames(
            frames=frames, prompt_text=config.prompt, max_new_tokens=config.max_new_tokens,
        )
    except Exception as exc:
        return EvalResult(
            video=str(item.video_path), label=item.label, status="error",
            ground_truth_prompt=item.ground_truth_prompt, error=str(exc),
        )

    response = generated["response"]
    scores: dict[str, float | None] = {}
    for scorer in scorers:
        try:
            scores[scorer.name] = scorer.score(response, item.ground_truth_prompt)
        except Exception as exc:
            logger.warning("  scorer %s failed: %s", scorer.name, exc)
            scores[scorer.name] = None
    return EvalResult(
        video=str(item.video_path), label=item.label, status="success",
        ground_truth_prompt=item.ground_truth_prompt, response=response,
        scores=scores, query_latency_ms=generated["elapsed_ms"],
    )
