from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from vlm_eval.config import JudgeConfig
from vlm_eval.judge.parser import extract_score, load_predictions_jsonl, parse_legacy_log_file
from vlm_eval.judge.prompts import build_judge_prompt
from vlm_eval.judge.text_metrics import score_items
from vlm_eval.llm.factory import get_llm_instance
from vlm_eval.logging_utils import configure_logging
from vlm_eval.paths import find_latest_run, label_from_video_dir, model_name_from_id


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _append_jsonl(path: Path, data: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        f.write("\n")


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def resolve_prediction_source(config: JudgeConfig) -> tuple[Path | None, Path | None, bool]:
    if config.predictions_path is not None:
        predictions = Path(config.predictions_path)
        return predictions, predictions.parent, False

    if config.run_dir is not None:
        run_dir = Path(config.run_dir)
        return run_dir / "predictions.jsonl", run_dir, False

    if config.legacy_log_path is not None:
        legacy_path = Path(config.legacy_log_path)
        return legacy_path, legacy_path.parent, True

    if config.model_id is not None and config.video_dir is not None:
        run_dir = find_latest_run(
            output_root=Path(config.output_root),
            model_id=config.model_id,
            video_dir=Path(config.video_dir),
            sample_fps=config.sample_fps,
        )
        if run_dir is not None:
            return run_dir / "predictions.jsonl", run_dir, False

        legacy_log = _find_latest_legacy_log(
            model_id=config.model_id,
            video_dir=Path(config.video_dir),
            sample_fps=config.sample_fps,
            num_frames=config.num_frames,
        )
        if legacy_log is not None:
            return legacy_log, legacy_log.parent, True

    return None, None, False


def _find_latest_legacy_log(
    model_id: str,
    video_dir: Path,
    sample_fps: float | None,
    num_frames: int | None,
) -> Path | None:
    label = label_from_video_dir(video_dir)
    model_name = model_name_from_id(model_id)
    candidates: list[Path] = []

    if sample_fps is not None:
        candidates.extend(Path.cwd().glob(f"{model_name}-{sample_fps:g}fps/{label}.log"))
    if num_frames is not None:
        candidates.extend(Path.cwd().glob(f"{model_name}-{num_frames}frames/{label}.log"))
    candidates.extend(Path.cwd().glob(f"{model_name}-*/{label}.log"))

    existing = [path for path in candidates if path.exists()]
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)


def run_judge(config: JudgeConfig) -> Path | None:
    _load_dotenv_if_available()
    source_path, run_dir, is_legacy = resolve_prediction_source(config)

    if source_path is None or run_dir is None:
        configure_logging(None)
        logging.error("Could not resolve predictions source. Provide --run_dir or --predictions.")
        return None

    output_log = run_dir / "judge.log"
    configure_logging(output_log, mode="w")

    if not source_path.exists():
        logging.error("Prediction source not found: %s", source_path)
        return None

    logging.info("Reading predictions from: %s", source_path)
    items = parse_legacy_log_file(source_path) if is_legacy else load_predictions_jsonl(source_path)
    logging.info("Loaded %s predictions for judging.", len(items))

    if not items:
        logging.warning("No judgeable predictions found.")
        return run_dir

    text_item_scores, text_summary = score_items(items)
    logging.info(
        "Text metrics: BLEU %.4f, corpus BLEU %.4f, ROUGE-L %.4f, CIDEr %.4f",
        text_summary["bleu"],
        text_summary["corpus_bleu"],
        text_summary["rouge_l"],
        text_summary["cider"],
    )

    llm = None
    if not config.skip_llm_judge:
        try:
            llm = get_llm_instance(config.judge_model, backend=config.backend)
        except Exception as exc:
            logging.error("Could not initialize judge model '%s': %s", config.judge_model, exc)
            return None

    results_path = run_dir / "judge_results.jsonl"
    summary_path = run_dir / "judge_summary.json"
    if results_path.exists():
        results_path.unlink()

    total_score = 0
    valid_count = 0

    logging.info("")
    if config.skip_llm_judge:
        logging.info("Skipping LLM-as-a-judge evaluation.")
    else:
        logging.info("Starting LLM-as-a-judge evaluation.")
    logging.info("")

    for index, item in enumerate(items, start=1):
        video = item["video"]
        answer = item.get("response") or item.get("answer", "")
        label = item.get("label") or "Unknown Action"
        text_metrics = text_item_scores[index - 1]

        logging.info("[%s/%s] Evaluating video: %s", index, len(items), video)
        logging.info("Ground Truth: %s", label)
        logging.info("VLM answer: %s", answer)
        logging.info(
            "Text metrics: BLEU %.4f, ROUGE-L %.4f, CIDEr %.4f",
            text_metrics["bleu"],
            text_metrics["rouge_l"],
            text_metrics["cider"],
        )

        if llm is None:
            judge_result = None
            prompt_tokens = 0
            completion_tokens = 0
        else:
            try:
                judge_result, prompt_tokens, completion_tokens = llm.generate(
                    build_judge_prompt(answer, label)
                )
                judge_result = judge_result.strip()
            except Exception as exc:
                judge_result = f"Judge evaluation failed: {exc}"
                prompt_tokens = 0
                completion_tokens = 0

        score = extract_score(judge_result) if judge_result else None
        if score is not None:
            total_score += score
            valid_count += 1
        elif llm is not None:
            logging.warning("Could not parse Score")

        output_item = {
            "video": video,
            "label": label,
            "answer": answer,
            "bleu": text_metrics["bleu"],
            "rouge_l": text_metrics["rouge_l"],
            "cider": text_metrics["cider"],
            "text_metrics": text_metrics,
            "judge_result": judge_result,
            "score": score,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
        _append_jsonl(results_path, output_item)

        if judge_result is not None:
            logging.info("Judge result:\n%s", judge_result)
        logging.info("-" * 50)

    average_score = total_score / valid_count if valid_count > 0 else None
    summary = {
        "judge_model": config.judge_model,
        "backend": config.backend,
        "source": str(source_path),
        "total_score": total_score,
        "valid_count": valid_count,
        "attempted_count": len(items),
        "average_score": average_score,
        "text_metrics": text_summary,
    }
    _write_json(summary_path, summary)

    logging.info("")
    logging.info("Evaluation Summary")
    logging.info("Text BLEU: %.4f", text_summary["bleu"])
    logging.info("Text corpus BLEU: %.4f", text_summary["corpus_bleu"])
    logging.info("Text ROUGE-L: %.4f", text_summary["rouge_l"])
    logging.info("Text CIDEr: %.4f", text_summary["cider"])
    if average_score is not None:
        logging.info("Total score: %s", total_score)
        logging.info("Valid samples: %s", valid_count)
        logging.info("Average score: %.2f", average_score)
    else:
        logging.warning("No valid scores were parsed.")
    logging.info("Judge results written to: %s", results_path)
    logging.info("Judge summary written to: %s", summary_path)

    return run_dir
