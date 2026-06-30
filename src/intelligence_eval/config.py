from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from intelligence_eval.dataset import DEFAULT_DATASET


# The model is asked to describe the clip richly, since its output is scored
# against the (paragraph-length) Sora generation prompt that created the video.
DEFAULT_PROMPT = (
    "Describe this video in detail. Include the setting, the person, and the "
    "main action or event that takes place."
)

# Small, fast, widely-used sentence embedding model. Loaded via plain
# transformers (already a dependency) so no new package is needed.
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# NLI cross-encoder for entailment scoring. roberta-large-mnli uses a BPE
# tokenizer (no sentencepiece) and ships standard MNLI labels.
DEFAULT_NLI_MODEL = "roberta-large-mnli"

# Encoder backing BERTScore token-matching F1.
DEFAULT_BERT_MODEL = "bert-base-uncased"

# Chat model used by the rubric LLM judge (routed through vlm_eval.llm).
DEFAULT_JUDGE_MODEL = "gpt-4o-mini"


@dataclass(frozen=True)
class IntelligenceConfig:
    """Configuration for an intelligence (semantic-similarity) evaluation.

    Attributes:
        dataset: Dataset source: a HuggingFace dataset repo id (default) or a
            local directory containing ``metadata.csv`` plus the ``videos/``
            and ``records/`` trees it references.
        model_id: HuggingFace VLM ID to evaluate.
        prompt: Instruction sent to the VLM with the sampled frames.
        num_frames: Frames sampled per video.
        max_new_tokens: Generation cap for the VLM.
        sample_size: Cap on evaluated videos (``None`` = all).
        seed: RNG seed for sampling; ``None`` keeps metadata order.
        embedding_model: Sentence-embedding model used to score similarity.
        nli_model: NLI cross-encoder used for entailment scoring.
        bert_model: Encoder backing BERTScore.
        judge_model: Chat model used by the rubric LLM judge.
        judge_backend: Backend for the judge model (``None`` = auto-detect).
        output_root: Parent directory for run outputs.
        run_id: Explicit run id; ``None`` builds one from model/dataset/time.
    """

    model_id: str
    dataset: str = DEFAULT_DATASET
    prompt: str = DEFAULT_PROMPT
    num_frames: int = 8
    max_new_tokens: int = 150
    sample_size: int | None = None
    seed: int | None = None
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    nli_model: str = DEFAULT_NLI_MODEL
    bert_model: str = DEFAULT_BERT_MODEL
    judge_model: str = DEFAULT_JUDGE_MODEL
    judge_backend: str | None = None
    output_root: Path = Path("intelligence_runs")
    run_id: str | None = None
