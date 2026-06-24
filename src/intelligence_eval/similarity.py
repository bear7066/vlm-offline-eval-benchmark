from __future__ import annotations

from typing import Any

from intelligence_eval.config import DEFAULT_EMBEDDING_MODEL


class SemanticSimilarity:
    """Cosine similarity between sentence embeddings.

    Wraps a HuggingFace transformer encoder with mean pooling, the standard
    Sentence-Transformers recipe, implemented on plain ``transformers`` so no
    extra dependency is pulled in. Lexical metrics (BLEU/ROUGE) would compare
    surface words; embeddings compare meaning, which is what's wanted when a
    short model description is matched against a paragraph-length prompt.
    """

    def __init__(self, model_id: str = DEFAULT_EMBEDDING_MODEL, hf_token: str | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        self.model = AutoModel.from_pretrained(model_id, token=hf_token)
        self.model.eval()

    def embed(self, texts: list[str]) -> Any:
        """Encode texts into L2-normalized mean-pooled embeddings.

        Args:
            texts: Input strings.

        Returns:
            A ``(len(texts), hidden)`` tensor of unit-norm embeddings.
        """
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        ).to(self.model.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)

        # Mean-pool token embeddings, masking out padding positions.
        mask = inputs["attention_mask"].unsqueeze(-1).to(outputs.last_hidden_state.dtype)
        summed = (outputs.last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts
        return self.torch.nn.functional.normalize(pooled, p=2, dim=1)

    def score(self, text_a: str, text_b: str) -> float:
        """Return cosine similarity in ``[-1, 1]`` between two texts."""
        embeddings = self.embed([text_a, text_b])
        return float((embeddings[0] * embeddings[1]).sum())
