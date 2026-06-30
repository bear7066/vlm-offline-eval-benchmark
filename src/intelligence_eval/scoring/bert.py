from __future__ import annotations

from typing import Any

from intelligence_eval.config import DEFAULT_BERT_MODEL
from intelligence_eval.scoring.base import Scorer


class BERTScore(Scorer):
    """BERTScore token-matching F1.

    Embeds both texts with a contextual encoder, then greedily matches each
    candidate token to its most similar reference token (precision) and vice
    versa (recall), and returns their F1. This is the core BERTScore recipe on
    plain ``transformers`` so no extra dependency is pulled in.

    ponytail: no IDF weighting or baseline rescaling; add them if absolute
    scores need to be comparable to the published ``bert_score`` package.
    """

    name = "bert_score"

    def __init__(self, model_id: str = DEFAULT_BERT_MODEL, hf_token: str | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        self.model = AutoModel.from_pretrained(model_id, token=hf_token)
        self.model.eval()

    def _token_embeddings(self, text: str) -> Any:
        """Return L2-normalized contextual embeddings for non-special tokens."""
        inputs = self.tokenizer(text, truncation=True, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            hidden = self.model(**inputs).last_hidden_state[0]
        special = self.torch.tensor(
            self.tokenizer.get_special_tokens_mask(
                inputs["input_ids"][0].tolist(), already_has_special_tokens=True
            ),
            device=hidden.device,
            dtype=self.torch.bool,
        )
        embeddings = hidden[~special]
        return self.torch.nn.functional.normalize(embeddings, p=2, dim=1)

    def score(self, text_a: str, text_b: str) -> float:
        """Return the BERTScore F1 in ``[0, 1]`` of candidate vs reference.

        Args:
            text_a: Candidate text.
            text_b: Reference text.

        Returns:
            The greedy-matching F1; ``0.0`` if either text has no tokens.
        """
        cand = self._token_embeddings(text_a)
        ref = self._token_embeddings(text_b)
        if cand.shape[0] == 0 or ref.shape[0] == 0:
            return 0.0
        sim = cand @ ref.T  # (candidate_tokens, reference_tokens)
        precision = sim.max(dim=1).values.mean()
        recall = sim.max(dim=0).values.mean()
        f1 = 2 * precision * recall / (precision + recall)
        return float(f1)
