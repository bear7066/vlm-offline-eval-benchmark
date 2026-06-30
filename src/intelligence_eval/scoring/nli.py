from __future__ import annotations

from intelligence_eval.config import DEFAULT_NLI_MODEL
from intelligence_eval.scoring.base import Scorer


class NLIEntailment(Scorer):
    """Natural-language-inference entailment probability.

    Treats the reference (ground-truth prompt) as the premise and the
    candidate (VLM description) as the hypothesis, then returns the model's
    probability that the premise entails the hypothesis. Unlike cosine
    similarity this is directional: it rewards a candidate whose claims are
    actually supported by the reference rather than merely topically close.
    """

    name = "nli"

    def __init__(self, model_id: str = DEFAULT_NLI_MODEL, hf_token: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, token=hf_token)
        self.model.eval()
        # Label order varies by checkpoint, so find the entailment class by name.
        self.entail_idx = next(
            idx for idx, label in self.model.config.id2label.items()
            if label.lower().startswith("entail")
        )

    def score(self, text_a: str, text_b: str) -> float:
        """Return P(entailment) in ``[0, 1]`` that reference ``text_b`` entails ``text_a``.

        Args:
            text_a: Candidate text (hypothesis).
            text_b: Reference text (premise).

        Returns:
            The entailment probability.
        """
        inputs = self.tokenizer(
            text_b, text_a, truncation=True, return_tensors="pt"
        ).to(self.model.device)
        with self.torch.no_grad():
            logits = self.model(**inputs).logits
        probs = logits.softmax(dim=-1)[0]
        return float(probs[self.entail_idx])
