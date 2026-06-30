from __future__ import annotations

from abc import ABC, abstractmethod


class Scorer(ABC):
    """A reference-based text scorer.

    Each subclass compares a candidate text against a reference text and
    returns a single float from :meth:`score`. The scale is metric-specific
    (cosine in ``[-1, 1]``, a probability in ``[0, 1]``, a rubric point in
    ``[0, 4]``), so callers should not assume a shared range across scorers.
    ``name`` is the key the score is written under in ``summary.json``.
    """

    name: str = "score"

    @abstractmethod
    def score(self, text_a: str, text_b: str) -> float:
        """Score candidate ``text_a`` against reference ``text_b``."""
        raise NotImplementedError
