from __future__ import annotations

import re

from intelligence_eval.config import DEFAULT_JUDGE_MODEL
from intelligence_eval.scoring.base import Scorer

_SYSTEM_PROMPT = (
    "You are a strict evaluator for an accident-detection benchmark. You grade "
    "how well a model's video description matches the ground-truth description "
    "on the single question of whether an accident occurred. Reply with only "
    "the integer score."
)

_RUBRIC = """\
Score the candidate description against the ground truth using this rubric:
- 4: Correctly identifies whether an accident happened or not.
- 3: Mostly correct, but slightly uncertain or vague.
- 2: Ambiguous answer; does not clearly say accident or normal.
- 1: Mostly wrong, but contains some weak hint related to the ground truth.
- 0: Wrong accident/no-accident decision.

Ground truth:
{reference}

Candidate description:
{candidate}

Respond with only a single integer from 0 to 4."""


class LLMJudge(Scorer):
    """Rubric-based LLM-as-a-judge for accident detection.

    Routes a fixed 0-4 rubric prompt through the project's existing
    ``vlm_eval.llm`` backends (OpenAI by default), comparing the VLM's
    description against the ground-truth prompt on the single question of
    whether an accident occurred.
    """

    name = "llm_judge"

    def __init__(self, model: str = DEFAULT_JUDGE_MODEL, backend: str | None = None):
        from vlm_eval.llm.factory import get_llm_instance

        self.llm = get_llm_instance(model, backend=backend)

    def score(self, text_a: str, text_b: str) -> float:
        """Return the rubric score in ``[0, 4]`` for candidate vs reference.

        Args:
            text_a: Candidate description from the VLM.
            text_b: Ground-truth description.

        Returns:
            The integer rubric score as a float.

        Raises:
            ValueError: If the judge reply contains no 0-4 digit.
        """
        prompt = _RUBRIC.format(reference=text_b, candidate=text_a)
        reply, _, _ = self.llm.generate(prompt, sys_prompt=_SYSTEM_PROMPT)
        match = re.search(r"[0-4]", reply)
        if match is None:
            raise ValueError(f"Judge returned no 0-4 score: {reply!r}")
        return float(match.group())
