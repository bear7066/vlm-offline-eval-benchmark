from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def ngrams(tokens: list[str], order: int) -> list[tuple[str, ...]]:
    if order <= 0 or len(tokens) < order:
        return []
    return [tuple(tokens[i : i + order]) for i in range(len(tokens) - order + 1)]


def sentence_bleu(reference: str, hypothesis: str, max_order: int = 4) -> float:
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)
    if not ref_tokens or not hyp_tokens:
        return 0.0

    precisions: list[float] = []
    for order in range(1, max_order + 1):
        hyp_counts = Counter(ngrams(hyp_tokens, order))
        if not hyp_counts:
            continue

        ref_counts = Counter(ngrams(ref_tokens, order))
        clipped = sum(min(count, ref_counts[ngram]) for ngram, count in hyp_counts.items())
        total = sum(hyp_counts.values())
        precisions.append(clipped / total if clipped else 1.0 / (2.0 * total))

    if not precisions:
        return 0.0

    ref_len = len(ref_tokens)
    hyp_len = len(hyp_tokens)
    brevity_penalty = 1.0 if hyp_len > ref_len else math.exp(1.0 - ref_len / hyp_len)
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    return brevity_penalty * geo_mean


def corpus_bleu(references: list[str], hypotheses: list[str], max_order: int = 4) -> float:
    total_ref_len = 0
    total_hyp_len = 0
    matches_by_order = [0] * max_order
    possible_by_order = [0] * max_order

    for reference, hypothesis in zip(references, hypotheses):
        ref_tokens = tokenize(reference)
        hyp_tokens = tokenize(hypothesis)
        total_ref_len += len(ref_tokens)
        total_hyp_len += len(hyp_tokens)

        for order in range(1, max_order + 1):
            hyp_counts = Counter(ngrams(hyp_tokens, order))
            ref_counts = Counter(ngrams(ref_tokens, order))
            matches_by_order[order - 1] += sum(
                min(count, ref_counts[ngram]) for ngram, count in hyp_counts.items()
            )
            possible_by_order[order - 1] += sum(hyp_counts.values())

    if total_hyp_len == 0:
        return 0.0

    precisions: list[float] = []
    for matches, possible in zip(matches_by_order, possible_by_order):
        if possible:
            precisions.append((matches + 1.0) / (possible + 1.0))

    if not precisions:
        return 0.0

    brevity_penalty = 1.0 if total_hyp_len > total_ref_len else math.exp(
        1.0 - total_ref_len / total_hyp_len
    )
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    return brevity_penalty * geo_mean


def rouge_l(reference: str, hypothesis: str) -> float:
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)
    if not ref_tokens or not hyp_tokens:
        return 0.0

    lcs = _lcs_length(ref_tokens, hyp_tokens)
    if lcs == 0:
        return 0.0

    precision = lcs / len(hyp_tokens)
    recall = lcs / len(ref_tokens)
    return 2.0 * precision * recall / (precision + recall)


def _lcs_length(a: list[str], b: list[str]) -> int:
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[-1]))
        previous = current
    return previous[-1]


class CiderScorer:
    def __init__(self, references: list[str], max_order: int = 4):
        self.max_order = max_order
        self.document_count = max(len(references), 1)
        self.document_frequency: dict[int, Counter[tuple[str, ...]]] = {}

        for order in range(1, max_order + 1):
            document_frequency: Counter[tuple[str, ...]] = Counter()
            for reference in references:
                document_frequency.update(set(ngrams(tokenize(reference), order)))
            self.document_frequency[order] = document_frequency

    def score(self, reference: str, hypothesis: str) -> float:
        ref_tokens = tokenize(reference)
        hyp_tokens = tokenize(hypothesis)
        if not ref_tokens or not hyp_tokens:
            return 0.0

        similarities: list[float] = []
        for order in range(1, self.max_order + 1):
            ref_vector = self._tfidf_vector(ref_tokens, order)
            hyp_vector = self._tfidf_vector(hyp_tokens, order)
            if ref_vector and hyp_vector:
                similarities.append(_cosine_similarity(ref_vector, hyp_vector))

        if not similarities:
            return 0.0
        return 10.0 * sum(similarities) / len(similarities)

    def _tfidf_vector(self, tokens: list[str], order: int) -> dict[tuple[str, ...], float]:
        counts = Counter(ngrams(tokens, order))
        total = sum(counts.values())
        if total == 0:
            return {}

        vector: dict[tuple[str, ...], float] = {}
        for ngram, count in counts.items():
            tf = count / total
            df = self.document_frequency[order].get(ngram, 0)
            idf = math.log((self.document_count + 1.0) / (df + 1.0)) + 1.0
            vector[ngram] = tf * idf
        return vector


def _cosine_similarity(
    left: dict[tuple[str, ...], float],
    right: dict[tuple[str, ...], float],
) -> float:
    numerator = sum(value * right.get(key, 0.0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def score_items(items: list[dict[str, Any]]) -> tuple[list[dict[str, float]], dict[str, float]]:
    references = [str(item.get("label", "")) for item in items]
    hypotheses = [str(item.get("response") or item.get("answer", "")) for item in items]
    cider_scorer = CiderScorer(references)

    item_scores: list[dict[str, float]] = []
    totals: defaultdict[str, float] = defaultdict(float)
    for reference, hypothesis in zip(references, hypotheses):
        scores = {
            "bleu": sentence_bleu(reference, hypothesis),
            "rouge_l": rouge_l(reference, hypothesis),
            "cider": cider_scorer.score(reference, hypothesis),
        }
        item_scores.append(scores)
        for name, value in scores.items():
            totals[name] += value

    count = len(item_scores)
    if count == 0:
        return item_scores, {"bleu": 0.0, "corpus_bleu": 0.0, "rouge_l": 0.0, "cider": 0.0}

    return item_scores, {
        "bleu": totals["bleu"] / count,
        "corpus_bleu": corpus_bleu(references, hypotheses),
        "rouge_l": totals["rouge_l"] / count,
        "cider": totals["cider"] / count,
    }
