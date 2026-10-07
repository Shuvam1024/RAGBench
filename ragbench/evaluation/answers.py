"""Lexical overlap between an answer, a reference, and retrieved context.

``answer_exact_match``, ``answer_token_f1``, and ``context_token_precision``
compare tokens. ``context_token_precision`` does not establish factual support.
It is not a faithfulness score.
"""

import re
from collections import Counter
from collections.abc import Sequence

from ragbench.generation.providers import ContextSource


def tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


def score_answer(answer: str, reference: str, context: Sequence[ContextSource]) -> dict[str, float]:
    predicted, expected = tokens(answer), tokens(reference)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    precision = overlap / len(predicted) if predicted else 0.0
    recall = overlap / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    evidence = set(tokens(" ".join(source.text for source in context)))
    support = sum(token in evidence for token in predicted) / len(predicted) if predicted else 0.0
    return {
        "answer_exact_match": float(bool(predicted) and predicted == expected),
        "answer_token_f1": f1,
        "context_token_precision": support,
    }
