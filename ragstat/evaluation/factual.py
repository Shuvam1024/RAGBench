"""Verdict accuracy and evidence-sentence scores for labeled claims.

``answer_exact_match``, ``answer_token_f1``, and ``context_token_precision`` in
``answers.py`` are lexical overlap. They do not establish factual support, and
they are not faithfulness. The scores in this module compare a predicted
SUPPORT, CONTRADICT, or NEI label, and a set of evidence sentences, with the
gold rationale. They are not token overlap.
"""

from collections.abc import Sequence

VERDICTS = ("SUPPORT", "CONTRADICT", "NEI")
Sentence = tuple[str, int]


def sentence_scores(gold: set[Sentence], predicted: set[Sentence]) -> tuple[float, float, float]:
    """Return precision, recall, and F1 for one claim's evidence sentences.

    Both sets empty scores 1, 1, 1: the prediction abstained and the gold
    rationale is empty. An empty prediction against a nonempty gold set scores
    0, 0, 0. A nonempty prediction against an empty gold set also scores
    0, 0, 0. Otherwise precision is the overlap divided by the prediction
    count, recall is the overlap divided by the gold count, and F1 is their
    harmonic mean. F1 is 0 when precision and recall are both 0.
    """
    if not gold and not predicted:
        return 1.0, 1.0, 1.0
    if not gold or not predicted:
        return 0.0, 0.0, 0.0
    overlap = len(gold & predicted)
    precision = overlap / len(predicted)
    recall = overlap / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _counts(gold: Sequence[str], predicted: Sequence[str], label: str) -> tuple[int, int, int]:
    if len(gold) != len(predicted):
        raise ValueError("Gold and predicted verdicts must have the same length")
    for actual, guess in zip(gold, predicted, strict=True):
        if actual not in VERDICTS or guess not in VERDICTS:
            raise ValueError(f"Verdicts must be one of {VERDICTS}")
    true_positive = sum(
        actual == label and guess == label for actual, guess in zip(gold, predicted, strict=True)
    )
    false_positive = sum(
        guess == label and actual != label for actual, guess in zip(gold, predicted, strict=True)
    )
    false_negative = sum(
        actual == label and guess != label for actual, guess in zip(gold, predicted, strict=True)
    )
    return true_positive, false_positive, false_negative


def class_f1(gold: Sequence[str], predicted: Sequence[str], label: str) -> float:
    """F1 for one verdict. A class with no gold and no predictions scores 0."""
    if label not in VERDICTS:
        raise ValueError(f"Unknown verdict {label}")
    true_positive, false_positive, false_negative = _counts(gold, predicted, label)
    precision = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    )
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def macro_f1(gold: Sequence[str], predicted: Sequence[str]) -> float:
    """Unweighted mean of SUPPORT, CONTRADICT, and NEI F1."""
    if not gold:
        raise ValueError("Macro-F1 requires at least one claim")
    return sum(class_f1(gold, predicted, label) for label in VERDICTS) / len(VERDICTS)


def accuracy(gold: Sequence[str], predicted: Sequence[str]) -> float:
    """Fraction of claims whose predicted verdict equals the gold verdict."""
    if not gold or len(gold) != len(predicted):
        raise ValueError("Accuracy requires one predicted verdict per gold verdict")
    for actual, guess in zip(gold, predicted, strict=True):
        if actual not in VERDICTS or guess not in VERDICTS:
            raise ValueError(f"Verdicts must be one of {VERDICTS}")
    return sum(actual == guess for actual, guess in zip(gold, predicted, strict=True)) / len(gold)


def micro_sentence_scores(
    gold: Sequence[set[Sentence]], predicted: Sequence[set[Sentence]]
) -> tuple[float, float, float]:
    """Pooled sentence precision, recall, and F1.

    Claims with both sets empty add nothing to the pool. When every claim is
    in that case, the pooled scores are 1, 1, 1.
    """
    if len(gold) != len(predicted) or not gold:
        raise ValueError("Micro sentence scores require one prediction set per gold set")
    overlap = 0
    predicted_count = 0
    gold_count = 0
    for actual, guess in zip(gold, predicted, strict=True):
        overlap += len(actual & guess)
        predicted_count += len(guess)
        gold_count += len(actual)
    if predicted_count == 0 and gold_count == 0:
        return 1.0, 1.0, 1.0
    precision = overlap / predicted_count if predicted_count else 0.0
    recall = overlap / gold_count if gold_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1
