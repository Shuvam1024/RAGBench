"""Hand-checked verdict and evidence-sentence scores."""

import json
from pathlib import Path

import pytest

from ragstat.evaluation.factual import (
    accuracy,
    class_f1,
    evidence_only_sentence_f1,
    macro_f1,
    micro_sentence_scores,
    sentence_scores,
)

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "factual_metrics.json").read_text(encoding="utf-8")
)

# Arithmetic for the fixture, written out rather than copied from the scorer.
# Claim a: overlap 1, both sets size 2, so precision = recall = F1 = 1/2.
# Claim b: gold nonempty, prediction empty, so 0, 0, 0.
# Claim c: both empty, so 1, 1, 1.
# Claim d: gold empty, prediction nonempty, so 0, 0, 0.
# Claim e: overlap 1, prediction size 2, gold size 1, so precision 1/2, recall 1, F1 2/3.
# Accuracy is 3/5. SUPPORT F1 is 4/5 (tp 2, fp 1, fn 0). CONTRADICT F1 is 0.
# NEI F1 is 1/2 (tp 1, fp 1, fn 1). Macro-F1 is (4/5 + 0 + 1/2) / 3 = 13/30.
# Mean sentence F1 is (1/2 + 0 + 1 + 0 + 2/3) / 5 = 13/30.
# Micro: overlap 2, predicted 5, gold 4, so precision 2/5, recall 1/2, F1 4/9.
HAND = {
    "accuracy": 3 / 5,
    "macro_f1": (4 / 5 + 0 + 1 / 2) / 3,
    "class_f1": {"SUPPORT": 4 / 5, "CONTRADICT": 0.0, "NEI": 1 / 2},
    "sentence_precision": (1 / 2 + 0 + 1 + 0 + 1 / 2) / 5,
    "sentence_recall": (1 / 2 + 0 + 1 + 0 + 1) / 5,
    "sentence_f1": (1 / 2 + 0 + 1 + 0 + 2 / 3) / 5,
    "micro_sentence_precision": 2 / 5,
    "micro_sentence_recall": 1 / 2,
    "micro_sentence_f1": 4 / 9,
}


def _sentence_set(values: list[str]) -> set[tuple[str, int]]:
    parsed: set[tuple[str, int]] = set()
    for value in values:
        document_id, index = value.rsplit(":", 1)
        parsed.add((document_id, int(index)))
    return parsed


def test_fixture_matches_the_hand_arithmetic_and_the_scorer() -> None:
    for name, expected in HAND.items():
        if name == "class_f1":
            continue
        assert FIXTURE[name] == pytest.approx(expected)
    for label, expected in HAND["class_f1"].items():
        assert FIXTURE["class_f1"][label] == pytest.approx(expected)

    gold = [claim["gold"] for claim in FIXTURE["claims"]]
    predicted = [claim["predicted"] for claim in FIXTURE["claims"]]
    assert accuracy(gold, predicted) == pytest.approx(HAND["accuracy"])
    assert macro_f1(gold, predicted) == pytest.approx(HAND["macro_f1"])
    for label, expected in HAND["class_f1"].items():
        assert class_f1(gold, predicted, label) == pytest.approx(expected)

    per_claim = [
        sentence_scores(
            _sentence_set(claim["gold_sentences"]), _sentence_set(claim["predicted_sentences"])
        )
        for claim in FIXTURE["claims"]
    ]
    for claim, scores in zip(FIXTURE["claims"], per_claim, strict=True):
        assert scores[0] == pytest.approx(claim["sentence_precision"])
        assert scores[1] == pytest.approx(claim["sentence_recall"])
        assert scores[2] == pytest.approx(claim["sentence_f1"])
    assert sum(item[0] for item in per_claim) / 5 == pytest.approx(HAND["sentence_precision"])
    assert sum(item[2] for item in per_claim) / 5 == pytest.approx(HAND["sentence_f1"])
    micro = micro_sentence_scores(
        [_sentence_set(claim["gold_sentences"]) for claim in FIXTURE["claims"]],
        [_sentence_set(claim["predicted_sentences"]) for claim in FIXTURE["claims"]],
    )
    assert micro[0] == pytest.approx(HAND["micro_sentence_precision"])
    assert micro[1] == pytest.approx(HAND["micro_sentence_recall"])
    assert micro[2] == pytest.approx(HAND["micro_sentence_f1"])
    evidence_only = evidence_only_sentence_f1(
        [_sentence_set(claim["gold_sentences"]) for claim in FIXTURE["claims"]],
        [_sentence_set(claim["predicted_sentences"]) for claim in FIXTURE["claims"]],
    )
    # Claims a, b, and e have gold evidence. c and d do not.
    assert evidence_only == pytest.approx((1 / 2 + 0 + 2 / 3) / 3)


def test_docs_say_lexical_metrics_are_not_factual_support() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    results = Path("docs/results.md").read_text(encoding="utf-8")
    providers = Path("docs/providers.md").read_text(encoding="utf-8")
    assert "does not establish factual support" in readme
    assert "does not establish factual support" in results
    assert "does not establish factual support" in providers
    assert "BEIR qrels" in readme
    assert "BEIR qrels" in results
    assert "claims_test.jsonl" in results


def test_empty_sets_and_rejected_labels() -> None:
    assert sentence_scores(set(), set()) == (1.0, 1.0, 1.0)
    assert sentence_scores({("d", 0)}, set()) == (0.0, 0.0, 0.0)
    assert sentence_scores(set(), {("d", 0)}) == (0.0, 0.0, 0.0)
    assert micro_sentence_scores([set(), set()], [set(), set()]) == (1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="one predicted"):
        accuracy([], [])
    with pytest.raises(ValueError, match="at least one"):
        macro_f1([], [])
    with pytest.raises(ValueError, match="same length"):
        class_f1(["SUPPORT"], [], "SUPPORT")
    with pytest.raises(ValueError, match="Unknown"):
        class_f1(["SUPPORT"], ["SUPPORT"], "MAYBE")
    with pytest.raises(ValueError, match="Verdicts"):
        accuracy(["SUPPORT"], ["MAYBE"])
    with pytest.raises(ValueError, match="one prediction"):
        micro_sentence_scores([set()], [])
    assert evidence_only_sentence_f1([set(), {("d", 0)}], [set(), set()]) == pytest.approx(0.0)
    assert evidence_only_sentence_f1([set(), set()], [set(), set()]) is None
    with pytest.raises(ValueError, match="one prediction"):
        evidence_only_sentence_f1([set()], [])
