"""Verdict policy, injected NLI scorer, and paired comparison."""

import warnings
from pathlib import Path

import numpy as np
import pytest
import yaml

from ragbench.datasets.scifact_claims import AbstractDocument, EvidenceSentence, SciFactClaim
from ragbench.evaluation.nli import CrossEncoderNli, NliModelConfig, NliScores, classify_nli
from ragbench.evaluation.statistics import bootstrap_mean_ci
from ragbench.evaluation.verdict import (
    ClaimPrediction,
    VerdictEvaluation,
    VerdictPolicy,
    apply_policy,
    assert_train_selection,
    build_sentence_jobs,
    choose_policy,
    compare_verdicts,
    evaluate_predictions,
    iter_policies,
    load_verdict_experiment,
    majority_verdict,
    predict_baseline,
    predictions_for_policy,
    premise_text,
    retrieve_top_documents,
    score_sentence_jobs,
)
from ragbench.ingestion.loader import Document


class _FakeNli:
    def __init__(self, rows: list[list[float]]) -> None:
        self.rows = rows
        self.pairs: list[tuple[str, str]] = []
        self.kwargs: dict[str, object] = {}

    def tokenizer(
        self, premises: list[str], hypotheses: list[str], **_kwargs: object
    ) -> dict[str, list[list[int]]]:
        self.pairs = list(zip(premises, hypotheses, strict=True))
        lengths = [3, 9]
        return {"input_ids": [[1] * lengths[index % 2] for index in range(len(premises))]}

    def predict(self, pairs: list[tuple[str, str]], **kwargs: object) -> np.ndarray:
        self.kwargs = kwargs
        return np.asarray(self.rows, dtype=np.float64)


def _claim(
    claim_id: str,
    text: str,
    label: str | None,
    sentences: tuple[tuple[str, int], ...] = (),
) -> SciFactClaim:
    evidence = tuple(
        EvidenceSentence(
            document_id=document_id,
            sentence=index,
            label="SUPPORT" if label == "SUPPORT" else "CONTRADICT",
        )
        for document_id, index in sentences
    )
    return SciFactClaim(
        id=claim_id,
        text=text,
        evidence=evidence,
        cited_doc_ids=tuple(dict.fromkeys(document_id for document_id, _index in sentences))
        or ("9",),
    )


def test_classify_breaks_ties_toward_contradiction() -> None:
    assert classify_nli(NliScores(0.2, 0.7, 0.1)) == ("SUPPORT", 0.7)
    assert classify_nli(NliScores(0.5, 0.5, 0.0)) == ("CONTRADICT", 0.5)
    assert classify_nli(NliScores(0.2, 0.2, 0.6)) == ("NEI", 0.6)
    with pytest.raises(ValueError, match="nonnegative"):
        NliScores(0.0, -0.1, 0.0)


def test_policy_keeps_a_prefix_and_abstains_below_the_gate() -> None:
    rows = score_sentence_jobs(
        build_sentence_jobs(
            [_claim("1", "Vitamin C cures scurvy.", "SUPPORT", (("a", 0),))],
            {
                "a": AbstractDocument(
                    id="a", title="A", sentences=("Supports the claim.", "Neutral aside.")
                ),
                "b": AbstractDocument(id="b", title="B", sentences=("Contradicts the claim.",)),
                "c": AbstractDocument(id="c", title="C", sentences=("Also supports.",)),
            },
            {"1": ("a", "b", "c")},
            doc_k=3,
        ),
        _Recording(),
    )["1"]
    # Ranks: a0 SUPPORT 0.80, a1 NEI 0.70, b0 CONTRADICT 0.75, c0 SUPPORT 0.90.
    assert (
        apply_policy(rows, VerdictPolicy(doc_k=1, sentence_k=1, min_confidence=0.5)).verdict
        == "SUPPORT"
    )
    assert apply_policy(
        rows, VerdictPolicy(doc_k=1, sentence_k=1, min_confidence=0.5)
    ).sentences == (("a", 0),)
    abstain = apply_policy(rows, VerdictPolicy(doc_k=1, sentence_k=1, min_confidence=0.9))
    assert abstain.verdict == "NEI" and abstain.sentences == ()
    mixed = apply_policy(rows, VerdictPolicy(doc_k=3, sentence_k=2, min_confidence=0.5))
    assert mixed.verdict == "SUPPORT"
    assert mixed.sentences == (("c", 0), ("a", 0))
    top = apply_policy(rows, VerdictPolicy(doc_k=3, sentence_k=1, min_confidence=0.7))
    assert top.sentences == (("c", 0),)


class _Recording:
    def score(self, pairs: list[tuple[str, str]]) -> list[NliScores]:
        by_premise = {
            "Supports the claim.": NliScores(0.1, 0.8, 0.1),
            "Neutral aside.": NliScores(0.1, 0.2, 0.7),
            "Contradicts the claim.": NliScores(0.75, 0.15, 0.1),
            "Also supports.": NliScores(0.05, 0.9, 0.05),
        }
        return [by_premise[premise] for premise, _hypothesis in pairs]

    @property
    def metadata(self) -> dict[str, str | int]:
        return {"nli_revision": "test-revision"}

    @property
    def truncated_pairs(self) -> int:
        return 0


def test_selection_tie_break_and_majority_baseline() -> None:
    claims = [
        _claim("1", "Alpha beta", "SUPPORT", (("a", 0),)),
        _claim("2", "Gamma delta", None),
    ]
    low = evaluate_predictions(
        claims,
        {
            "1": ClaimPrediction(verdict="NEI", sentences=()),
            "2": ClaimPrediction(verdict="NEI", sentences=()),
        },
        split="train",
        system="nli",
        model_name="recording",
        model_revision="test-revision",
        pair_order="premise_sentence_hypothesis_claim",
    )
    high = evaluate_predictions(
        claims,
        {
            "1": ClaimPrediction(verdict="SUPPORT", sentences=(("a", 0),)),
            "2": ClaimPrediction(verdict="NEI", sentences=()),
        },
        split="train",
        system="nli",
        model_name="recording",
        model_revision="test-revision",
        pair_order="premise_sentence_hypothesis_claim",
        doc_k=1,
        sentence_k=1,
        min_confidence=0.7,
    )
    same = high.model_copy(update={})
    winner, _report = choose_policy(
        [
            (VerdictPolicy(doc_k=3, sentence_k=2, min_confidence=0.5), low),
            (VerdictPolicy(doc_k=3, sentence_k=1, min_confidence=0.5), high),
            (VerdictPolicy(doc_k=1, sentence_k=2, min_confidence=0.5), same),
            (VerdictPolicy(doc_k=1, sentence_k=1, min_confidence=0.7), same),
            (VerdictPolicy(doc_k=1, sentence_k=1, min_confidence=0.5), same),
        ]
    )
    assert (winner.doc_k, winner.sentence_k, winner.min_confidence) == (1, 1, 0.7)
    assert majority_verdict(["SUPPORT", "SUPPORT", "NEI"]) == (
        "SUPPORT",
        {"SUPPORT": 2, "CONTRADICT": 0, "NEI": 1},
    )
    assert majority_verdict(["SUPPORT", "NEI"])[0] == "NEI"
    with pytest.raises(ValueError, match="train"):
        assert_train_selection("dev")
    with pytest.raises(ValueError, match="at least one"):
        choose_policy([])
    assert (
        len(iter_policies(load_verdict_experiment(Path("configs/scifact-verdict-train.yaml")).grid))
        == 8
    )


def test_document_retrieval_and_baseline_sentence_zero() -> None:
    documents = [
        Document(id="a", text="Alpha beta gamma", content_sha256="1"),
        Document(id="b", text="Other topic entirely", content_sha256="2"),
        Document(id="c", text="Unrelated catalog index", content_sha256="3"),
    ]
    found, elapsed = retrieve_top_documents(documents, [("1", "alpha beta")], k=1, k1=0.9, b=0.4)
    assert found["1"] == ("a",)
    assert elapsed and elapsed[0] >= 0
    abstracts = {
        "a": AbstractDocument(id="a", title="A", sentences=("Alpha sentence.", "Later.")),
        "b": AbstractDocument(id="b", title="B", sentences=("Other.",)),
    }
    claims = [_claim("1", "alpha beta", "SUPPORT", (("a", 0),))]
    baseline = predict_baseline(claims, found, abstracts, "SUPPORT")
    assert baseline["1"].sentences == (("a", 0),)
    nei = predict_baseline(claims, found, abstracts, "NEI")
    assert nei["1"].verdict == "NEI" and nei["1"].sentences == ()
    report = evaluate_predictions(
        claims,
        baseline,
        split="dev",
        system="majority_baseline",
        model_name="majority_train_verdict",
        model_revision="none",
        pair_order="not_applicable",
        baseline_verdict="SUPPORT",
        doc_k=1,
        sentence_k=1,
    )
    assert report.accuracy == 1
    assert report.sentence_f1 == 1
    with pytest.raises(ValueError, match="withheld"):
        evaluate_predictions(
            [SciFactClaim(id="9", text="Hidden.", labels_withheld=True)],
            {"9": ClaimPrediction(verdict="NEI")},
            split="dev",
            system="nli",
            model_name="recording",
            model_revision="test-revision",
            pair_order="premise_sentence_hypothesis_claim",
        )


def test_paired_comparison_uses_the_shared_bootstrap() -> None:
    claims = [
        _claim("1", "Alpha beta", "SUPPORT", (("a", 0),)),
        _claim("2", "Gamma delta", None),
    ]
    baseline = evaluate_predictions(
        claims,
        {
            "1": ClaimPrediction(verdict="SUPPORT", sentences=(("a", 1),)),
            "2": ClaimPrediction(verdict="SUPPORT", sentences=(("a", 0),)),
        },
        split="dev",
        system="majority_baseline",
        model_name="majority_train_verdict",
        model_revision="none",
        pair_order="not_applicable",
        baseline_verdict="SUPPORT",
    )
    candidate = evaluate_predictions(
        claims,
        {
            "1": ClaimPrediction(verdict="SUPPORT", sentences=(("a", 0),)),
            "2": ClaimPrediction(verdict="NEI", sentences=()),
        },
        split="dev",
        system="nli",
        model_name="recording",
        model_revision="test-revision",
        pair_order="premise_sentence_hypothesis_claim",
    )
    comparison = compare_verdicts(baseline, candidate, bootstrap_samples=20, permutation_samples=20)
    accuracy = next(check for check in comparison.checks if check.metric == "accuracy")
    deltas = [1.0 - 1.0, 1.0 - 0.0]
    direct = bootstrap_mean_ci(
        deltas,
        samples=20,
        seed="ragbench-stats-v1:0:bootstrap:accuracy",
        confidence=0.95,
    )
    assert accuracy.mean_delta == pytest.approx(direct[0])
    assert accuracy.ci_low == pytest.approx(direct[1])
    assert accuracy.permutation_p_value is not None
    macro = next(check for check in comparison.checks if check.metric == "macro_f1")
    assert macro.permutation_p_value is None
    assert macro.mean_delta == pytest.approx(candidate.macro_f1 - baseline.macro_f1)
    assert comparison.claim_changes
    with pytest.raises(ValueError, match="same split"):
        compare_verdicts(baseline, candidate.model_copy(update={"split": "train"}))


def test_injected_cross_encoder_warns_once_and_checks_shape() -> None:
    config = NliModelConfig(max_length=4, revision="injected-revision")
    model = _FakeNli([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]])
    scorer = CrossEncoderNli(config, model=model, revision="injected-revision")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scores = scorer.score([("sentence", "claim"), ("other", "claim")])
        scorer.score([("sentence", "claim"), ("other", "claim")])
    assert len(caught) == 1
    assert scores[0].entailment == pytest.approx(0.8)
    assert scorer.truncated_pairs == 2
    assert scorer.metadata["nli_revision"] == "injected-revision"
    assert scorer.metadata["nli_pair_order"] == "premise_sentence_hypothesis_claim"
    assert model.kwargs["apply_softmax"] is True
    model.rows = [[0.1, 0.8]]
    with pytest.raises(ValueError, match="three-class"):
        scorer.score([("sentence", "claim")])
    with pytest.raises(ValueError, match="revision"):
        CrossEncoderNli(config, model=model, revision="")
    assert premise_text("Hello\n world") == "Hello  world"


def test_train_yaml_is_the_frozen_document_retriever(tmp_path: Path) -> None:
    experiment = load_verdict_experiment(Path("configs/scifact-verdict-train.yaml"))
    document = yaml.safe_load(
        Path("configs/scifact-bm25-document.yaml").read_text(encoding="utf-8")
    )
    assert experiment.split == "train"
    assert experiment.retrieval.k1 == document["retrieval"]["k1"]
    assert experiment.retrieval.b == document["retrieval"]["b"]
    assert experiment.retrieval.tokenizer == document["retrieval"]["tokenizer"] == "stem"
    assert experiment.retrieval.unit == document["chunking"]["unit"] == "document"
    assert experiment.nli.model_name == "cross-encoder/nli-MiniLM2-L6-H768"
    assert experiment.nli.revision == "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d"
    assert experiment.nli.max_length == 512
    assert list(experiment.grid.doc_k) == [1, 3]
    assert list(experiment.grid.sentence_k) == [1, 2]
    assert list(experiment.grid.min_confidence) == [0.5, 0.7]
    raw = yaml.safe_load(Path("configs/scifact-verdict-train.yaml").read_text(encoding="utf-8"))
    raw["split"] = "dev"
    path = tmp_path / "dev.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen policy"):
        load_verdict_experiment(path)
    raw["policy"] = {"doc_k": 1, "sentence_k": 1, "min_confidence": 0.5}
    raw["baseline_verdict"] = "SUPPORT"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    dev = load_verdict_experiment(path)
    assert dev.policy is not None and dev.policy.doc_k == 1
    with pytest.raises(ValueError):
        load_verdict_experiment(Path("configs/scifact-bm25.yaml"))


def test_report_rejects_a_mismatched_aggregate() -> None:
    claims = [_claim("1", "Alpha beta", "SUPPORT", (("a", 0),))]
    report = evaluate_predictions(
        claims,
        {"1": ClaimPrediction(verdict="SUPPORT", sentences=(("a", 0),))},
        split="train",
        system="nli",
        model_name="recording",
        model_revision="test-revision",
        pair_order="premise_sentence_hypothesis_claim",
    )
    raw = report.model_dump()
    raw["accuracy"] = 0.0
    with pytest.raises(ValueError, match="accuracy"):
        VerdictEvaluation.model_validate(raw)
    raw = report.model_dump()
    raw["claims"][0]["predicted_sentences"] = ["a:1"]
    with pytest.raises(ValueError, match="Claim 1"):
        VerdictEvaluation.model_validate(raw)
    with pytest.raises(ValueError, match="cover"):
        evaluate_predictions(
            claims,
            {},
            split="train",
            system="nli",
            model_name="recording",
            model_revision="test-revision",
            pair_order="premise_sentence_hypothesis_claim",
        )
    with pytest.raises(ValueError, match="no SciFact abstract"):
        build_sentence_jobs(claims, {}, {"1": ("missing",)}, 1)
    with pytest.raises(ValueError, match="one NLI"):
        score_sentence_jobs(
            build_sentence_jobs(
                claims,
                {"a": AbstractDocument(id="a", title="A", sentences=("Alpha.",))},
                {"1": ("a",)},
                1,
            ),
            _Short(),
        )


class _Short:
    def score(self, pairs: list[tuple[str, str]]) -> list[NliScores]:
        return []

    @property
    def metadata(self) -> dict[str, str | int]:
        return {}

    @property
    def truncated_pairs(self) -> int:
        return 0


def test_predictions_for_policy_abstains_without_rows() -> None:
    claims = [_claim("1", "Alpha beta", None)]
    predicted = predictions_for_policy(
        claims, {}, VerdictPolicy(doc_k=1, sentence_k=1, min_confidence=0.5)
    )
    assert predicted["1"].verdict == "NEI"
    with pytest.raises(ValueError, match="empty"):
        premise_text(" \n ")
