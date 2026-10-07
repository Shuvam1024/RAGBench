import pytest

from ragstat.evaluation.train_selection import (
    choose_candidate_k,
    choose_dense_weight,
    selection_key,
)


def test_train_tie_break_prefers_smaller_chunks_then_whitespace() -> None:
    tied = {
        "metrics": {"ndcg@10": 0.5},
    }
    rows = [
        {**tied, "tokenizer": "stem", "chunk_size": 120},
        {**tied, "tokenizer": "whitespace", "chunk_size": 240},
        {**tied, "tokenizer": "whitespace", "chunk_size": 120},
        {**tied, "tokenizer": "stem", "chunk_size": 60, "metrics": {"ndcg@10": 0.4}},
    ]
    ordered = sorted(rows, key=selection_key)
    assert [(row["tokenizer"], row["chunk_size"]) for row in ordered] == [
        ("whitespace", 120),
        ("stem", 120),
        ("whitespace", 240),
        ("stem", 60),
    ]
    with pytest.raises(TypeError):
        selection_key({"chunk_size": "120", "metrics": {"ndcg@10": 0.5}})
    with pytest.raises(TypeError):
        selection_key({"chunk_size": 120, "metrics": {"ndcg@10": True}})


def test_dense_weight_tie_prefers_the_smaller_weight() -> None:
    rows = [(1.0, 0.70), (0.5, 0.75), (0.75, 0.75), (0.0, 0.60), (0.25, 0.75)]
    assert choose_dense_weight(rows) == 0.25
    with pytest.raises(ValueError, match="at least one"):
        choose_dense_weight(())
    with pytest.raises(TypeError):
        choose_dense_weight([(True, 0.5)])


def test_candidate_depth_tie_prefers_the_smaller_depth() -> None:
    rows = [(100, 0.70), (50, 0.75), (20, 0.75)]
    assert choose_candidate_k(rows) == 20
    with pytest.raises(ValueError, match="at least one"):
        choose_candidate_k(())
    with pytest.raises(TypeError):
        choose_candidate_k([(True, 0.5)])
