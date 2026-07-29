import numpy as np
import pytest
import torch

from urban_s2.pa_mcsp import (
    SafeBinaryGate,
    SafeResidualReranker,
    build_multicue_signals,
    candidate_errors_km,
    oracle_summary,
    paired_union_oracle,
    reranking_loss,
    topk_cosine,
)


def test_topk_cosine_returns_exact_sorted_candidates() -> None:
    gallery = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]],
        dtype=torch.float32,
    )
    query = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    result = topk_cosine(query, gallery, top_k=2, batch_size=1)
    assert result.indices.tolist() == [[0, 2]]
    assert result.scores[0, 0] >= result.scores[0, 1]


def test_candidate_oracle_uses_best_natural_candidate() -> None:
    truth = np.array([[0.0, 0.0], [10.0, 10.0]])
    candidates = np.array(
        [
            [[0.0, 1.0], [0.0, 0.0]],
            [[11.0, 10.0], [20.0, 20.0]],
        ]
    )
    errors = candidate_errors_km(truth, candidates)
    summary = oracle_summary(errors)
    assert summary["queries"] == 2
    assert summary["candidates_per_query"] == 2
    assert summary["accuracy_1km"] == pytest.approx(0.5)


def test_union_oracle_never_worse_than_either_pool() -> None:
    prototype = np.array([[100.0, 200.0], [20.0, 40.0]])
    retrieval = np.array([[10.0, 30.0], [50.0, 60.0]])
    union = paired_union_oracle(prototype, retrieval, top_k=2)
    assert union["median_error_km"] == pytest.approx(15.0)
    assert union["candidates_per_query"] == 4


def test_multicue_reranker_starts_exactly_at_anchor() -> None:
    prototype_scores = np.array([[0.9, 0.8]], dtype=np.float32)
    retrieval_scores = np.array([[0.85, 0.7]], dtype=np.float32)
    prototype_points = np.array([[[0.0, 0.0], [0.0, 1.0]]])
    retrieval_points = np.array([[[0.0, 0.1], [1.0, 1.0]]])
    density = np.ones((1, 2), dtype=np.float32)
    signals, anchor = build_multicue_signals(
        prototype_scores,
        retrieval_scores,
        prototype_points,
        retrieval_points,
        density,
        density,
    )
    model = SafeResidualReranker(signal_dimension=signals.shape[-1])
    logits, residual = model(
        torch.from_numpy(signals),
        torch.from_numpy(anchor),
    )
    assert torch.equal(logits, torch.from_numpy(anchor))
    assert torch.count_nonzero(residual) == 0


def test_reranking_loss_is_finite_without_gold_injection() -> None:
    logits = torch.tensor([[1.0, 0.0]], requires_grad=True)
    residual = torch.zeros_like(logits, requires_grad=True)
    errors = torch.tensor([[100.0, 10.0]])
    loss = reranking_loss(logits, residual, errors)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None


def test_binary_gate_starts_at_raw_similarity_decision() -> None:
    model = SafeBinaryGate(signal_dimension=4)
    signals = torch.randn(3, 4)
    anchor = torch.tensor([-1.0, 0.5, 2.0])
    logits, residual = model(signals, anchor)
    assert torch.equal(logits, anchor)
    assert torch.count_nonzero(residual) == 0
