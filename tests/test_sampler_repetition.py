"""CPU regression tests; no model weights, transformers, or audio dependencies.

Run from the repository root: python -m pytest -q tests/test_sampler_repetition.py
SAMPLER_SOURCE can point to an unmodified sampler for before/after comparisons.
"""
import importlib.util
import os
from pathlib import Path

import pytest
import torch


SOURCE = Path(os.environ.get(
    "SAMPLER_SOURCE",
    Path(__file__).resolve().parents[1] / "kimia_infer/utils/sampler.py",
))
SPEC = importlib.util.spec_from_file_location("sampler_under_test", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture(params=["audio", "text"])
def kind(request):
    return request.param


def sampler(kind, penalty=2.0, window=4, temperature=0.0, top_k=0):
    obj = MODULE.KimiASampler(
        audio_top_k=top_k, audio_temperature=temperature,
        audio_repetition_penalty=penalty, audio_repetition_window_size=window,
        text_top_k=top_k, text_temperature=temperature,
        text_repetition_penalty=penalty, text_repetition_window_size=window,
    )
    return getattr(obj, f"sample_{kind}_logits")


def reference(logits, history, penalty, window):
    """Independent per-row oracle: a set avoids compounding duplicate tokens."""
    out = (logits[:, -1] if logits.ndim == 3 else logits).clone()
    if penalty <= 1 or history is None or window <= 0:
        return out
    if history.ndim == 1:
        history = history.unsqueeze(0)
    for row in range(out.size(0)):
        recent = history[0 if history.size(0) == 1 else row, -window:]
        for token in set(recent.tolist()):
            value = out[row, token]
            out[row, token] = value * penalty if value < 0 else value / penalty
    return out


@pytest.mark.parametrize("length", [1, 3, 4, 5])
def test_penalty_applies_before_at_and_after_window(kind, length):
    logits = torch.tensor([[4.0, 3.0, -2.0]])
    history = torch.zeros(length, dtype=torch.long)
    assert sampler(kind)(logits, history).tolist() == [1]


def test_shared_history_preserves_every_batch_row(kind):
    logits = torch.tensor([[4.0, 3.0, -2.0], [4.0, 1.0, 3.0]])
    history = torch.zeros(5, dtype=torch.long)
    assert sampler(kind)(logits, history).tolist() == [1, 2]


@pytest.mark.parametrize("history", [
    torch.tensor([[0], [1]]),
    torch.tensor([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]),
])
def test_per_row_history_does_not_cross_contaminate(kind, history):
    logits = torch.tensor([[4.0, 3.0, 0.0], [3.0, 4.0, 0.0]])
    assert sampler(kind)(logits, history).tolist() == [1, 0]


def test_batch_larger_than_window(kind):
    logits = torch.tensor([[4.0, 3.0]]).repeat(8, 1)
    history = torch.zeros(8, 5, dtype=torch.long)
    result = sampler(kind)(logits, history)
    assert result.shape == (8,)
    assert result.tolist() == [1] * 8


@pytest.mark.parametrize("sequence", [False, True])
def test_caller_logits_are_not_mutated(kind, sequence):
    logits = torch.tensor([[8.0, 3.0, -1.0]])
    if sequence:
        logits = logits.unsqueeze(1).repeat(1, 3, 1)
    before = logits.clone()
    history = torch.tensor([0, 0, 0, 0, 0])
    sample = sampler(kind)
    assert sample(logits, history).tolist() == [0]
    torch.testing.assert_close(logits, before, rtol=0, atol=0)
    assert sample(logits, history).tolist() == [0]


def test_duplicate_ids_are_penalized_once(kind):
    logits = torch.tensor([[8.0, 3.0]])
    history = torch.tensor([0, 0, 0, 0, 0])
    assert sampler(kind)(logits, history).tolist() == [0]


def test_negative_scores_are_multiplied(kind):
    logits = torch.tensor([[-1.0, -1.5, -3.0]])
    history = torch.tensor([0, 0, 0, 0, 0])
    assert sampler(kind)(logits, history).tolist() == [1]


def test_only_recent_window_is_used(kind):
    logits = torch.tensor([[4.0, 3.0, 1.0]])
    history = torch.tensor([0, 1, 1, 1, 1])
    assert sampler(kind)(logits, history).tolist() == [0]


@pytest.mark.parametrize("window", [0, -1])
def test_nonpositive_window_disables_penalty(kind, window):
    assert sampler(kind, window=window)(
        torch.tensor([[4.0, 3.0]]), torch.tensor([0, 0, 0])
    ).tolist() == [0]


@pytest.mark.parametrize("history", [None, torch.empty(0, dtype=torch.long),
                                     torch.empty(2, 0, dtype=torch.long)])
def test_no_history_preserves_greedy_output(kind, history):
    logits = torch.tensor([[4.0, 3.0], [3.0, 4.0]])
    assert sampler(kind)(logits, history).tolist() == [0, 1]


def test_unit_penalty_preserves_output(kind):
    logits = torch.tensor([[4.0, 3.0], [3.0, 4.0]])
    assert sampler(kind, penalty=1)(logits, torch.zeros(9, dtype=torch.long)).tolist() == [0, 1]


@pytest.mark.parametrize("history", [torch.zeros(3, 6, dtype=torch.long),
                                     torch.zeros(2, 1, 6, dtype=torch.long)])
def test_invalid_history_shape_is_explicit(kind, history):
    with pytest.raises(ValueError, match="recent_tokens"):
        sampler(kind)(torch.ones(2, 3), history)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch", [1, 2, 7])
def test_randomized_greedy_matches_independent_oracle(kind, dtype, batch):
    generator = torch.Generator().manual_seed(881 + batch)
    for history_length in (1, 4, 9):
        logits = torch.randn(batch, 3, 17, generator=generator).to(dtype)
        history = torch.randint(17, (batch, history_length), generator=generator)
        expected = reference(logits, history, penalty=1.7, window=4).argmax(-1)
        actual = sampler(kind, penalty=1.7)(logits, history)
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("temperature,top_k", [(1.0, 0), (0.8, 3)])
def test_sampling_receives_reference_distribution(kind, monkeypatch, temperature, top_k):
    logits = torch.tensor([[4.0, 3.0, 1.0, -1.0], [-0.5, -1.0, 1.0, 2.0]])
    history = torch.tensor([[0, 0, 0], [3, 3, 3]])
    corrected = reference(logits, history, 2.0, 4)
    # Match the existing temperature/top-k implementation; do not conflate
    # this repetition-penalty regression with the independent temperature PR.
    weights = (corrected.log_softmax(-1) / temperature).exp()
    if top_k:
        weights = weights.topk(top_k, dim=-1).values
    expected_probs = weights / weights.sum(-1, keepdim=True)
    calls = []

    def capture(probs, num_samples):
        calls.append(probs / probs.sum(-1, keepdim=True))
        return probs.argmax(-1, keepdim=True)

    monkeypatch.setattr(torch, "multinomial", capture)
    result = sampler(kind, temperature=temperature, top_k=top_k)(logits, history)
    assert result.shape == (2,)
    assert len(calls) == 1
    torch.testing.assert_close(calls[0], expected_probs)


@pytest.mark.skipif(not torch.cuda.is_available() or os.environ.get("CPU_REGRESSION_ONLY") == "1",
    reason="CUDA hardware unavailable or CPU-only regression run")
def test_cuda_with_cpu_history(kind):
    logits = torch.tensor([[4.0, 3.0], [3.0, 4.0]], device="cuda")
    history = torch.tensor([[0], [1]])
    actual = sampler(kind)(logits, history)
    assert actual.device.type == "cuda"
    assert actual.tolist() == [1, 0]
