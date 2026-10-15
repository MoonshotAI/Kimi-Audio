"""Compare the non-Flash path with an explicit head-wise reference."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


@pytest.fixture
def dit(monkeypatch):
    # Unused extension placeholders expose the baseline's missing fallback.
    flash = types.ModuleType("flash_attn")

    def unused_flash(*args, **kwargs):
        raise AssertionError("The non-Flash path must not call FlashAttention")

    flash.flash_attn_varlen_func = unused_flash
    flash.flash_attn_varlen_qkvpacked_func = unused_flash
    monkeypatch.setitem(sys.modules, "flash_attn", flash)
    spec = importlib.util.spec_from_file_location(
        "dit_block",
        Path(__file__).resolve().parents[1]
        / "kimia_infer/models/detokenizer/flow_matching/dit_block.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def explicit_attention(layer, state, lengths=None, rotary=None, previous=None):
    bsz, n, _ = state.shape
    q, k, v = (
        layer.qkv(state).reshape(bsz, n, 3, layer.num_heads, layer.head_dim).unbind(2)
    )
    q, k = layer.q_norm(q), layer.k_norm(k)
    if rotary is not None:

        def rotate(tensor):
            pairs = tensor.reshape(*tensor.shape[:-1], -1, 2)
            c, s = rotary.real.unsqueeze(2), rotary.imag.unsqueeze(2)
            a, b = pairs.unbind(-1)
            return torch.stack((a * c - b * s, a * s + b * c), -1).flatten(-2)

        q, k = rotate(q), rotate(k)
    if previous is not None:
        k = torch.cat((previous["prev_k"], k), dim=1)
        v = torch.cat((previous["prev_v"], v), dim=1)
    batches = []
    for i in range(bsz):
        length = n if lengths is None else int(lengths[i])
        heads = []
        for j in range(layer.num_heads):
            key_len = k.shape[1] if lengths is None else length
            scores = q[i, :length, j] @ k[i, :key_len, j].T / layer.head_dim**0.5
            heads.append(scores.softmax(-1) @ v[i, :key_len, j])
        row = torch.stack(heads, 1).reshape(length, -1)
        batches.append(torch.nn.functional.pad(row, (0, 0, 0, n - length)))
    return layer.proj(torch.stack(batches)), k, v


def run(layer, x, lengths=None, rotary=None, cache=None):
    return layer(
        x,
        seq_len=lengths,
        cu_seqlens=None,
        max_seqlen=x.shape[1],
        cu_seqlens_k=None,
        max_seqlen_k=x.shape[1],
        rotary_pos_emb=rotary,
        incremental_state=cache,
        nopadding=lengths is None,
    )


@pytest.mark.parametrize("qk_norm", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_output_and_gradients_match_explicit_heads(dit, qk_norm, dtype):
    torch.manual_seed(71)
    layer = (
        dit.Attention(
            8, num_heads=2, qkv_bias=True, qk_norm=qk_norm, flash_attention=False
        )
        .to(dtype)
        .eval()
    )
    x = torch.randn(2, 3, 8, dtype=dtype, requires_grad=True)
    result = run(layer, x)
    reference, _, _ = explicit_attention(layer, x)
    torch.testing.assert_close(result, reference)
    args = (x, layer.qkv.weight, layer.proj.weight)
    grads = torch.autograd.grad(result.square().sum(), args, retain_graph=True)
    refs = torch.autograd.grad(reference.square().sum(), args)
    for grad, expected in zip(grads, refs):
        torch.testing.assert_close(grad, expected)


def test_padded_batch_ignores_padding_values_and_gradients(dit):
    layer = (
        dit.Attention(8, num_heads=2, qkv_bias=True, flash_attention=False)
        .double()
        .eval()
    )
    lengths = torch.tensor([3, 1])
    x = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
    result = run(layer, x, lengths)
    reference, _, _ = explicit_attention(layer, x, lengths)
    torch.testing.assert_close(result, reference)
    changed = x.detach().clone()
    changed[1, 1:] = 1e8
    torch.testing.assert_close(run(layer, changed, lengths), result)
    (grad,) = torch.autograd.grad(result.sum(), x)
    assert torch.count_nonzero(grad[1, 1:]) == 0


def test_rotary_and_streaming_cache_match_explicit_reference(dit):
    layer = (
        dit.Attention(
            8, num_heads=2, qkv_bias=True, qk_norm=True, flash_attention=False
        )
        .double()
        .eval()
    )
    x = torch.randn(2, 3, 8, dtype=torch.float64)
    phase = torch.randn(2, 3, 2, dtype=torch.float64)
    rotary = torch.polar(torch.ones_like(phase), phase)
    cache = {
        "prev_k": torch.randn(2, 4, 2, 4, dtype=torch.float64),
        "prev_v": torch.randn(2, 4, 2, 4, dtype=torch.float64),
    }
    result = run(layer, x, rotary=rotary, cache=cache)
    reference, keys, values = explicit_attention(
        layer, x, rotary=rotary, previous=cache
    )
    torch.testing.assert_close(result, reference)
    torch.testing.assert_close(cache["cur_k"], keys)
    torch.testing.assert_close(cache["cur_v"], values)
    assert keys.shape[1] == 7


def test_eval_disables_attention_dropout(dit):
    layer = dit.Attention(8, num_heads=2, attn_drop=0.7, flash_attention=False).eval()
    x = torch.randn(2, 3, 8)
    torch.testing.assert_close(run(layer, x), run(layer, x))


def test_optional_flash_import_is_not_required_for_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "flash_attn", None)
    spec = importlib.util.spec_from_file_location(
        "dit_without_flash",
        Path(__file__).resolve().parents[1]
        / "kimia_infer/models/detokenizer/flow_matching/dit_block.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    layer = module.Attention(8, num_heads=2, flash_attention=False).eval()
    assert run(layer, torch.randn(1, 3, 8)).shape == (1, 3, 8)
    with pytest.raises(ImportError, match="flash_attn is required"):
        module.Attention(8, num_heads=2, flash_attention=True)


def test_empty_padded_sequence_has_finite_gradients(dit):
    layer = dit.Attention(8, num_heads=2, flash_attention=False).double().eval()
    x = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
    result = run(layer, x, lengths=torch.tensor([3, 0]))
    torch.testing.assert_close(result[1], layer.proj.bias.expand(3, -1))
    gradients = torch.autograd.grad(result.sum(), (x, layer.qkv.weight))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert torch.count_nonzero(gradients[0][1]) == 0
