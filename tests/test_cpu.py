"""Small deterministic CPU checks; no downloaded corpus or snapshot is required."""

import math

import pytest
import torch
import torch.nn.functional as F

from .adapters import (
    get_adamw_cls,
    run_cross_entropy,
    run_embedding,
    run_get_lr_cosine_schedule,
    run_linear,
    run_multihead_self_attention,
    run_multihead_self_attention_with_rope,
    run_rmsnorm,
    run_rope,
    run_scaled_dot_product_attention,
    run_softmax,
    run_swiglu,
    run_train_bpe,
    run_transformer_block,
    run_transformer_lm,
)


def test_linear_and_embedding():
    weights = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    inputs = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4)
    torch.testing.assert_close(run_linear(4, 3, weights, inputs), F.linear(inputs, weights))
    ids = torch.tensor([[0, 2], [1, 0]])
    torch.testing.assert_close(run_embedding(3, 4, weights, ids), F.embedding(ids, weights))


def test_swiglu():
    torch.manual_seed(4)
    x = torch.randn(2, 3, 4)
    w1, w2, w3 = torch.randn(6, 4), torch.randn(4, 6), torch.randn(6, 4)
    expected = F.linear(F.silu(F.linear(x, w1)) * F.linear(x, w3), w2)
    torch.testing.assert_close(run_swiglu(4, 6, w1, w2, w3, x), expected)


def test_rmsnorm_import_and_output():
    x = torch.tensor([[1.0, 2.0, -1.0, 0.0]])
    weight = torch.tensor([1.0, 2.0, 3.0, 4.0])
    expected = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5) * weight
    torch.testing.assert_close(run_rmsnorm(4, 1e-5, weight, x), expected)


def test_softmax_and_cross_entropy():
    logits = torch.tensor([[1000.0, 1001.0, 999.0], [-1.0, 2.0, 3.0]])
    torch.testing.assert_close(run_softmax(logits, -1), F.softmax(logits, -1))
    targets = torch.tensor([1, 2])
    torch.testing.assert_close(run_cross_entropy(logits, targets), F.cross_entropy(logits, targets))


def test_attention_masked_future():
    q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    k = q.clone()
    v = torch.tensor([[[2.0, 3.0], [7.0, 11.0]]])
    mask = torch.triu(torch.ones(2, 2, dtype=torch.bool), diagonal=1)
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=~mask)
    torch.testing.assert_close(run_scaled_dot_product_attention(q, k, v, mask), expected)
    assert torch.equal(run_scaled_dot_product_attention(q, k, v, mask)[0, 0], v[0, 0])


def test_multihead_attention_without_rope():
    torch.manual_seed(6)
    x = torch.randn(1, 3, 4)
    eye = torch.eye(4)
    output = run_multihead_self_attention(4, 2, eye, eye, eye, eye, x)
    heads = x.reshape(1, 3, 2, 2).transpose(1, 2)
    mask = torch.ones(3, 3, dtype=torch.bool).tril()
    expected = F.scaled_dot_product_attention(heads, heads, heads, attn_mask=mask)
    torch.testing.assert_close(output, expected.transpose(1, 2).reshape(1, 3, 4))


def test_rope_zero_position_and_rotation():
    x = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    actual = run_rope(2, 10000.0, 4, x, torch.tensor([[0, 1]]))
    expected = torch.tensor([[[1.0, 0.0], [math.cos(1), math.sin(1)]]])
    torch.testing.assert_close(actual, expected)


def test_multihead_attention_with_rope():
    torch.manual_seed(7)
    x = torch.randn(1, 3, 4)
    eye = torch.eye(4)
    positions = torch.arange(3)
    actual = run_multihead_self_attention_with_rope(
        4, 2, 4, 10000.0, eye, eye, eye, eye, x, positions,
    )
    heads = x.reshape(1, 3, 2, 2).transpose(1, 2)
    rotated = run_rope(2, 10000.0, 4, heads, positions)
    expected = F.scaled_dot_product_attention(
        rotated, rotated, heads, attn_mask=torch.ones(3, 3, dtype=torch.bool).tril(),
    )
    torch.testing.assert_close(actual, expected.transpose(1, 2).reshape(1, 3, 4))


def test_block_and_lm_adapters_with_generated_weights():
    torch.manual_seed(8)
    block_weights = {
        "ln1.weight": torch.ones(4), "ln2.weight": torch.ones(4),
        "attn.q_proj.weight": torch.zeros(4, 4),
        "attn.k_proj.weight": torch.zeros(4, 4),
        "attn.v_proj.weight": torch.zeros(4, 4),
        "attn.output_proj.weight": torch.zeros(4, 4),
        "ffn.w1.weight": torch.zeros(8, 4),
        "ffn.w2.weight": torch.zeros(4, 8),
        "ffn.w3.weight": torch.zeros(8, 4),
    }
    x = torch.randn(1, 3, 4)
    torch.testing.assert_close(
        run_transformer_block(4, 2, 8, 4, 10000.0, block_weights, x), x,
    )
    embeddings = torch.randn(9, 4)
    head = torch.randn(9, 4)
    weights = {
        "token_embeddings.weight": embeddings,
        "ln_final.weight": torch.ones(4),
        "lm_head.weight": head,
        **{f"layers.0.{key}": value for key, value in block_weights.items()},
    }
    ids = torch.tensor([[1, 2, 3]])
    output = run_transformer_lm(9, 4, 4, 1, 2, 8, 10000.0, weights, ids)
    embedded = F.embedding(ids, embeddings)
    expected = F.linear(
        embedded * torch.rsqrt(embedded.square().mean(-1, keepdim=True) + 1e-5),
        head,
    )
    torch.testing.assert_close(output, expected)


def test_transformer_forward_and_causal_prefix():
    from cs336_basics.Transformer import Transformer

    torch.manual_seed(1)
    model = Transformer(
        dff=8, d_model=4, n_heads=2, vocab_size=12,
        context_length=4, num_layers=1, theta=10000,
        token_positions=torch.arange(4),
    )
    first = torch.tensor([[1, 2, 3, 4]])
    second = torch.tensor([[1, 2, 7, 9]])
    with torch.no_grad():
        a, b = model(first), model(second)
    assert a.shape == (1, 4, 12)
    torch.testing.assert_close(a[:, :2], b[:, :2])


@pytest.mark.parametrize("step,expected", [
    (0, 0.0), (2, 0.5), (4, 1.0), (8, 0.55), (12, 0.1), (15, 0.1),
])
def test_cosine_schedule(step, expected):
    assert run_get_lr_cosine_schedule(step, 1.0, 0.1, 4, 12) == pytest.approx(expected)


def test_adamw_matches_reference_updates():
    value = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    reference = torch.nn.Parameter(value.detach().clone())
    optimizer = get_adamw_cls()([value], lambd=0.01, lr=0.001)
    baseline = torch.optim.AdamW([reference], weight_decay=0.01, lr=0.001)
    for grad in ([0.3, -0.4], [-0.1, 0.5], [0.2, 0.1]):
        value.grad = torch.tensor(grad)
        reference.grad = torch.tensor(grad)
        optimizer.step()
        baseline.step()
    torch.testing.assert_close(value, reference, atol=1e-6, rtol=1e-6)


def test_train_bpe_on_generated_corpus(tmp_path):
    corpus = tmp_path / "tiny.txt"
    corpus.write_text("aa aa<|endoftext|>aa aa<|endoftext|>aa aa", encoding="utf-8")
    vocab, merges = run_train_bpe(corpus, 258, ["<|endoftext|>"])
    assert vocab[256] == b"<|endoftext|>"
    assert vocab[257] == b"aa"
    assert merges == [(b"a", b"a")]
