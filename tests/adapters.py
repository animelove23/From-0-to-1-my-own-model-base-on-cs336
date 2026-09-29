"""Compatibility layer for tests of the current cs336_basics implementation."""

from __future__ import annotations

from torch import Tensor


def run_linear(d_in: int, d_out: int, weights: Tensor, in_features: Tensor) -> Tensor:
    from cs336_basics.linear_model import LinearModel

    layer = LinearModel(d_in, d_out)
    layer.load_state_dict({"weight": weights})
    return layer(in_features)


def run_embedding(vocab_size: int, d_model: int, weights: Tensor, token_ids: Tensor) -> Tensor:
    from cs336_basics.embadding import Embedding

    layer = Embedding(vocab_size, d_model)
    layer.load_state_dict({"embedding_matrix": weights})
    return layer(token_ids)


def run_swiglu(
    d_model: int, d_ff: int, w1_weight: Tensor, w2_weight: Tensor,
    w3_weight: Tensor, in_features: Tensor,
) -> Tensor:
    from cs336_basics.FFN import SwiGLU

    layer = SwiGLU(d_model, d_ff)
    layer.load_state_dict({
        "w1_weight": w1_weight,
        "w2_weight": w2_weight,
        "w3_weight": w3_weight,
    })
    return layer(in_features)


def run_scaled_dot_product_attention(
    Q: Tensor, K: Tensor, V: Tensor, mask: Tensor | None = None,
) -> Tensor:
    from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention

    # The current implementation masks positions where mask is True.
    return scaled_dot_product_attention(Q, K, V, mask)


def run_multihead_self_attention(
    d_model: int, num_heads: int, q_proj_weight: Tensor, k_proj_weight: Tensor,
    v_proj_weight: Tensor, o_proj_weight: Tensor, in_features: Tensor,
) -> Tensor:
    from cs336_basics.multihead_self_attention import MultiHeadSelfAttention

    layer = MultiHeadSelfAttention(d_model, num_heads, open_pos=False)
    layer.load_state_dict({
        "w_q": q_proj_weight, "w_k": k_proj_weight,
        "w_v": v_proj_weight, "w_o": o_proj_weight,
    })
    return layer(in_features)


def run_multihead_self_attention_with_rope(
    d_model: int, num_heads: int, max_seq_len: int, theta: float,
    q_proj_weight: Tensor, k_proj_weight: Tensor, v_proj_weight: Tensor,
    o_proj_weight: Tensor, in_features: Tensor, token_positions: Tensor | None = None,
) -> Tensor:
    from cs336_basics.multihead_self_attention import MultiHeadSelfAttention
    import torch

    if token_positions is None:
        token_positions = torch.arange(in_features.shape[-2], device=in_features.device)
    layer = MultiHeadSelfAttention(
        d_model, num_heads, open_pos=True, theta=theta,
        max_seq_len=max_seq_len, token_positions=token_positions,
    )
    layer.load_state_dict({
        "w_q": q_proj_weight, "w_k": k_proj_weight,
        "w_v": v_proj_weight, "w_o": o_proj_weight,
    })
    return layer(in_features)


def run_rope(
    d_k: int, theta: float, max_seq_len: int,
    in_query_or_key: Tensor, token_positions: Tensor,
) -> Tensor:
    from cs336_basics.RoPE import RoPE

    return RoPE(theta, d_k, max_seq_len)(in_query_or_key, token_positions)


def run_rmsnorm(d_model: int, eps: float, weights: Tensor, in_features: Tensor) -> Tensor:
    from cs336_basics.RMSNorm import RMSNorm

    layer = RMSNorm(d_model, eps)
    layer.load_state_dict({"g": weights})
    return layer(in_features)


def run_transformer_block(
    d_model: int, num_heads: int, d_ff: int, max_seq_len: int,
    theta: float, weights: dict[str, Tensor], in_features: Tensor,
) -> Tensor:
    import torch
    from cs336_basics.transformer_block import TransformerBlock

    layer = TransformerBlock(
        dff=d_ff, d_model=d_model, n_heads=num_heads, open_pos=True,
        theta=theta, max_seq_len=max_seq_len,
        token_positions=torch.arange(in_features.shape[-2], device=in_features.device),
    ).to(device=in_features.device, dtype=in_features.dtype)
    translation = {
        "rmsnorm1.g": "ln1.weight",
        "rmsnorm2.g": "ln2.weight",
        "causal_att.w_q": "attn.q_proj.weight",
        "causal_att.w_k": "attn.k_proj.weight",
        "causal_att.w_v": "attn.v_proj.weight",
        "causal_att.w_o": "attn.output_proj.weight",
        "ffn.w1_weight": "ffn.w1.weight",
        "ffn.w2_weight": "ffn.w2.weight",
        "ffn.w3_weight": "ffn.w3.weight",
    }
    layer.load_state_dict({key: weights[value] for key, value in translation.items()})
    return layer(in_features)


def run_transformer_lm(
    vocab_size: int, context_length: int, d_model: int, num_layers: int,
    num_heads: int, d_ff: int, rope_theta: float,
    weights: dict[str, Tensor], in_indices: Tensor,
) -> Tensor:
    import torch
    from cs336_basics.Transformer import Transformer

    model = Transformer(
        dff=d_ff, d_model=d_model, n_heads=num_heads, vocab_size=vocab_size,
        context_length=context_length, num_layers=num_layers, theta=rope_theta,
        token_positions=torch.arange(in_indices.shape[-1], device=in_indices.device),
    ).to(in_indices.device)
    state = {
        "embedding.embedding_matrix": weights["token_embeddings.weight"],
        "RMSNorm.g": weights["ln_final.weight"],
        "linear.weight": weights["lm_head.weight"],
    }
    mapping = {
        "rmsnorm1.g": "ln1.weight",
        "rmsnorm2.g": "ln2.weight",
        "causal_att.w_q": "attn.q_proj.weight",
        "causal_att.w_k": "attn.k_proj.weight",
        "causal_att.w_v": "attn.v_proj.weight",
        "causal_att.w_o": "attn.output_proj.weight",
        "ffn.w1_weight": "ffn.w1.weight",
        "ffn.w2_weight": "ffn.w2.weight",
        "ffn.w3_weight": "ffn.w3.weight",
    }
    for i in range(num_layers):
        state.update({
            f"transformer_layers.{i}.{name}": weights[f"layers.{i}.{reference}"]
            for name, reference in mapping.items()
        })
    model.load_state_dict(state)
    return model(in_indices)


def run_softmax(in_features: Tensor, dim: int) -> Tensor:
    from cs336_basics.softmax import softmax

    return softmax(in_features, dim)


def run_cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    from cs336_basics.cross_entropy import cross_entropy

    return cross_entropy(inputs, targets)


def get_adamw_cls():
    from cs336_basics.AdamW import AdamW

    return AdamW


def run_get_lr_cosine_schedule(
    it: int, max_learning_rate: float, min_learning_rate: float,
    warmup_iters: int, cosine_cycle_iters: int,
) -> float:
    from cs336_basics.learning_rate_schedule import learning_rate_schedule

    return learning_rate_schedule(
        it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters,
    )


def run_train_bpe(input_path, vocab_size: int, special_tokens: list[str]):
    from cs336_basics.BPE import bpe_tokenization

    return bpe_tokenization(input_path, vocab_size, special_tokens)
