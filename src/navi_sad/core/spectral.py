"""Spectral core — dual-path attention comparison primitives.

Provides softmax and linear attention for the last-token query,
GQA head expansion, and per-head cosine distance. These form the
foundation for computing Spectral Attention Divergence (SAD).
"""

import math

import torch
import torch.nn.functional as F


def softmax_attention_last_token(
    q_last: torch.Tensor,
    k_prefix: torch.Tensor,
    v_prefix: torch.Tensor,
) -> torch.Tensor:
    """Standard scaled dot-product attention for the newest token.

    Args:
        q_last: Query for the last generated token. Shape [B, H, 1, D].
        k_prefix: Keys for all prefix positions. Shape [B, H, L, D].
        v_prefix: Values for all prefix positions. Shape [B, H, L, D].

    Returns:
        Attention output. Shape [B, H, 1, D].
    """
    d_k = q_last.shape[-1]
    scale = math.sqrt(d_k)

    # [B, H, 1, D] @ [B, H, D, L] -> [B, H, 1, L]
    scores = torch.matmul(q_last, k_prefix.transpose(-2, -1)) / scale
    weights = torch.softmax(scores, dim=-1)

    # [B, H, 1, L] @ [B, H, L, D] -> [B, H, 1, D]
    return torch.matmul(weights, v_prefix)


def linear_attention_last_token(
    q_last: torch.Tensor,
    k_prefix: torch.Tensor,
    v_prefix: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Linear attention using ELU+1 feature map.

    Accumulates K^T @ V over the full prefix via einsum, yielding
    O(L * D^2) cost per head instead of O(L^2 * D) for softmax.

    Args:
        q_last: Query for the last generated token. Shape [B, H, 1, D].
        k_prefix: Keys for all prefix positions. Shape [B, H, L, D].
        v_prefix: Values for all prefix positions. Shape [B, H, L, D].
        eps: Small constant to prevent division by zero.

    Returns:
        Attention output. Shape [B, H, 1, D].
    """
    # Feature map: elu(x) + 1 ensures non-negative features
    q_mapped = F.elu(q_last) + 1  # [B, H, 1, D]
    k_mapped = F.elu(k_prefix) + 1  # [B, H, L, D]

    # Accumulate: S = K^T @ V via einsum — [B, H, D, D]
    s = torch.einsum("bhld,bhlv->bhdv", k_mapped, v_prefix)

    # Normalizer: sum of mapped keys — [B, H, D]
    z = k_mapped.sum(dim=2)

    # Numerator: Q_mapped @ S — [B, H, 1, D]
    numerator = torch.matmul(q_mapped, s)

    # Denominator: dot(Q_mapped, z) — [B, H, 1, 1]
    denominator = torch.einsum("bhqd,bhd->bhq", q_mapped, z).unsqueeze(-1) + eps

    return numerator / denominator


def expand_kv_heads(kv: torch.Tensor, num_q_heads: int) -> torch.Tensor:
    """Expand KV heads to match query head count for GQA models.

    Uses repeat_interleave to replicate each KV head the required number
    of times. No-op when KV heads already equal query heads.

    Args:
        kv: Key or value tensor. Shape [B, num_kv_heads, L, D].
        num_q_heads: Number of query attention heads.

    Returns:
        Expanded tensor. Shape [B, num_q_heads, L, D].
    """
    num_kv_heads = kv.shape[1]
    if num_kv_heads == num_q_heads:
        return kv

    repeats = num_q_heads // num_kv_heads
    return kv.repeat_interleave(repeats, dim=1)


def per_head_cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine distance between two attention outputs, per head.

    Computes 1 - cosine_similarity for each head, averaged over the batch.

    Args:
        a: First attention output. Shape [B, H, 1, D].
        b: Second attention output. Shape [B, H, 1, D].

    Returns:
        Cosine distance per head, averaged over batch. Shape [H].
    """
    # Squeeze the singleton sequence dim -> [B, H, D]
    a_flat = a.squeeze(2)
    b_flat = b.squeeze(2)

    # Cosine similarity per (batch, head) pair -> [B, H]
    similarity = F.cosine_similarity(a_flat, b_flat, dim=-1)

    # Distance = 1 - similarity, averaged over batch -> [H]
    return (1.0 - similarity).mean(dim=0)
