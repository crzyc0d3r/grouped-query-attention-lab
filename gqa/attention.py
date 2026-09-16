"""Attention with a configurable number of key/value heads.

MHA, GQA and MQA are not three algorithms. They are one algorithm with one knob:
how many key/value heads the query heads have to share.

    n_kv_heads == n_heads   ->  multi-head attention      (every query head has its own KV)
    1 < n_kv_heads < n_heads ->  grouped-query attention  (each group of query heads shares one KV)
    n_kv_heads == 1         ->  multi-query attention     (all query heads share one KV)

So this module implements the general case once, and the three names are just
points on the ``n_kv_heads`` axis. Everything is numpy and everything is float64
by default, because the point is to inspect the arithmetic rather than to be
fast.

Shapes used throughout:

    q         (batch, n_heads,    seq_q, head_dim)
    k, v      (batch, n_kv_heads, seq_k, head_dim)
    output    (batch, n_heads,    seq_q, head_dim)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=axis, keepdims=True)


def repeat_kv(x: np.ndarray, repeats: int) -> np.ndarray:
    """Broadcast each KV head to the query heads in its group.

    This is what a serving engine does at runtime. It is worth being precise
    about what it does *not* do: the repeated heads are views of the same stored
    tensor, so the KV **cache** still holds ``n_kv_heads`` heads. The saving is in
    memory and in bytes read per decode step, not in the arithmetic.
    """

    if repeats == 1:
        return x
    return np.repeat(x, repeats, axis=1)


def causal_mask(seq_q: int, seq_k: int) -> np.ndarray:
    """True where a position must be masked out.

    Handles ``seq_q < seq_k``, which is the decode case: one new query attending
    over the whole cached prefix.
    """

    if seq_q > seq_k:
        raise ValueError(
            "causal attention needs seq_q <= seq_k; got {} > {}".format(seq_q, seq_k)
        )
    offset = seq_k - seq_q
    rows = np.arange(seq_q)[:, None]
    cols = np.arange(seq_k)[None, :]
    return cols > (rows + offset)


def attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    causal: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Scaled dot-product attention for any KV-head count.

    Returns ``(output, weights)``.
    """

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k and v must all be 4-D (batch, heads, seq, head_dim)")

    n_heads = q.shape[1]
    n_kv_heads = k.shape[1]
    if k.shape != v.shape:
        raise ValueError("k and v must have the same shape")
    if n_heads % n_kv_heads != 0:
        raise ValueError(
            "n_heads ({}) must be divisible by n_kv_heads ({})".format(n_heads, n_kv_heads)
        )
    head_dim = q.shape[3]
    if k.shape[3] != head_dim:
        raise ValueError("q and k must share head_dim")

    group_size = n_heads // n_kv_heads
    k_full = repeat_kv(k, group_size)
    v_full = repeat_kv(v, group_size)

    scores = np.matmul(q, np.swapaxes(k_full, -1, -2)) / np.sqrt(head_dim)

    if causal:
        mask = causal_mask(q.shape[2], k_full.shape[2])
        scores = np.where(mask, -np.inf, scores)

    weights = softmax(scores, axis=-1)
    return np.matmul(weights, v_full), weights


def pool_kv(x: np.ndarray, n_kv_heads: int) -> np.ndarray:
    """Collapse a full set of KV heads down to ``n_kv_heads`` by averaging groups.

    This is how the GQA paper produces a grouped checkpoint from a multi-head one:
    mean-pool the key and value projections within each group, then continue
    pre-training briefly to recover. Averaging alone is the crude version, and
    running it here is instructive - it shows directly how much attention output
    moves as the groups get wider, with no training to paper over the difference.
    """

    batch, n_heads, seq, head_dim = x.shape
    if n_heads % n_kv_heads != 0:
        raise ValueError(
            "n_heads ({}) must be divisible by n_kv_heads ({})".format(n_heads, n_kv_heads)
        )
    group_size = n_heads // n_kv_heads
    grouped = x.reshape(batch, n_kv_heads, group_size, seq, head_dim)
    return grouped.mean(axis=2)


def random_qkv(
    batch: int = 1,
    n_heads: int = 8,
    seq: int = 16,
    head_dim: int = 32,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic random Q, K, V in full multi-head layout."""

    rng = np.random.default_rng(seed)
    shape = (batch, n_heads, seq, head_dim)
    scale = 1.0 / np.sqrt(head_dim)
    return (
        rng.normal(scale=scale, size=shape),
        rng.normal(scale=scale, size=shape),
        rng.normal(scale=scale, size=shape),
    )


def fidelity_sweep(
    batch: int = 1,
    n_heads: int = 8,
    seq: int = 16,
    head_dim: int = 32,
    seed: int = 0,
):
    """How far does attention output drift as KV heads are merged?

    Yields ``(n_kv_heads, relative_l2_error)`` for every divisor of ``n_heads``,
    measured against the full multi-head output on the same inputs. The error at
    ``n_kv_heads == n_heads`` is exactly zero, which is the check that MHA really
    is the ``n_kv_heads == n_heads`` case of this code path.
    """

    q, k, v = random_qkv(batch, n_heads, seq, head_dim, seed)
    reference, _ = attention(q, k, v)
    denominator = np.linalg.norm(reference)

    results = []
    for n_kv_heads in sorted({d for d in range(1, n_heads + 1) if n_heads % d == 0}):
        pooled_k = pool_kv(k, n_kv_heads)
        pooled_v = pool_kv(v, n_kv_heads)
        output, _ = attention(q, pooled_k, pooled_v)
        error = float(np.linalg.norm(output - reference) / denominator)
        results.append((n_kv_heads, error))
    return results
