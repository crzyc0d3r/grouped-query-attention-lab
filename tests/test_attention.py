"""Tests for the attention kernel shared by all three layouts."""

import numpy as np
import pytest

from gqa.attention import (
    attention,
    causal_mask,
    fidelity_sweep,
    pool_kv,
    random_qkv,
    repeat_kv,
    softmax,
)


def reference_multi_head(q, k, v):
    """A deliberately slow per-head implementation to check the batched one."""

    batch, heads, seq, dim = q.shape
    out = np.zeros_like(q)
    upper = np.triu(np.ones((seq, seq), dtype=bool), k=1)
    for b in range(batch):
        for h in range(heads):
            scores = q[b, h] @ k[b, h].T / np.sqrt(dim)
            scores = np.where(upper, -np.inf, scores)
            shifted = scores - scores.max(axis=-1, keepdims=True)
            weights = np.exp(shifted)
            weights = weights / weights.sum(axis=-1, keepdims=True)
            out[b, h] = weights @ v[b, h]
    return out


def test_softmax_rows_sum_to_one():
    values = np.array([[1.0, 2.0, 3.0], [-5.0, 0.0, 5.0]])
    assert np.allclose(softmax(values).sum(axis=-1), 1.0)


def test_multi_head_matches_the_naive_reference():
    q, k, v = random_qkv(batch=2, n_heads=4, seq=12, head_dim=8, seed=1)
    output, _ = attention(q, k, v)
    assert np.allclose(output, reference_multi_head(q, k, v))


def test_weights_are_a_distribution():
    q, k, v = random_qkv(batch=1, n_heads=4, seq=10, head_dim=8, seed=2)
    _, weights = attention(q, k, v)
    assert np.allclose(weights.sum(axis=-1), 1.0)
    assert (weights >= 0).all()


def test_causal_attention_ignores_the_future():
    q, k, v = random_qkv(batch=1, n_heads=2, seq=8, head_dim=4, seed=3)
    _, weights = attention(q, k, v, causal=True)
    upper = np.triu(np.ones((8, 8), dtype=bool), k=1)
    assert np.allclose(weights[..., upper], 0.0)


def test_non_causal_attends_everywhere():
    q, k, v = random_qkv(batch=1, n_heads=2, seq=6, head_dim=4, seed=4)
    _, weights = attention(q, k, v, causal=False)
    assert (weights > 0).all()


def test_causal_mask_handles_the_decode_case():
    """One new query attending over a cached prefix: nothing may be masked."""

    mask = causal_mask(seq_q=1, seq_k=9)
    assert mask.shape == (1, 9)
    assert not mask.any()


def test_causal_mask_rejects_more_queries_than_keys():
    with pytest.raises(ValueError):
        causal_mask(seq_q=5, seq_k=3)


def test_repeat_kv_expands_groups_in_order():
    x = np.arange(2 * 2 * 1 * 1, dtype=float).reshape(2, 2, 1, 1)
    repeated = repeat_kv(x, 3)
    assert repeated.shape == (2, 6, 1, 1)
    assert list(repeated[0, :, 0, 0]) == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]


def test_grouped_query_equals_multi_head_when_groups_are_singletons():
    """MHA is not a special case bolted on; it is n_kv_heads == n_heads."""

    q, k, v = random_qkv(batch=1, n_heads=8, seq=10, head_dim=8, seed=5)
    full, _ = attention(q, k, v)
    pooled, _ = attention(q, pool_kv(k, 8), pool_kv(v, 8))
    assert np.allclose(full, pooled)


def test_multi_query_shares_one_head():
    q, k, v = random_qkv(batch=1, n_heads=8, seq=10, head_dim=8, seed=6)
    k1, v1 = pool_kv(k, 1), pool_kv(v, 1)
    assert k1.shape == (1, 1, 10, 8)

    grouped, _ = attention(q, k1, v1)
    expanded, _ = attention(q, repeat_kv(k1, 8), repeat_kv(v1, 8))
    assert np.allclose(grouped, expanded)


def test_pool_kv_averages_within_the_group():
    x = np.zeros((1, 4, 1, 1))
    x[0, :, 0, 0] = [1.0, 3.0, 10.0, 20.0]
    pooled = pool_kv(x, 2)
    assert pooled.shape == (1, 2, 1, 1)
    assert list(pooled[0, :, 0, 0]) == [2.0, 15.0]


def test_head_counts_must_divide():
    q, k, v = random_qkv(batch=1, n_heads=6, seq=4, head_dim=4, seed=7)
    with pytest.raises(ValueError):
        attention(q, pool_kv(k, 3), pool_kv(v, 2))
    with pytest.raises(ValueError):
        pool_kv(k, 4)


def test_fidelity_sweep_is_exact_at_full_width_and_worst_at_one():
    results = dict(fidelity_sweep(n_heads=8, seq=32, head_dim=16, seed=8))
    assert results[8] == pytest.approx(0.0, abs=1e-12)
    assert results[1] > results[2] > results[4] > 0.0
