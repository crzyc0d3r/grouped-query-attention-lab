"""Tests for the cache arithmetic and the published model shapes."""

import pytest

from gqa.cache import (
    GIB,
    KIB,
    MIB,
    compare_layouts,
    decode_bytes_per_step,
    decode_tokens_per_second,
    dtype_size,
    human,
    kv_bytes,
    kv_bytes_per_token,
    max_context,
)
from gqa.models import CATALOGUE, ModelShape, get, names


def test_llama_3_70b_costs_320_kib_per_token():
    """2 x 80 layers x 8 kv heads x 128 dims x 2 bytes."""

    assert kv_bytes_per_token(get("llama-3-70b")) == 320 * KIB


def test_the_same_model_with_multi_head_would_cost_eight_times_more():
    shape = get("llama-3-70b")
    grouped = kv_bytes_per_token(shape)
    multi_head = kv_bytes_per_token(shape, n_kv_heads=shape.n_heads)
    assert multi_head / grouped == 8.0
    assert multi_head == 2.5 * MIB


def test_llama_3_8b_costs_128_kib_per_token():
    assert kv_bytes_per_token(get("llama-3-8b")) == 128 * KIB


def test_query_head_count_does_not_enter_the_formula():
    """The whole GQA argument, as a test."""

    base = get("llama-3-70b")
    wider = ModelShape("wider", base.n_layers, 128, base.n_kv_heads, base.head_dim)
    assert kv_bytes_per_token(wider) == kv_bytes_per_token(base)


def test_cache_scales_with_context_and_batch():
    shape = get("mistral-7b")
    per_token = kv_bytes_per_token(shape)
    assert kv_bytes(shape, context=1000, batch=3) == per_token * 3000


def test_dtype_halves_the_cache():
    shape = get("llama-3-70b")
    assert kv_bytes_per_token(shape, "fp16") == 2 * kv_bytes_per_token(shape, "fp8")
    assert kv_bytes_per_token(shape, "fp32") == 2 * kv_bytes_per_token(shape, "fp16")


def test_unknown_dtype_is_rejected():
    with pytest.raises(KeyError):
        dtype_size("bfloat13")


def test_unknown_model_is_rejected():
    with pytest.raises(KeyError):
        get("llama-9000")


def test_max_context_inverts_the_cache_formula():
    shape = get("llama-3-70b")
    budget = 40 * GIB
    tokens = max_context(shape, budget)
    assert kv_bytes(shape, tokens) <= budget
    assert kv_bytes(shape, tokens + 1) > budget


def test_grouping_buys_proportionally_more_context():
    shape = get("llama-3-70b")
    budget = 40 * GIB
    grouped = max_context(shape, budget)
    multi_head = max_context(shape, budget, n_kv_heads=shape.n_heads)
    assert grouped == multi_head * 8


def test_decode_bytes_include_weights_and_cache():
    shape = get("llama-3-8b")
    weights = 16e9
    total = decode_bytes_per_step(shape, context=4096, active_param_bytes=weights)
    assert total == weights + kv_bytes(shape, 4096)


def test_throughput_falls_as_context_grows():
    shape = get("llama-3-8b")
    weights = 16e9
    bandwidth = 2000 * GIB
    short = decode_tokens_per_second(
        bandwidth, decode_bytes_per_step(shape, 1024, weights)
    )
    long = decode_tokens_per_second(
        bandwidth, decode_bytes_per_step(shape, 131072, weights)
    )
    assert short > long > 0


def test_compare_layouts_deduplicates_for_a_multi_head_model():
    """Llama 2 7B is already MHA, so MHA and 'as released' are one row."""

    rows = compare_layouts(get("llama-2-7b"), context=4096)
    assert [row.n_kv_heads for row in rows] == [32, 1]
    assert rows[0].ratio_vs_mha == 1.0


def test_compare_layouts_gives_three_rows_for_a_grouped_model():
    rows = compare_layouts(get("llama-3-70b"), context=4096)
    assert [row.n_kv_heads for row in rows] == [64, 8, 1]
    assert [row.group_size for row in rows] == [1, 8, 64]


def test_every_catalogue_entry_is_self_consistent():
    for key in names():
        shape = CATALOGUE[key]
        assert shape.n_heads % shape.n_kv_heads == 0
        assert shape.group_size * shape.n_kv_heads == shape.n_heads
        assert shape.layout in {"MHA", "GQA", "MQA"}
        assert shape.params_b > 0


def test_shape_rejects_indivisible_head_counts():
    with pytest.raises(ValueError):
        ModelShape("bad", 8, 12, 5, 64)


def test_human_formats_each_scale():
    assert human(512) == "512 B"
    assert human(2 * KIB) == "2.00 KiB"
    assert human(3 * MIB) == "3.00 MiB"
    assert human(1.5 * GIB) == "1.50 GiB"
