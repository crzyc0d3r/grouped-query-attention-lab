"""KV cache arithmetic.

Every formula here comes from one sentence: during decoding the model keeps one
key vector and one value vector per **key/value head**, per layer, for every
token it has seen. So

    bytes per token = 2 x n_layers x n_kv_heads x head_dim x bytes_per_element
                      ^
                      K and V

Note what is *absent*: the query-head count. That is the entire GQA argument.
Sharing key/value heads across query heads leaves the model's width unchanged and
divides this number by the group size.

The second thing the cache costs is bandwidth. Decoding one token requires
streaming every active weight **and the whole cache so far** through the memory
bus, so a cache that is 8x smaller is also 8x fewer bytes read per step - which
matters more than the capacity saving once contexts get long.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .models import ModelShape

KIB = 1024
MIB = 1024 * 1024
GIB = 1024 * 1024 * 1024

DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1, "int8": 1, "int4": 0.5}


def dtype_size(dtype: str) -> float:
    try:
        return DTYPE_BYTES[dtype]
    except KeyError:
        raise KeyError(
            "unknown dtype {!r}; known: {}".format(dtype, ", ".join(sorted(DTYPE_BYTES)))
        )


def kv_bytes_per_token(
    shape: ModelShape,
    dtype: str = "fp16",
    n_kv_heads: Optional[int] = None,
) -> float:
    """Cache cost of one token, across all layers.

    Pass ``n_kv_heads`` to price a hypothetical layout - for instance what the
    same model would cost if it had been built with full multi-head attention.
    """

    heads = shape.n_kv_heads if n_kv_heads is None else n_kv_heads
    return 2.0 * shape.n_layers * heads * shape.head_dim * dtype_size(dtype)


def kv_bytes(
    shape: ModelShape,
    context: int,
    batch: int = 1,
    dtype: str = "fp16",
    n_kv_heads: Optional[int] = None,
) -> float:
    """Total cache for ``batch`` sequences of ``context`` tokens."""

    return kv_bytes_per_token(shape, dtype, n_kv_heads) * context * batch


def max_context(
    shape: ModelShape,
    budget_bytes: float,
    batch: int = 1,
    dtype: str = "fp16",
    n_kv_heads: Optional[int] = None,
) -> int:
    """Longest context whose cache fits in ``budget_bytes``."""

    per_token = kv_bytes_per_token(shape, dtype, n_kv_heads) * batch
    if per_token <= 0:
        return 0
    return int(budget_bytes // per_token)


def decode_bytes_per_step(
    shape: ModelShape,
    context: int,
    active_param_bytes: float,
    batch: int = 1,
    dtype: str = "fp16",
    n_kv_heads: Optional[int] = None,
) -> float:
    """Bytes the memory bus must move to emit one more token.

    Weights are read once per step regardless of batch size; the cache is read
    once per sequence in the batch.
    """

    return active_param_bytes + kv_bytes(shape, context, batch, dtype, n_kv_heads)


def decode_tokens_per_second(
    bandwidth_bytes_per_second: float,
    bytes_per_step: float,
    efficiency: float = 0.7,
) -> float:
    """A first-order decode-speed estimate.

    Decoding is bandwidth-bound rather than compute-bound, so throughput is
    roughly bandwidth divided by bytes moved per step. ``efficiency`` absorbs
    everything this ignores - kernel overhead, imperfect overlap, attention that
    is not a pure streaming read. Treat the number as an upper-bound sanity
    check, not a benchmark.
    """

    if bytes_per_step <= 0:
        return 0.0
    return bandwidth_bytes_per_second / bytes_per_step * efficiency


@dataclass
class CacheComparison:
    """One layout's cost, ready to print."""

    label: str
    n_kv_heads: int
    group_size: int
    bytes_per_token: float
    total_bytes: float
    ratio_vs_mha: float


def compare_layouts(
    shape: ModelShape,
    context: int,
    batch: int = 1,
    dtype: str = "fp16",
):
    """Price MHA, the model's real layout, and MQA for the same model."""

    mha_per_token = kv_bytes_per_token(shape, dtype, shape.n_heads)

    layouts = [
        ("MHA (n_kv_heads = n_heads)", shape.n_heads),
        ("{} (as released: {} kv heads)".format(shape.layout, shape.n_kv_heads), shape.n_kv_heads),
        ("MQA (n_kv_heads = 1)", 1),
    ]

    seen = set()
    rows = []
    for label, heads in layouts:
        if heads in seen:
            continue
        seen.add(heads)
        per_token = kv_bytes_per_token(shape, dtype, heads)
        rows.append(
            CacheComparison(
                label=label,
                n_kv_heads=heads,
                group_size=shape.n_heads // heads,
                bytes_per_token=per_token,
                total_bytes=per_token * context * batch,
                ratio_vs_mha=mha_per_token / per_token if per_token else float("inf"),
            )
        )
    return rows


def human(num_bytes: float) -> str:
    """Render a byte count the way a person would say it."""

    value = float(num_bytes)
    for unit, size in (("GiB", GIB), ("MiB", MIB), ("KiB", KIB)):
        if abs(value) >= size:
            return "{:.2f} {}".format(value / size, unit)
    return "{:.0f} B".format(value)
