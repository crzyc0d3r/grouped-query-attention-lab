"""Tables. Everything here is presentation over ``cache`` and ``attention``."""

from __future__ import annotations

from typing import List, Sequence

from .attention import fidelity_sweep
from .cache import (
    compare_layouts,
    decode_bytes_per_step,
    decode_tokens_per_second,
    human,
    kv_bytes_per_token,
    max_context,
)
from .models import CATALOGUE, ModelShape


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def line(cells: Sequence[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)).rstrip()

    out = [line(headers), "  ".join("-" * w for w in widths)]
    out.extend(line(row) for row in rows)
    return "\n".join(out)


def layout_report(shape: ModelShape, context: int, batch: int = 1, dtype: str = "fp16") -> str:
    rows = []
    for entry in compare_layouts(shape, context, batch, dtype):
        rows.append(
            [
                entry.label,
                str(entry.n_kv_heads),
                "{}:1".format(entry.group_size),
                human(entry.bytes_per_token),
                human(entry.total_bytes),
                "{:.1f}x".format(entry.ratio_vs_mha),
            ]
        )
    header = "{}  -  {} layers, {} query heads, head_dim {}, {} @ {} tokens x {}".format(
        shape.name, shape.n_layers, shape.n_heads, shape.head_dim, dtype, context, batch
    )
    table = _table(
        ["layout", "kv heads", "sharing", "per token", "total cache", "saving"],
        rows,
    )
    return "{}\n{}\n{}".format(header, "=" * len(header), table)


def catalogue_report(context: int, batch: int = 1, dtype: str = "fp16") -> str:
    rows = []
    for key in sorted(CATALOGUE):
        shape = CATALOGUE[key]
        per_token = kv_bytes_per_token(shape, dtype)
        mha_per_token = kv_bytes_per_token(shape, dtype, shape.n_heads)
        rows.append(
            [
                key,
                shape.layout,
                "{}/{}".format(shape.n_kv_heads, shape.n_heads),
                "{}:1".format(shape.group_size),
                human(per_token),
                human(per_token * context * batch),
                "{:.1f}x".format(mha_per_token / per_token),
            ]
        )
    header = "KV cache across the catalogue  -  {} @ {} tokens x {}".format(dtype, context, batch)
    table = _table(
        ["model", "layout", "kv/q heads", "sharing", "per token", "cache", "vs MHA"],
        rows,
    )
    return "{}\n{}\n{}".format(header, "=" * len(header), table)


def budget_report(
    shape: ModelShape,
    budget_bytes: float,
    batch: int = 1,
    dtype: str = "fp16",
) -> str:
    """How much context each layout buys out of the same memory budget."""

    rows = []
    for label, heads in (
        ("MHA", shape.n_heads),
        (shape.layout, shape.n_kv_heads),
        ("MQA", 1),
    ):
        if any(row[0] == label for row in rows):
            continue
        rows.append(
            [
                label,
                str(heads),
                "{:,}".format(max_context(shape, budget_bytes, batch, dtype, heads)),
            ]
        )
    header = "{}  -  context that fits in {} of cache (batch {})".format(
        shape.name, human(budget_bytes), batch
    )
    return "{}\n{}\n{}".format(
        header, "=" * len(header), _table(["layout", "kv heads", "max tokens"], rows)
    )


def bandwidth_report(
    shape: ModelShape,
    context: int,
    active_param_bytes: float,
    bandwidth_bytes_per_second: float,
    batch: int = 1,
    dtype: str = "fp16",
    efficiency: float = 0.7,
) -> str:
    """Decode speed, which is where the cache saving actually shows up."""

    rows = []
    seen = set()
    for label, heads in (
        ("MHA", shape.n_heads),
        (shape.layout, shape.n_kv_heads),
        ("MQA", 1),
    ):
        if heads in seen:
            continue
        seen.add(heads)
        step_bytes = decode_bytes_per_step(
            shape, context, active_param_bytes, batch, dtype, heads
        )
        cache_share = (step_bytes - active_param_bytes) / step_bytes * 100.0
        rows.append(
            [
                label,
                str(heads),
                human(step_bytes),
                "{:.0f}%".format(cache_share),
                "{:.1f}".format(
                    decode_tokens_per_second(
                        bandwidth_bytes_per_second, step_bytes, efficiency
                    )
                ),
            ]
        )
    header = "{}  -  bytes moved per decoded token at {} tokens of context".format(
        shape.name, context
    )
    footer = "weights {} active, bandwidth {}/s, efficiency {:.0%}".format(
        human(active_param_bytes), human(bandwidth_bytes_per_second), efficiency
    )
    table = _table(["layout", "kv heads", "bytes/step", "cache share", "tok/s (est)"], rows)
    return "{}\n{}\n{}\n{}".format(header, "=" * len(header), table, footer)


def fidelity_report(n_heads: int = 8, seq: int = 64, head_dim: int = 32, seed: int = 0) -> str:
    """What merging KV heads does to the attention output, with no retraining."""

    rows: List[List[str]] = []
    for n_kv_heads, error in fidelity_sweep(
        n_heads=n_heads, seq=seq, head_dim=head_dim, seed=seed
    ):
        if n_kv_heads == n_heads:
            layout = "MHA"
        elif n_kv_heads == 1:
            layout = "MQA"
        else:
            layout = "GQA"
        rows.append(
            [
                str(n_kv_heads),
                layout,
                "{}:1".format(n_heads // n_kv_heads),
                "{:.4f}".format(error),
            ]
        )
    header = "Output drift from mean-pooling KV heads  -  {} query heads, seq {}".format(
        n_heads, seq
    )
    note = (
        "Relative L2 distance from full multi-head output on identical inputs.\n"
        "Zero at n_kv_heads = n_heads confirms MHA is this same code path.\n"
        "Pooling alone is the crude conversion: published GQA checkpoints\n"
        "mean-pool and then continue pre-training to recover most of this gap."
    )
    table = _table(["kv heads", "layout", "sharing", "relative error"], rows)
    return "{}\n{}\n{}\n\n{}".format(header, "=" * len(header), table, note)
