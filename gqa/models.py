"""Attention shapes of released models, as published in their configs.

Only fields that affect the KV cache are recorded: layer count, query-head count,
key/value-head count and head dimension. Anything else (vocabulary, MLP width,
rotary base) does not enter the cache arithmetic.

The catalogue is deliberately small and checkable. Every entry corresponds to
values in that model's public ``config.json`` - ``num_hidden_layers``,
``num_attention_heads``, ``num_key_value_heads``, and ``hidden_size /
num_attention_heads`` for the head dimension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ModelShape:
    """Everything about a model that the KV cache depends on."""

    name: str
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    note: str = ""
    # Approximate total parameter count, in billions. Only used for the
    # bandwidth estimate, where it stands in for "bytes of weights read per
    # decode step". These models are dense, so all parameters are active.
    params_b: float = 0.0

    def __post_init__(self) -> None:
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                "{}: n_heads ({}) must be divisible by n_kv_heads ({})".format(
                    self.name, self.n_heads, self.n_kv_heads
                )
            )

    @property
    def group_size(self) -> int:
        """How many query heads share one KV head."""

        return self.n_heads // self.n_kv_heads

    @property
    def layout(self) -> str:
        if self.n_kv_heads == self.n_heads:
            return "MHA"
        if self.n_kv_heads == 1:
            return "MQA"
        return "GQA"

    def with_kv_heads(self, n_kv_heads: int) -> "ModelShape":
        """The same model re-imagined with a different KV-head count."""

        return ModelShape(
            name="{} ({} kv)".format(self.name, n_kv_heads),
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=self.head_dim,
            note="hypothetical",
            params_b=self.params_b,
        )


CATALOGUE: Dict[str, ModelShape] = {
    "gpt-2": ModelShape("GPT-2 (124M)", 12, 12, 12, 64, "multi-head, pre-GQA", 0.124),
    "llama-2-7b": ModelShape("Llama 2 7B", 32, 32, 32, 128, "multi-head", 6.7),
    "llama-2-70b": ModelShape("Llama 2 70B", 80, 64, 8, 128, "8 query heads per kv head", 69.0),
    "llama-3-8b": ModelShape("Llama 3 8B", 32, 32, 8, 128, "4 query heads per kv head", 8.0),
    "llama-3-70b": ModelShape("Llama 3 70B", 80, 64, 8, 128, "8 query heads per kv head", 70.6),
    "mistral-7b": ModelShape("Mistral 7B v0.1", 32, 32, 8, 128, "4 query heads per kv head", 7.2),
}


def get(key: str) -> ModelShape:
    try:
        return CATALOGUE[key]
    except KeyError:
        raise KeyError(
            "unknown model {!r}; known: {}".format(key, ", ".join(sorted(CATALOGUE)))
        )


def names() -> List[str]:
    return sorted(CATALOGUE)
