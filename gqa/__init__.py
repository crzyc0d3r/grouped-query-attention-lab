"""Multi-head, grouped-query and multi-query attention, side by side.

    gqa.attention  - one attention implementation covering all three layouts
    gqa.models     - real model shapes, as published
    gqa.cache      - KV cache arithmetic and decode-bandwidth estimates
    gqa.report     - the comparison tables
"""

__all__ = ["attention", "models", "cache", "report"]
