"""Compare multi-head, grouped-query and multi-query attention.

    python cli.py                          # everything, for Llama 3 70B
    python cli.py --model llama-3-8b --context 32768
    python cli.py --catalogue
    python cli.py --fidelity

Every number is computed from the model's published attention shape; nothing is
looked up or hard-coded per layout.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from gqa.cache import GIB, dtype_size, human
from gqa.models import get, names
from gqa.report import (
    bandwidth_report,
    budget_report,
    catalogue_report,
    fidelity_report,
    layout_report,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="llama-3-70b", choices=names())
    parser.add_argument("--context", type=int, default=8192, help="sequence length in tokens")
    parser.add_argument("--batch", type=int, default=1, help="concurrent sequences")
    parser.add_argument("--dtype", default="fp16", help="KV cache element type")
    parser.add_argument(
        "--weight-dtype", default="fp16", help="element type the weights are served in"
    )
    parser.add_argument(
        "--budget-gib", type=float, default=40.0, help="memory available for the cache"
    )
    parser.add_argument(
        "--bandwidth-gibs",
        type=float,
        default=2000.0,
        help="memory bandwidth in GiB/s (an H100 SXM is roughly 3000)",
    )
    parser.add_argument("--efficiency", type=float, default=0.7)
    parser.add_argument("--catalogue", action="store_true", help="compare every known model")
    parser.add_argument("--fidelity", action="store_true", help="only the output-drift sweep")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)

    if args.fidelity:
        print(fidelity_report())
        return 0

    if args.catalogue:
        print(catalogue_report(args.context, args.batch, args.dtype))
        return 0

    shape = get(args.model)
    active_param_bytes = shape.params_b * 1e9 * dtype_size(args.weight_dtype)

    sections = [
        layout_report(shape, args.context, args.batch, args.dtype),
        budget_report(shape, args.budget_gib * GIB, args.batch, args.dtype),
        bandwidth_report(
            shape,
            args.context,
            active_param_bytes,
            args.bandwidth_gibs * GIB,
            args.batch,
            args.dtype,
            args.efficiency,
        ),
        fidelity_report(),
    ]
    print("\n\n".join(sections))
    print()
    print(
        "Note: the layout is a property of the trained model, not a serving flag. "
        "You cannot switch an MHA checkpoint to GQA at load time; the {} row above "
        "is what the architecture would have had to be.".format("MHA" if shape.layout != "MHA" else "MQA")
    )
    print("Weights assumed at {} ({} for {:.1f}B parameters).".format(
        args.weight_dtype, human(active_param_bytes), shape.params_b
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
