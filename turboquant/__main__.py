"""
TurboQuant CLI.

Usage:
    turboquant codebook --dim 256
    turboquant proof
"""
import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
logger = logging.getLogger("turboquant")


def cmd_codebook(args):
    from turboquant.codebook import get_codebook
    for bits in range(1, 5):
        cb = get_codebook(args.dim, bits)
        logger.info(f"d={args.dim}, b={bits}: {2**bits} centroids, MSE/coord={cb['mse_per_coord']:.6e}")


def cmd_proof(args):
    """Show TQ memory savings vs BF16/FP8 for common model configs."""
    from turboquant.proof import print_memory_proof
    print_memory_proof()


def main():
    parser = argparse.ArgumentParser(
        prog="turboquant",
        description="TurboQuant: KV cache quantization for LLM inference",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_cb = sub.add_parser("codebook", help="Pre-compute codebooks for a head dimension")
    p_cb.add_argument("--dim", type=int, required=True)

    sub.add_parser("proof", help="Print KV cache memory comparison table")

    args = parser.parse_args()
    if args.command == "codebook":
        cmd_codebook(args)
    elif args.command == "proof":
        cmd_proof(args)


if __name__ == "__main__":
    main()
