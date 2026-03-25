#!/usr/bin/env python3
"""
Verify TQ memory savings with actual GPU tensor allocations.

Requires CUDA. Allocates real tensors and measures torch.cuda.memory_allocated.

Usage:
    python proof_memory.py
"""
import torch
import gc


def mem():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated(0)


def measure_fp8(n_layers, n_kv, hdim, n_tok):
    gc.collect(); torch.cuda.empty_cache()
    before = mem()
    caches = []
    for _ in range(n_layers):
        k = torch.zeros(1, n_kv, n_tok, hdim, dtype=torch.uint8, device="cuda")
        v = torch.zeros(1, n_kv, n_tok, hdim, dtype=torch.uint8, device="cuda")
        caches.append((k, v))
    used = mem() - before
    del caches; gc.collect(); torch.cuda.empty_cache()
    return used


def measure_tq(n_layers, n_kv, hdim, n_tok):
    from turboquant.kv_cache import TurboQuantKVCache
    gc.collect(); torch.cuda.empty_cache()
    before = mem()
    caches = []
    for i in range(n_layers):
        c = TurboQuantKVCache(
            head_dim=hdim, key_bits=3, value_bits=2,
            value_group_size=min(32, hdim), buffer_size=0,
            device=torch.device("cuda"), layer_idx=i,
        )
        k = torch.randn(1, n_kv, n_tok, hdim, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(1, n_kv, n_tok, hdim, dtype=torch.bfloat16, device="cuda")
        c.prefill(k, v)
        del k, v
        caches.append(c)
    gc.collect(); torch.cuda.empty_cache()
    used = mem() - before
    del caches; gc.collect(); torch.cuda.empty_cache()
    return used


def main():
    torch.cuda.set_device(0)

    # Print the analytical table first
    from turboquant.proof import print_memory_proof
    print_memory_proof()

    # Then verify one config with actual GPU allocation
    print("  Verification (actual GPU tensors, 1 layer, 8192 tokens, d=128):")
    fp8 = measure_fp8(1, 1, 128, 8192)
    tq = measure_tq(1, 1, 128, 8192)
    print(f"    FP8: {fp8/1e6:.2f} MB")
    print(f"    TQ:  {tq/1e6:.2f} MB")
    print(f"    Ratio: {tq/fp8:.2f}x\n")


if __name__ == "__main__":
    main()
