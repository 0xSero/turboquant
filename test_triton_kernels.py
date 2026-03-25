#!/usr/bin/env python3
"""
Test TurboQuant Triton kernels against PyTorch reference implementation.

Tests:
1. MSE score kernel vs PyTorch dequantize-then-matmul
2. QJL score kernel vs PyTorch unpack-sketch-dot
3. Combined attention score vs full PyTorch path
4. Fused decode (scores + softmax + value aggr) vs PyTorch
5. Performance benchmark: Triton vs PyTorch
"""

import math
import time
import torch
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from turboquant.quantizer import TurboQuantProd, TurboQuantMSE, MSEQuantized, ProdQuantized
from turboquant.kv_cache import quantize_values, dequantize_values
from turboquant.triton_kernels import (
    turboquant_mse_score,
    turboquant_qjl_score,
    turboquant_attention_score,
    turboquant_fused_decode,
    _get_packing_params,
)

torch.manual_seed(42)
device = torch.device("cuda")

PASS = 0
FAIL = 0


def run_test(name, fn, **kwargs):
    global PASS, FAIL
    try:
        fn(**kwargs)
        print(f"  ✓ {name}")
        PASS += 1
    except Exception as e:
        import traceback
        print(f"  ✗ {name}")
        traceback.print_exc()
        FAIL += 1


# ─── Test 1: MSE Score Kernel ─────────────────────────────────────────

def test_mse_score(B=2, H=4, N=128, D=64, bits=3):
    """Compare Triton MSE score vs PyTorch reference."""
    BH = B * H

    # Create quantizer and random keys
    quantizer = TurboQuantMSE(dim=D, bits=bits - 1, device=device, seed=42)  # bits-1 for MSE stage
    keys = torch.randn(BH, N, D, device=device, dtype=torch.float16)

    # Quantize
    mse_q = quantizer.quantize(keys)

    # --- PyTorch reference ---
    keys_dequant = quantizer.dequantize(mse_q)
    query = torch.randn(BH, D, device=device, dtype=torch.float16)
    ref_scores = torch.matmul(query.float().unsqueeze(1), keys_dequant.float().transpose(-2, -1)).squeeze(1)

    # --- Triton kernel ---
    # Rotate query: q_rot = q @ Pi^T
    q_rot = torch.matmul(query.float(), quantizer.Pi.T)

    triton_scores = turboquant_mse_score(
        q_rot, mse_q.indices, mse_q.norms, quantizer.centroids, mse_q.bits
    )

    # Compare
    torch.testing.assert_close(triton_scores, ref_scores.float(), atol=1e-2, rtol=1e-2)


# ─── Test 2: QJL Score Kernel ─────────────────────────────────────────

def test_qjl_score(B=2, H=4, N=128, D=64, bits=3):
    """Compare Triton QJL score vs PyTorch reference."""
    BH = B * H

    # Create full quantizer
    quantizer = TurboQuantProd(dim=D, bits=bits, device=device, seed=42)
    keys = torch.randn(BH, N, D, device=device, dtype=torch.float16)
    query = torch.randn(BH, D, device=device, dtype=torch.float16)

    # Quantize
    prod_q = quantizer.quantize(keys)

    # --- PyTorch reference: QJL part only ---
    signs = quantizer._unpack_qjl_signs(prod_q.qjl_signs)
    q_sketched = torch.matmul(query.float(), quantizer.S.T)
    ref_qjl = torch.matmul(q_sketched.unsqueeze(1), signs.transpose(-2, -1)).squeeze(1)
    ref_qjl = ref_qjl * (quantizer.qjl_scale * prod_q.residual_norms)

    # --- Triton kernel ---
    q_sketch = torch.matmul(query.float(), quantizer.S.T)
    triton_qjl = turboquant_qjl_score(
        q_sketch, prod_q.qjl_signs, prod_q.residual_norms, quantizer.qjl_scale
    )

    torch.testing.assert_close(triton_qjl, ref_qjl.float(), atol=1e-2, rtol=1e-2)


# ─── Test 3: Combined Score ───────────────────────────────────────────

def test_combined_score(B=2, H=2, N=256, D=64, bits=3):
    """Compare Triton combined score vs PyTorch TurboQuantProd.attention_score."""
    BH = B * H

    quantizer = TurboQuantProd(dim=D, bits=bits, device=device, seed=42)
    keys = torch.randn(BH, N, D, device=device, dtype=torch.float16)
    query = torch.randn(BH, 1, D, device=device, dtype=torch.float16)

    prod_q = quantizer.quantize(keys)

    # --- PyTorch reference ---
    ref_scores = quantizer.attention_score(query, prod_q).squeeze(1)  # (BH, N)

    # --- Triton ---
    triton_scores = turboquant_attention_score(
        query, prod_q,
        quantizer.mse_quantizer.Pi,
        quantizer.S,
        quantizer.mse_quantizer.centroids,
        prod_q.mse_bits,
        quantizer.qjl_scale,
    )

    torch.testing.assert_close(triton_scores, ref_scores.float(), atol=0.1, rtol=0.05)


# ─── Test 4: Fused Decode ─────────────────────────────────────────────

def test_fused_decode(B=1, H=2, N=64, D=64, bits=3, group_size=32):
    """Compare Triton fused decode vs PyTorch score+softmax+value."""
    BH = B * H

    quantizer = TurboQuantProd(dim=D, bits=bits, device=device, seed=42)
    keys = torch.randn(BH, N, D, device=device, dtype=torch.float16)
    values = torch.randn(BH, N, D, device=device, dtype=torch.float16)
    query = torch.randn(BH, 1, D, device=device, dtype=torch.float16)

    sm_scale = 1.0 / math.sqrt(D)

    # Quantize
    prod_q = quantizer.quantize(keys)
    val_q = quantize_values(values, bits=2, group_size=group_size)

    # --- PyTorch reference ---
    ref_scores = quantizer.attention_score(query, prod_q)  # (BH, 1, N)
    ref_scores = ref_scores * sm_scale
    ref_weights = torch.softmax(ref_scores.float(), dim=-1)
    ref_values = dequantize_values(val_q, group_size)  # (BH, N, D)
    ref_output = torch.matmul(ref_weights, ref_values.float()).squeeze(1)  # (BH, D)

    # --- Triton fused ---
    triton_output = turboquant_fused_decode(
        query, prod_q, val_q,
        quantizer.mse_quantizer.Pi,
        quantizer.S,
        quantizer.mse_quantizer.centroids,
        prod_q.mse_bits,
        quantizer.qjl_scale,
        sm_scale,
        group_size,
    )

    # Attention output should be close (not exact due to float accumulation order)
    cos_sim = torch.nn.functional.cosine_similarity(
        triton_output.flatten().float(),
        ref_output.flatten().float(),
        dim=0,
    )
    print(f"      cosine_sim={cos_sim.item():.6f}, "
          f"mse={((triton_output.float() - ref_output.float()) ** 2).mean().item():.2e}")
    assert cos_sim > 0.95, f"Cosine similarity too low: {cos_sim.item()}"


# ─── Test 5: Various configs ──────────────────────────────────────────

def test_various_configs():
    """Test with different bit widths and dimensions."""
    for D in [64, 128]:
        for bits in [2, 3, 4]:
            BH, N = 4, 64
            quantizer = TurboQuantProd(dim=D, bits=bits, device=device, seed=42)
            keys = torch.randn(BH, N, D, device=device, dtype=torch.float16)
            query = torch.randn(BH, 1, D, device=device, dtype=torch.float16)

            prod_q = quantizer.quantize(keys)

            ref = quantizer.attention_score(query, prod_q).squeeze(1)
            triton_out = turboquant_attention_score(
                query, prod_q,
                quantizer.mse_quantizer.Pi,
                quantizer.S,
                quantizer.mse_quantizer.centroids,
                prod_q.mse_bits,
                quantizer.qjl_scale,
            )

            max_err = (triton_out - ref.float()).abs().max().item()
            print(f"      D={D}, bits={bits}: max_err={max_err:.4f}")
            assert max_err < 1.0, f"Max error too high: {max_err}"


# ─── Test 6: Benchmark ───────────────────────────────────────────────

def benchmark():
    """Benchmark Triton vs PyTorch for score computation."""
    print()
    print("  ── Performance Benchmark ──")
    B, H, D, bits = 1, 32, 64, 3
    BH = B * H

    quantizer = TurboQuantProd(dim=D, bits=bits, device=device, seed=42)

    for N in [256, 512, 1024, 2048, 4096]:
        keys = torch.randn(BH, N, D, device=device, dtype=torch.float16)
        query = torch.randn(BH, 1, D, device=device, dtype=torch.float16)
        prod_q = quantizer.quantize(keys)

        # Warmup
        for _ in range(5):
            _ = quantizer.attention_score(query, prod_q)
            _ = turboquant_attention_score(
                query, prod_q, quantizer.mse_quantizer.Pi, quantizer.S,
                quantizer.mse_quantizer.centroids, prod_q.mse_bits, quantizer.qjl_scale,
            )
        torch.cuda.synchronize()

        # PyTorch
        N_ITER = 50
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(N_ITER):
            _ = quantizer.attention_score(query, prod_q)
        torch.cuda.synchronize()
        pytorch_ms = (time.time() - t0) / N_ITER * 1000

        # Triton
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(N_ITER):
            _ = turboquant_attention_score(
                query, prod_q, quantizer.mse_quantizer.Pi, quantizer.S,
                quantizer.mse_quantizer.centroids, prod_q.mse_bits, quantizer.qjl_scale,
            )
        torch.cuda.synchronize()
        triton_ms = (time.time() - t0) / N_ITER * 1000

        speedup = pytorch_ms / triton_ms if triton_ms > 0 else float('inf')
        print(f"    N={N:5d}  PyTorch: {pytorch_ms:6.2f}ms  Triton: {triton_ms:6.2f}ms  Speedup: {speedup:.2f}x")


# ─── Run ──────────────────────────────────────────────────────────────

print(f"GPU: {torch.cuda.get_device_name()}")
print()

print("═" * 60)
print("TurboQuant Triton Kernel Tests")
print("═" * 60)

run_test("MSE score (B=2 H=4 N=128 D=64 bits=3)", test_mse_score)
run_test("MSE score (B=1 H=8 N=512 D=64 bits=4)", test_mse_score, B=1, H=8, N=512, D=64, bits=4)
run_test("MSE score (B=1 H=4 N=256 D=128 bits=3)", test_mse_score, B=1, H=4, N=256, D=128, bits=3)

run_test("QJL score (B=2 H=4 N=128 D=64 bits=3)", test_qjl_score)
run_test("QJL score (B=1 H=8 N=512 D=64 bits=4)", test_qjl_score, B=1, H=8, N=512, D=64, bits=4)
run_test("QJL score (B=1 H=4 N=256 D=128 bits=3)", test_qjl_score, B=1, H=4, N=256, D=128, bits=3)

run_test("Combined score (B=2 H=2 N=256 D=64 bits=3)", test_combined_score)
run_test("Combined score (B=1 H=4 N=512 D=128 bits=4)", test_combined_score, B=1, H=4, N=512, D=128, bits=4)

run_test("Fused decode (B=1 H=2 N=64 D=64 bits=3)", test_fused_decode)
run_test("Fused decode (B=1 H=4 N=128 D=64 bits=3)", test_fused_decode, B=1, H=4, N=128)

run_test("Various configs (D=64/128, bits=2/3/4)", test_various_configs)

run_test("Benchmark", benchmark)

print()
print("═" * 60)
print(f"Results: {PASS} passed, {FAIL} failed (total {PASS + FAIL})")
if FAIL == 0:
    print("🎉 All tests passed!")
else:
    print(f"⚠️  {FAIL} test(s) failed")
print("═" * 60)
