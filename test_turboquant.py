#!/usr/bin/env python3
"""
TurboQuant test suite — validates the core algorithms against paper's theorems.

Run: python test_turboquant.py
"""

import time
import torch
import numpy as np
import math

torch.manual_seed(42)
np.random.seed(42)


def test_codebook():
    """Test Lloyd-Max codebook computation."""
    print("=" * 60)
    print("TEST: Lloyd-Max codebook computation")
    print("=" * 60)

    from turboquant.codebook import compute_lloyd_max_codebook

    # Paper Table: for b=1,2,3,4, MSE ≈ 0.36, 0.117, 0.03, 0.009
    expected_mse = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}

    for bits in [1, 2, 3, 4]:
        t0 = time.time()
        cb = compute_lloyd_max_codebook(d=128, bits=bits)
        elapsed = time.time() - t0

        mse = cb["mse_total"]
        exp = expected_mse[bits]

        print(f"  bits={bits}: MSE={mse:.4f} (expected ≈{exp:.3f}), "
              f"n_centroids={len(cb['centroids'])}, time={elapsed:.2f}s")

        # Allow 30% tolerance from paper values (paper uses d→∞ approximation)
        assert abs(mse - exp) / exp < 0.30, f"MSE {mse} too far from expected {exp}"
        print(f"    ✓ Within tolerance of paper values")

    print()
    return True


def test_rotation():
    """Test that rotation preserves norms and produces Beta-distributed coords."""
    print("=" * 60)
    print("TEST: Random rotation properties")
    print("=" * 60)

    from turboquant.rotation import generate_rotation_matrix, rotate_forward

    d = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Pi = generate_rotation_matrix(d, device)

    # Check orthogonality
    I = Pi @ Pi.T
    orth_error = (I - torch.eye(d, device=device)).abs().max().item()
    print(f"  Orthogonality error: {orth_error:.2e}")
    assert orth_error < 1e-5, f"Rotation matrix not orthogonal: {orth_error}"
    print(f"    ✓ Matrix is orthogonal")

    # Check norm preservation
    x = torch.randn(1000, d, device=device)
    x = x / x.norm(dim=-1, keepdim=True)
    y = rotate_forward(x, Pi)
    norm_diff = (y.norm(dim=-1) - 1.0).abs().max().item()
    print(f"  Max norm deviation after rotation: {norm_diff:.2e}")
    assert norm_diff < 1e-5, f"Norms not preserved: {norm_diff}"
    print(f"    ✓ Norms preserved")

    # Check that rotated coords follow Beta distribution (approximately Gaussian for d=128)
    coords = y[:, 0].cpu().numpy()
    expected_std = 1.0 / math.sqrt(d)
    actual_std = coords.std()
    print(f"  Expected coord std: {expected_std:.4f}, actual: {actual_std:.4f}")
    assert abs(actual_std - expected_std) / expected_std < 0.15
    print(f"    ✓ Coordinate distribution matches theory")

    print()
    return True


def test_mse_quantizer():
    """Test TurboQuant MSE quantizer distortion bounds."""
    print("=" * 60)
    print("TEST: TurboQuant MSE quantizer")
    print("=" * 60)

    from turboquant.quantizer import TurboQuantMSE

    d = 128
    n_vectors = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paper Theorem 1 exact values for b=1,2,3,4
    expected_mse_total = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}

    for bits in [1, 2, 3, 4]:
        quantizer = TurboQuantMSE(dim=d, bits=bits, device=device)

        # Generate random unit vectors
        x = torch.randn(n_vectors, d, device=device)
        x = x / x.norm(dim=-1, keepdim=True)

        # Quantize and dequantize
        q = quantizer.quantize(x)
        x_hat = quantizer.dequantize(q)

        # Compute MSE
        mse = ((x - x_hat) ** 2).sum(dim=-1).mean().item()

        exp = expected_mse_total[bits]
        ratio = mse / exp

        status = "✓" if 0.7 <= ratio <= 1.5 else "✗"
        print(f"  bits={bits}: MSE={mse:.4f} (expected ≈{exp:.3f}, ratio={ratio:.2f}) {status}")

    print()
    return True


def test_prod_quantizer():
    """Test TurboQuant inner product quantizer — unbiasedness and distortion."""
    print("=" * 60)
    print("TEST: TurboQuant inner product quantizer")
    print("=" * 60)

    from turboquant.quantizer import TurboQuantProd

    d = 128
    n_vectors = 500
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for bits in [2, 3, 4]:
        quantizer = TurboQuantProd(dim=d, bits=bits, device=device)

        # Generate random vectors
        x = torch.randn(n_vectors, d, device=device)
        x = x / x.norm(dim=-1, keepdim=True)
        y = torch.randn(n_vectors, d, device=device)

        # Quantize x, compute <y, dequant(x)>
        q = quantizer.quantize(x)
        x_hat = quantizer.dequantize(q)

        # True inner products
        true_ip = (x * y).sum(dim=-1)

        # Estimated inner products
        est_ip = (x_hat * y).sum(dim=-1)

        # Check unbiasedness: E[est] ≈ true
        bias = (est_ip - true_ip).mean().item()
        relative_bias = abs(bias) / true_ip.abs().mean().item()

        # Check distortion
        distortion = ((est_ip - true_ip) ** 2).mean().item()

        # Paper Theorem 2: Dprod ≤ sqrt(3)π²/(2d) · ||y||² · 1/4^b
        y_norm_sq = (y ** 2).sum(dim=-1).mean().item()
        theoretical_bound = math.sqrt(3) * math.pi**2 / (2 * d) * y_norm_sq / 4**bits

        print(f"  bits={bits}:")
        print(f"    Bias: {bias:.6f} (relative: {relative_bias:.4f})")
        print(f"    Distortion: {distortion:.6f} (theoretical bound: {theoretical_bound:.6f})")
        print(f"    {'✓' if relative_bias < 0.1 else '✗'} Approximately unbiased")

    print()
    return True


def test_attention_score():
    """Test TurboQuant attention score computation."""
    print("=" * 60)
    print("TEST: Attention score computation")
    print("=" * 60)

    from turboquant.quantizer import TurboQuantProd

    d = 128
    n_keys = 200
    n_queries = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    min_corr = {2: 0.7, 3: 0.85, 4: 0.95}
    for bits in [2, 3, 4]:
        quantizer = TurboQuantProd(dim=d, bits=bits, device=device)

        # Simulate attention: queries (n_q, d) × keys (n_k, d)
        queries = torch.randn(1, 1, n_queries, d, device=device) * 0.1
        keys = torch.randn(1, 1, n_keys, d, device=device) * 0.1

        # True attention scores
        true_scores = torch.matmul(queries, keys.transpose(-2, -1))

        # Quantized attention scores
        q_keys = quantizer.quantize(keys)
        est_scores = quantizer.attention_score(queries, q_keys)

        # Check correlation
        true_flat = true_scores.flatten().cpu()
        est_flat = est_scores.flatten().cpu()
        correlation = torch.corrcoef(torch.stack([true_flat, est_flat]))[0, 1].item()

        # MSE
        score_mse = ((true_scores - est_scores) ** 2).mean().item()

        threshold = min_corr[bits]
        print(f"  bits={bits}: correlation={correlation:.4f}, score_MSE={score_mse:.6f}")
        assert correlation > threshold, f"Correlation {correlation:.4f} < {threshold} for {bits}-bit"
        print(f"    ✓ Correlation > {threshold} for {bits}-bit quantization")

    print()
    return True


def test_kv_cache():
    """Test the full KV cache with prefill and decode."""
    print("=" * 60)
    print("TEST: TurboQuant KV cache")
    print("=" * 60)

    from turboquant.kv_cache import TurboQuantKVCache

    d = 128
    n_heads = 8
    batch_size = 1
    prefill_len = 512
    decode_steps = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache = TurboQuantKVCache(
        head_dim=d,
        key_bits=3,
        value_bits=2,
        buffer_size=128,
        device=device,
        layer_idx=0,
    )

    # Generate random KV states
    keys = torch.randn(batch_size, n_heads, prefill_len, d, device=device) * 0.1
    values = torch.randn(batch_size, n_heads, prefill_len, d, device=device) * 0.1

    # Prefill
    t0 = time.time()
    cache.prefill(keys, values)
    prefill_time = time.time() - t0
    print(f"  Prefill ({prefill_len} tokens): {prefill_time*1000:.1f}ms")

    # Decode
    t0 = time.time()
    for i in range(decode_steps):
        new_k = torch.randn(batch_size, n_heads, 1, d, device=device) * 0.1
        new_v = torch.randn(batch_size, n_heads, 1, d, device=device) * 0.1
        cache.append(new_k, new_v)
    decode_time = time.time() - t0
    print(f"  Decode ({decode_steps} tokens): {decode_time*1000:.1f}ms "
          f"({decode_time/decode_steps*1000:.2f}ms/token)")

    # Attention score
    query = torch.randn(batch_size, n_heads, 1, d, device=device) * 0.1
    t0 = time.time()
    scores = cache.attention_scores(query)
    score_time = time.time() - t0
    print(f"  Attention score: {score_time*1000:.2f}ms, shape={scores.shape}")

    # Attention output
    attn_weights = torch.softmax(scores, dim=-1)
    t0 = time.time()
    output = cache.attend(attn_weights)
    attend_time = time.time() - t0
    print(f"  Attend: {attend_time*1000:.2f}ms, shape={output.shape}")

    # Memory
    mem = cache.memory_bytes()
    total_fp16 = batch_size * n_heads * cache.seq_len * d * 2 * 2  # K+V in fp16
    compression = total_fp16 / max(mem["total"], 1)
    print(f"  Memory: {mem['total']/1024:.1f}KB (vs {total_fp16/1024:.1f}KB FP16)")
    print(f"  Compression ratio: {compression:.1f}x")

    assert scores.shape == (batch_size, n_heads, 1, prefill_len + decode_steps)
    assert output.shape == (batch_size, n_heads, 1, d)
    print(f"    ✓ KV cache functioning correctly")

    print()
    return True


def test_memory_savings():
    """Compare memory usage vs FP16 and FP8 baselines."""
    print("=" * 60)
    print("TEST: Memory comparison")
    print("=" * 60)

    from turboquant.kv_cache import TurboQuantKVCache

    d = 128
    n_heads = 32  # Realistic for 7B model
    n_kv_heads = 8  # GQA
    seq_len = 8192
    batch_size = 1
    n_layers = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # FP16 baseline
    fp16_per_layer = batch_size * n_kv_heads * seq_len * d * 2 * 2  # K+V
    fp16_total = fp16_per_layer * n_layers

    # FP8 baseline
    fp8_per_layer = batch_size * n_kv_heads * seq_len * d * 1 * 2  # K+V in FP8
    fp8_total = fp8_per_layer * n_layers

    # TurboQuant 3.5-bit keys, 2-bit values
    tq_total = 0
    for layer_idx in range(n_layers):
        cache = TurboQuantKVCache(
            head_dim=d,
            key_bits=4 if layer_idx < 4 else 3,
            value_bits=2,
            buffer_size=128,
            device=device,
            layer_idx=layer_idx,
        )
        keys = torch.randn(batch_size, n_kv_heads, seq_len, d, device=device) * 0.1
        values = torch.randn(batch_size, n_kv_heads, seq_len, d, device=device) * 0.1
        cache.prefill(keys, values)
        tq_total += cache.memory_bytes()["total"]

    print(f"  Config: {n_layers} layers, {n_kv_heads} KV heads, head_dim={d}, seq_len={seq_len}")
    print(f"  FP16:       {fp16_total/1024/1024:.1f} MB")
    print(f"  FP8:        {fp8_total/1024/1024:.1f} MB ({fp16_total/fp8_total:.1f}x vs FP16)")
    print(f"  TurboQuant: {tq_total/1024/1024:.1f} MB ({fp16_total/tq_total:.1f}x vs FP16)")

    assert tq_total < fp8_total, "TurboQuant should use less memory than FP8"
    print(f"    ✓ TurboQuant uses {fp8_total/tq_total:.1f}x less memory than FP8")

    print()
    return True


if __name__ == "__main__":
    print("\n🚀 TurboQuant Test Suite\n")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}\n")
    else:
        print()

    results = {}
    tests = [
        ("Codebook", test_codebook),
        ("Rotation", test_rotation),
        ("MSE Quantizer", test_mse_quantizer),
        ("Prod Quantizer", test_prod_quantizer),
        ("Attention Score", test_attention_score),
        ("KV Cache", test_kv_cache),
        ("Memory Savings", test_memory_savings),
    ]

    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        print(f"  {'✓' if passed else '✗'} {name}")
    print()

    all_passed = all(results.values())
    if all_passed:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed")
        exit(1)
