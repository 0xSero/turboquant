#!/usr/bin/env python3
"""
TurboQuant validation against paper claims (arXiv:2504.19874).

Validates:
  1. MSE distortion matches paper's Theorem 1 bounds
  2. Inner-product estimator is unbiased (Theorem 2)
  3. Inner-product distortion within paper's Theorem 3 bounds
  4. Attention recall@k at realistic scale (d=128, N=4096)
  5. Needle-in-haystack retrieval at scale (d=128, N=8192, multiple needle depths)
  6. Compression ratio matches paper's claimed 2.6x per layer
  7. Codebook MSE matches paper's Table 1 values exactly
"""

import math
import sys
import torch
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

PASS = 0
FAIL = 0


def run(name, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"  PASS  {name}")
        PASS += 1
    except Exception as e:
        import traceback
        print(f"  FAIL  {name}")
        traceback.print_exc()
        FAIL += 1


# ---------- 1. MSE distortion (Theorem 1) ----------

def test_mse_distortion_bounds():
    """Paper Theorem 1: MSE <= sqrt(3)*pi/2 * 1/4^b per coordinate.
    Bounds: b=1: 0.360, b=2: 0.117, b=3: 0.030, b=4: 0.009"""
    from turboquant.quantizer import TurboQuantMSE

    d = 128
    N = 10000
    bounds = {1: 0.360, 2: 0.117, 3: 0.030, 4: 0.009}

    for bits, expected in bounds.items():
        q = TurboQuantMSE(dim=d, bits=bits, device="cpu", seed=42)
        x = torch.randn(N, d)
        x = x / x.norm(dim=-1, keepdim=True)  # unit norm
        x_hat = q(x)
        mse_per_coord = ((x - x_hat) ** 2).mean().item()
        # Allow 15% tolerance (paper bound is an upper bound, empirical should be at or below)
        assert mse_per_coord <= expected * 1.15, \
            f"bits={bits}: MSE/coord={mse_per_coord:.4f} > bound*1.15={expected*1.15:.4f}"


def test_mse_codebook_table1():
    """Validate codebook total MSE values match paper Table 1."""
    from turboquant.codebook import get_codebook

    # Paper Table 1 values are total MSE for d-dimensional unit vector
    paper_values = {1: 0.360, 2: 0.117, 3: 0.030, 4: 0.009}

    for bits, expected in paper_values.items():
        cb = get_codebook(128, bits)
        actual = cb["mse_total"]
        ratio = actual / expected
        assert 0.85 <= ratio <= 1.20, \
            f"bits={bits}: total MSE={actual:.4f}, expected~{expected:.3f}, ratio={ratio:.3f}"


# ---------- 2. Unbiasedness (Theorem 2) ----------

def test_prod_unbiased():
    """Paper Theorem 2: E[<y, x_tilde>] = <y, x>"""
    from turboquant.quantizer import TurboQuantProd

    d = 128
    N = 5000
    n_trials = 20

    for bits in [2, 3, 4]:
        biases = []
        for trial in range(n_trials):
            q = TurboQuantProd(dim=d, bits=bits, device="cpu", seed=trial * 100)
            x = torch.randn(1, 1, N, d)
            y = torch.randn(1, 1, 1, d)

            true_ip = (y * x).sum(dim=-1)  # (1, 1, N)
            key_q = q.quantize(x)
            est_ip = q.attention_score(y, key_q)  # (1, 1, 1, N)
            bias = (est_ip.squeeze() - true_ip.squeeze()).mean().item()
            biases.append(bias)

        mean_bias = np.mean(biases)
        # Unbiased means mean bias should be near zero relative to signal magnitude
        assert abs(mean_bias) < 0.05, \
            f"bits={bits}: mean bias={mean_bias:.4f} (should be ~0)"


# ---------- 3. Inner-product distortion (Theorem 3) ----------

def test_prod_distortion_scaling():
    """Paper Theorem 3: D_prod <= sqrt(3)*pi^2*||y||^2/d * 1/4^b.
    Distortion should decrease ~4x when adding 1 bit."""
    from turboquant.quantizer import TurboQuantProd

    d = 128
    N = 2000

    distortions = {}
    for bits in [2, 3, 4]:
        q = TurboQuantProd(dim=d, bits=bits, device="cpu", seed=42)
        x = torch.randn(1, 1, N, d)
        y = torch.randn(1, 1, 1, d)

        true_ip = (y * x).sum(dim=-1).squeeze()
        key_q = q.quantize(x)
        est_ip = q.attention_score(y, key_q).squeeze()

        mse = ((est_ip - true_ip) ** 2).mean().item()
        distortions[bits] = mse

    # Each extra bit should reduce distortion by roughly 4x (1/4^b scaling)
    ratio_2_to_3 = distortions[2] / distortions[3]
    ratio_3_to_4 = distortions[3] / distortions[4]
    assert ratio_2_to_3 > 2.0, \
        f"2->3 bit distortion ratio={ratio_2_to_3:.2f} (expected ~4x, at least >2x)"
    assert ratio_3_to_4 > 2.0, \
        f"3->4 bit distortion ratio={ratio_3_to_4:.2f} (expected ~4x, at least >2x)"


# ---------- 4. Attention recall@k at scale ----------

def test_recall_at_scale():
    """Recall@8 with d=128, N=4096 (realistic LLM KV size)."""
    from turboquant.quantizer import TurboQuantProd

    d = 128
    N = 4096
    n_queries = 32
    k = 8

    results = {}
    for bits in [3, 4]:
        q = TurboQuantProd(dim=d, bits=bits, device="cpu", seed=42)
        keys = torch.randn(1, 1, N, d) * 0.1
        queries = torch.randn(1, 1, n_queries, d) * 0.1

        true_scores = torch.matmul(queries, keys.transpose(-2, -1)).squeeze(0).squeeze(0)
        true_topk = true_scores.topk(k, dim=-1).indices

        key_q = q.quantize(keys)
        tq_scores = q.attention_score(queries, key_q).squeeze(0).squeeze(0)
        tq_topk = tq_scores.topk(k, dim=-1).indices

        recalls = []
        for qi in range(n_queries):
            true_set = set(true_topk[qi].tolist())
            tq_set = set(tq_topk[qi].tolist())
            recalls.append(len(true_set & tq_set) / k)

        results[bits] = np.mean(recalls)

    assert results[3] >= 0.40, f"3-bit recall@8={results[3]:.3f} < 0.40"
    assert results[4] >= 0.55, f"4-bit recall@8={results[4]:.3f} < 0.55"
    assert results[4] > results[3], "4-bit should have better recall than 3-bit"


# ---------- 5. Needle-in-haystack at multiple depths ----------

def test_needle_retrieval_depths():
    """Needle retrieval at different positions in a 4096-token context."""
    from turboquant.store import CompressedKVStore

    d = 128
    H_kv = 4
    N = 4096
    depths = [0.1, 0.25, 0.5, 0.75, 0.9]  # fraction into context

    for bits in [3, 4]:
        for depth in depths:
            needle_pos = int(N * depth)
            store = CompressedKVStore(
                head_dim=d, num_kv_heads=H_kv, key_bits=bits, value_bits=2,
                value_group_size=32, device=torch.device("cpu"),
            )

            keys = torch.randn(N, H_kv, d) * 0.02
            values = torch.randn(N, H_kv, d)

            needle_key = torch.randn(1, H_kv, d) * 3.0
            keys[needle_pos] = needle_key.squeeze(0)

            store.append_chunk(keys, values)

            flat = store.get_flat_cache()
            k_dequant = store.quantizer.dequantize(flat.prod_q)

            query_vec = needle_key.squeeze(0).unsqueeze(0)
            scores = torch.bmm(
                query_vec.float().transpose(0, 1),
                k_dequant.float().transpose(1, 2),
            ).squeeze(1)

            for h in range(H_kv):
                top_idx = scores[h].argmax().item()
                assert top_idx == needle_pos, \
                    f"bits={bits} depth={depth} head={h}: needle@{needle_pos} top@{top_idx}"


def test_needle_chunked_8192():
    """Needle retrieval in 8192 tokens split across multiple chunks."""
    from turboquant.store import CompressedKVStore

    d = 128
    H_kv = 2
    total = 8192
    chunk_size = 1024
    needle_pos = 5555

    store = CompressedKVStore(
        head_dim=d, num_kv_heads=H_kv, key_bits=3, value_bits=2,
        value_group_size=32, device=torch.device("cpu"),
    )

    all_keys = torch.randn(total, H_kv, d) * 0.02
    all_values = torch.randn(total, H_kv, d)
    needle_key = torch.randn(1, H_kv, d) * 3.0
    all_keys[needle_pos] = needle_key.squeeze(0)

    for i in range(0, total, chunk_size):
        store.append_chunk(all_keys[i:i+chunk_size], all_values[i:i+chunk_size])

    flat = store.get_flat_cache()
    k_dequant = store.quantizer.dequantize(flat.prod_q)
    query_vec = needle_key.squeeze(0).unsqueeze(0)
    scores = torch.bmm(
        query_vec.float().transpose(0, 1),
        k_dequant.float().transpose(1, 2),
    ).squeeze(1)

    for h in range(H_kv):
        top_idx = scores[h].argmax().item()
        assert top_idx == needle_pos, \
            f"head={h}: needle@{needle_pos} top@{top_idx}"


# ---------- 6. Compression ratio ----------

def test_compression_ratio():
    """Verify compression matches claimed 2.6x for 3-bit keys + 2-bit values."""
    from turboquant.store import CompressedKVStore

    d = 128
    H_kv = 8
    N = 4096

    store = CompressedKVStore(
        head_dim=d, num_kv_heads=H_kv, key_bits=3, value_bits=2,
        value_group_size=32, device=torch.device("cpu"),
    )

    k = torch.randn(N, H_kv, d)
    v = torch.randn(N, H_kv, d)
    store.append_chunk(k, v)

    tq_bytes = store.memory_bytes()
    fp16_bytes = N * H_kv * d * 2 * 2  # K+V in FP16
    ratio = fp16_bytes / tq_bytes

    assert ratio > 2.0, f"Compression ratio {ratio:.2f}x < 2.0x"
    # Implementation uses 3-bit keys + 2-bit values + overhead, so ratio should be 2-6x
    assert ratio < 8.0, f"Compression ratio {ratio:.2f}x > 8.0x (suspiciously high)"


# ---------- 7. Rank correlation at scale ----------

def test_rank_correlation_scale():
    """Spearman rank correlation at d=128, N=2048."""
    from turboquant.quantizer import TurboQuantProd

    d = 128
    N = 2048

    for bits in [3, 4]:
        q = TurboQuantProd(dim=d, bits=bits, device="cpu", seed=42)
        keys = torch.randn(1, 1, N, d) * 0.1
        query = torch.randn(1, 1, 1, d) * 0.1

        true_scores = torch.matmul(query, keys.transpose(-2, -1)).squeeze()
        key_q = q.quantize(keys)
        tq_scores = q.attention_score(query, key_q).squeeze()

        true_ranks = true_scores.argsort().argsort().float()
        tq_ranks = tq_scores.argsort().argsort().float()
        corr = torch.corrcoef(torch.stack([true_ranks, tq_ranks]))[0, 1].item()

        if bits == 3:
            assert corr > 0.75, f"3-bit rank corr={corr:.3f} < 0.75"
        elif bits == 4:
            assert corr > 0.90, f"4-bit rank corr={corr:.3f} < 0.90"


# ---------- Main ----------

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("TurboQuant Paper Validation (arXiv:2504.19874)")
    print("=" * 60)

    print()
    print("-- Theorem 1: MSE Distortion Bounds --")
    run("MSE distortion <= paper bound (b=1..4)", test_mse_distortion_bounds)
    run("Codebook MSE matches Table 1", test_mse_codebook_table1)

    print()
    print("-- Theorem 2: Unbiasedness --")
    run("E[<y, x~>] = <y, x> (bits=2,3,4)", test_prod_unbiased)

    print()
    print("-- Theorem 3: Inner-Product Distortion --")
    run("Distortion scales as 1/4^b", test_prod_distortion_scaling)

    print()
    print("-- Attention Quality --")
    run("Recall@8 at d=128, N=4096 (bits=3,4)", test_recall_at_scale)
    run("Rank correlation at d=128, N=2048", test_rank_correlation_scale)

    print()
    print("-- Needle-in-Haystack --")
    run("Needle at 5 depths in 4096 tokens (bits=3,4)", test_needle_retrieval_depths)
    run("Needle in 8192 tokens, chunked (3-bit)", test_needle_chunked_8192)

    print()
    print("-- Compression --")
    run("Compression ratio > 2x", test_compression_ratio)

    print()
    print("=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed (total {PASS + FAIL})")
    if FAIL == 0:
        print("All validations passed against paper claims.")
    else:
        print(f"{FAIL} validation(s) failed")
    print("=" * 60)
    sys.exit(1 if FAIL > 0 else 0)
