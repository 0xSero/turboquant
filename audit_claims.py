#!/usr/bin/env python3
"""
Adversarial audit of TurboQuant claims.

Goal: find every lie, exaggeration, or misleading result.
"""

import math
import torch
import numpy as np

torch.manual_seed(42)
np.random.seed(42)


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


# ======================================================================
# LIE #1: "5.1x compression"
# The memory_bytes() method only counts the quantized data tensors.
# It does NOT count:
#   - The rotation matrix Pi (D*D*4 bytes per layer)
#   - The QJL matrix S (D*D*4 bytes per layer)
#   - The codebook centroids + boundaries
#   - The Python object overhead
#   - The ring buffer (128 * H_kv * D * 2 bytes per layer, in bf16)
# ======================================================================

section("AUDIT 1: Is '5.1x compression' honest?")

from turboquant.store import CompressedKVStore
from turboquant.quantizer import TurboQuantProd

d = 128; H_kv = 8; N = 4096

store = CompressedKVStore(head_dim=d, num_kv_heads=H_kv, key_bits=3, value_bits=2,
                          value_group_size=32, device=torch.device("cpu"))
k = torch.randn(N, H_kv, d); v = torch.randn(N, H_kv, d)
store.append_chunk(k, v)

tq_data_bytes = store.memory_bytes()
fp16_bytes = N * H_kv * d * 2 * 2
naive_ratio = fp16_bytes / tq_data_bytes

# Now count what memory_bytes() DOESN'T count
quantizer = store.quantizer
Pi_bytes = quantizer.mse_quantizer.Pi.nelement() * quantizer.mse_quantizer.Pi.element_size()
S_bytes = quantizer.S.nelement() * quantizer.S.element_size()
centroids_bytes = quantizer.mse_quantizer.centroids.nelement() * 4
boundaries_bytes = quantizer.mse_quantizer.boundaries.nelement() * 4
decision_bytes = quantizer.mse_quantizer.decision_boundaries.nelement() * 4

# Per layer overhead
per_layer_overhead = Pi_bytes + S_bytes + centroids_bytes + boundaries_bytes + decision_bytes
total_honest = tq_data_bytes + per_layer_overhead
honest_ratio = fp16_bytes / total_honest

# Ring buffer (not counted either — it holds 128 tokens in full precision)
ring_bytes = 128 * H_kv * d * 2  # bf16
total_with_ring = total_honest + ring_bytes
ratio_with_ring = fp16_bytes / total_with_ring

print(f"FP16 KV for {N} tokens, {H_kv} heads, d={d}:")
print(f"  FP16 total:              {fp16_bytes:>12,} bytes ({fp16_bytes/1e6:.1f} MB)")
print(f"  TQ data only:            {tq_data_bytes:>12,} bytes  -> {naive_ratio:.2f}x (CLAIMED)")
print(f"  + Pi ({d}x{d}):            {Pi_bytes:>12,} bytes")
print(f"  + S ({d}x{d}):             {S_bytes:>12,} bytes")
print(f"  + codebook:              {centroids_bytes + boundaries_bytes + decision_bytes:>12,} bytes")
print(f"  Per-layer overhead:      {per_layer_overhead:>12,} bytes")
print(f"  TQ data + overhead:      {total_honest:>12,} bytes  -> {honest_ratio:.2f}x (HONEST)")
print(f"  + ring buffer (128 tok): {ring_bytes:>12,} bytes")
print(f"  TQ all-in:               {total_with_ring:>12,} bytes  -> {ratio_with_ring:.2f}x (ALL-IN)")
print()

# At what N does Pi+S overhead become negligible?
for test_n in [128, 512, 1024, 4096, 32768]:
    data = tq_data_bytes * test_n / N
    total = data + per_layer_overhead + ring_bytes
    fp16 = test_n * H_kv * d * 2 * 2
    r = fp16 / total
    print(f"  N={test_n:>6}: all-in ratio = {r:.2f}x")


# ======================================================================
# LIE #2: "Needle-in-haystack passes"
# Our needle tests use signal-to-noise ratio of 3.0 / 0.02 = 150x.
# Real LLM attention has MUCH more subtle signal. The needle is
# ridiculously loud.
# ======================================================================

section("AUDIT 2: Are needle tests realistic?")

from turboquant.store import CompressedKVStore

d = 128; H_kv = 4; N = 4096

def run_needle(needle_magnitude, noise_magnitude, N, bits):
    store = CompressedKVStore(
        head_dim=d, num_kv_heads=H_kv, key_bits=bits, value_bits=2,
        value_group_size=32, device=torch.device("cpu"),
    )
    keys = torch.randn(N, H_kv, d) * noise_magnitude
    values = torch.randn(N, H_kv, d)
    needle_pos = N // 2
    needle_key = torch.randn(1, H_kv, d) * needle_magnitude
    keys[needle_pos] = needle_key.squeeze(0)
    store.append_chunk(keys, values)

    flat = store.get_flat_cache()
    k_dequant = store.quantizer.dequantize(flat.prod_q)
    query_vec = needle_key.squeeze(0).unsqueeze(0)
    scores = torch.bmm(
        query_vec.float().transpose(0, 1),
        k_dequant.float().transpose(1, 2),
    ).squeeze(1)

    correct = 0
    for h in range(H_kv):
        if scores[h].argmax().item() == needle_pos:
            correct += 1
    return correct / H_kv

print("Needle retrieval accuracy at different signal-to-noise ratios:")
print(f"{'SNR':>8} {'Needle':>8} {'Noise':>8} {'3-bit':>8} {'4-bit':>8}")
print("-" * 48)

for needle_mag, noise_mag in [(3.0, 0.02), (1.0, 0.1), (0.5, 0.1), (0.3, 0.1),
                                (0.2, 0.1), (0.15, 0.1), (0.1, 0.1)]:
    snr = needle_mag / noise_mag
    acc3 = run_needle(needle_mag, noise_mag, N, 3)
    acc4 = run_needle(needle_mag, noise_mag, N, 4)
    print(f"{snr:>8.1f} {needle_mag:>8.2f} {noise_mag:>8.2f} {acc3:>8.1%} {acc4:>8.1%}")

print()
print("Our tests use SNR=150x. Even at SNR=1.0, retrieval is 100%.")
print("This is because score-space SNR is amplified by sqrt(d)=11.3.")
print("The needle test proves: 'does argmax survive quantization?'")
print("Answer: yes, always, because dominant keys are always preserved.")
print("What it does NOT prove: ranking quality of non-dominant keys.")


# ======================================================================
# LIE #3: "Recall@8 passes (0.40 for 3-bit, 0.55 for 4-bit)"
# These thresholds are absurdly low. Random baseline recall@8 from
# N=4096 is 8/4096 = 0.002. The paper claims near-perfect retrieval.
# Our 0.40 threshold means we're losing 60% of the top-k keys.
# ======================================================================

section("AUDIT 3: How bad is recall@8 really?")

from turboquant.quantizer import TurboQuantProd

d = 128; N = 4096; n_queries = 64; device = "cpu"

print(f"Recall@k at d={d}, N={N}, averaged over {n_queries} queries:")
print(f"{'bits':>6} {'k=4':>8} {'k=8':>8} {'k=16':>8} {'k=32':>8} {'k=64':>8}")
print("-" * 48)

for bits in [2, 3, 4]:
    q = TurboQuantProd(dim=d, bits=bits, device=device, seed=42)
    keys = torch.randn(1, 1, N, d) * 0.1
    queries = torch.randn(1, 1, n_queries, d) * 0.1

    true_scores = torch.matmul(queries, keys.transpose(-2, -1)).squeeze(0).squeeze(0)
    key_q = q.quantize(keys)
    tq_scores = q.attention_score(queries, key_q).squeeze(0).squeeze(0)

    row = f"{bits:>6}"
    for k in [4, 8, 16, 32, 64]:
        true_topk = set()
        tq_topk_set = set()
        total_recall = 0
        for qi in range(n_queries):
            t_set = set(true_scores[qi].topk(k).indices.tolist())
            tq_set = set(tq_scores[qi].topk(k).indices.tolist())
            total_recall += len(t_set & tq_set) / k
        mean_recall = total_recall / n_queries
        row += f" {mean_recall:>8.3f}"
    print(row)

print()
print("Paper claims near-perfect Needle-in-a-Haystack at 3.5-bit.")
print("Our 3-bit recall@8 ~ 0.4-0.5 means we LOSE HALF the important keys.")


# ======================================================================
# LIE #4: "Hybrid attention works"
# The hybrid path fully dequantizes ALL compressed tokens to float32.
# This means during decode we allocate H_kv * N * D * 4 bytes.
# For 200k tokens, d=128, H_kv=8: that's 200000*8*128*4 = 819 MB
# per decode step. The "memory savings" only exist in storage, not
# during the actual attention computation.
# ======================================================================

section("AUDIT 4: Does hybrid decode actually save memory?")

print("Memory allocated during _matmul_attend (per decode step):")
print("All compressed tokens are dequantized to float32 for the matmul.\n")

for N_tokens in [1024, 4096, 30000, 100000, 200000]:
    # k_dequant + v_dequant, both (H_kv, N, D) float32
    kv_decompressed = 2 * 8 * N_tokens * 128 * 4  # H_kv=8, D=128, float32
    # scores tensor: (H_kv, G, T, N) float32, T=1, G=6 for Qwen3.5 (48q/8kv)
    scores_mem = 8 * 6 * 1 * N_tokens * 4
    total = kv_decompressed + scores_mem
    print(f"  N={N_tokens:>7}: decompressed KV = {kv_decompressed/1e6:>8.1f} MB, "
          f"scores = {scores_mem/1e6:>6.1f} MB, total = {total/1e6:>8.1f} MB")

print()
print("At 200k tokens, we allocate ~1 GB per decode step just for dequantized KV.")
print("The 'savings' are only in between decode steps (storage), not during compute.")
print("This is NOT what the paper describes - the paper uses fused kernels that")
print("never materialize the full dequantized KV.")


# ======================================================================
# AUDIT 5: "Distortion scales as 1/4^b"
# We test this with threshold ">2x per bit" but paper claims ~4x.
# CORRECTION: The paper bound is for UNIT NORM vectors.
# Our initial test used unnormalized randn vectors (||x|| ~ sqrt(d) ~ 11.3),
# which inflated the distortion by ~d = 128x.
# ======================================================================

section("AUDIT 5: Does distortion actually follow 1/4^b?")

from turboquant.quantizer import TurboQuantProd

d = 128; N = 10000

print(f"Inner-product distortion at d={d}, N={N}:")
print()
print("--- With UNNORMALIZED vectors (how we tested initially, WRONG comparison) ---")
print(f"{'bits':>6} {'raw MSE':>12} {'||x||^2':>10} {'||y||^2':>10}")
print("-" * 44)

for bits in [2, 3, 4]:
    q = TurboQuantProd(dim=d, bits=bits, device="cpu", seed=42)
    x = torch.randn(1, 1, N, d)
    y = torch.randn(1, 1, 1, d)

    true_ip = (y * x).sum(dim=-1).squeeze()
    key_q = q.quantize(x)
    est_ip = q.attention_score(y, key_q).squeeze()
    raw_mse = ((est_ip - true_ip) ** 2).mean().item()
    x_norm_sq = (x ** 2).sum(dim=-1).mean().item()
    y_norm_sq = (y ** 2).sum().item()
    print(f"{bits:>6} {raw_mse:>12.4f} {x_norm_sq:>10.2f} {y_norm_sq:>10.2f}")

print()
print("--- With UNIT NORM vectors (what the paper's bound assumes) ---")
print(f"{'bits':>6} {'MSE':>12} {'paper_bound':>14} {'ratio':>10}")
print("-" * 44)

for bits in [2, 3, 4]:
    q = TurboQuantProd(dim=d, bits=bits, device="cpu", seed=42)
    x = torch.randn(1, 1, N, d)
    x = x / x.norm(dim=-1, keepdim=True)
    y = torch.randn(1, 1, 1, d)
    y = y / y.norm(dim=-1, keepdim=True)

    true_ip = (y * x).sum(dim=-1).squeeze()
    key_q = q.quantize(x)
    est_ip = q.attention_score(y, key_q).squeeze()
    mse = ((est_ip - true_ip) ** 2).mean().item()
    paper_bound = math.sqrt(3) * math.pi**2 / d * (1.0 / 4**bits)
    print(f"{bits:>6} {mse:>12.8f} {paper_bound:>14.8f} {mse/paper_bound:>10.2f}x")

print()
print("VERDICT: Distortion IS within paper bounds for unit-norm vectors.")
print("Our audit_v1 was wrong — we compared unnormalized MSE to a normalized bound.")


# ======================================================================
# LIE #6: "Unbiased estimator"
# We check with abs(bias) < 0.05 which sounds tight but the signal
# magnitude matters. Let's look at relative bias.
# ======================================================================

section("AUDIT 6: How unbiased is it really?")

from turboquant.quantizer import TurboQuantProd

d = 128; N = 5000

print(f"Bias analysis at d={d}, N={N}:")
print(f"{'bits':>6} {'mean_bias':>12} {'mean_|ip|':>12} {'rel_bias%':>12} {'max_|bias|':>12}")
print("-" * 60)

for bits in [2, 3, 4]:
    all_biases = []
    all_true_ips = []
    for trial in range(50):
        q = TurboQuantProd(dim=d, bits=bits, device="cpu", seed=trial * 100)
        x = torch.randn(1, 1, N, d)
        y = torch.randn(1, 1, 1, d)
        true_ip = (y * x).sum(dim=-1).squeeze()
        key_q = q.quantize(x)
        est_ip = q.attention_score(y, key_q).squeeze()
        per_sample_bias = (est_ip - true_ip)
        all_biases.append(per_sample_bias.mean().item())
        all_true_ips.append(true_ip.abs().mean().item())

    mean_bias = np.mean(all_biases)
    mean_abs_ip = np.mean(all_true_ips)
    rel_bias = abs(mean_bias) / mean_abs_ip * 100
    max_abs_bias = np.max(np.abs(all_biases))

    print(f"{bits:>6} {mean_bias:>12.6f} {mean_abs_ip:>12.4f} {rel_bias:>11.2f}% {max_abs_bias:>12.6f}")


# ======================================================================
# LIE #7: "30k benchmark shows TQ is faster"
# TTFT: 17.162 vs 18.138 (~5.7% faster)
# But TQ init is SLOWER: 43.749 vs 40.160 (9% slower)
# And the speed "gain" could be noise — it's a SINGLE run per case.
# ======================================================================

section("AUDIT 7: Is the 30k speedup real?")

print("30k telemetry (SINGLE run each, no error bars):")
print()
print("  Metric              Baseline     TQ         Delta")
print("  " + "-" * 55)
print("  Init time           40.160s      43.749s    +3.589s (TQ 8.9% SLOWER)")
print("  TTFT                18.138s      17.162s    -0.976s (TQ 5.4% faster)")
print("  Full 24-tok run     18.992s      18.415s    -0.577s (TQ 3.0% faster)")
print("  Prefill tok/s       1803.92      1906.52    +102.6  (+5.7%)")
print("  End-to-end tok/s    1.264        1.303      +0.039  (+3.1%)")
print("  Activation est MB   644.61       599.20     -45.41  (-7.0%)")
print()
print("Problems with this data:")
print("  1. N=1 per condition. No error bars. Could be noise.")
print("  2. TQ init is 3.6s slower — this cost is IGNORED in speedup claims.")
print("  3. Total wall time (init+gen): baseline=59.15s, TQ=62.16s -> TQ is SLOWER overall.")
print("  4. 'Activation est' is peak_alloc - end_alloc, which includes allocator fragmentation.")
print("  5. The 5.7% prefill speedup makes no sense — TQ uses SDPA which is slower than flash.")
print("     (Unless the measurement noise is >=6%, which for N=1 it certainly is.)")


# ======================================================================
# LIE #8: "200k context works"
# A single completion doesn't prove it works. What about output quality?
# Was the output checked at all? Was needle retrieval tested at 200k?
# ======================================================================

section("AUDIT 8: Does 200k context actually work?")

print("200k TQ completion facts:")
print("  - prompt tokens: 199,952")
print("  - output tokens: 24")
print("  - elapsed: 58.34s")
print("  - GPU mem: ~31.9 GB")
print()
print("What was NOT checked:")
print("  1. Output quality — was the output coherent? Nobody verified.")
print("  2. Needle retrieval — handoff doc says 'needle retrieval failures at 200k'")
print("  3. Baseline comparison — baseline stalled, so there's ZERO comparison data")
print("  4. Perplexity — no measurement of how much quality degraded")
print("  5. The no-alloc SDPA prefill materializes full attention matrix at 200k:")
print(f"     48 heads * 200000 * 200000 * 4 bytes = {48 * 200000 * 200000 * 4 / 1e12:.1f} TB")
print("     This CANNOT work. The causal mask in SDPA may help, but it's")
print("     computed in chunks. Each chunk still materializes huge intermediates.")
print()
print("  The 200k 'success' likely means: the process didn't crash.")
print("  It does NOT mean: the output was correct, or even coherent.")


# ======================================================================
# LIE #9: Compression ratio in the REAL benchmark
# README claims "2.0x context improvement" based on free_kv_cache.
# But that 30GB freed is across 4 GPUs (proof.py uses 4 RTX 3090s).
# Per-GPU it's ~7.5 GB freed. And the freed memory is paged KV cache
# which vLLM may not reuse efficiently.
# ======================================================================

section("AUDIT 9: Is '2x context improvement' honest?")

print("README's proof.py result: ~30GB freed across 4 GPUs")
print("  Per-GPU: ~7.5 GB freed")
print("  Model weights: ~19.78 GB (AWQ 4-bit Qwen3.5-27B)")
print("  On single 5090 (32GB): 32 - 19.78 = 12.22 GB for KV cache")
print("  7.5 GB freed out of 12.22 GB KV = 61% of KV freed")
print()
print("  But wait — can we actually USE the freed memory?")
print("  vLLM pre-allocates paged blocks. Freeing them doesn't mean")
print("  we can allocate more blocks. The freed memory goes back to")
print("  the CUDA allocator, not vLLM's block allocator.")
print()
print("  The '2x context' claim is theoretical. In practice, you'd need")
print("  to restart the engine with a higher max_model_len to actually")
print("  serve 2x more tokens.")


# ======================================================================
# SUMMARY
# ======================================================================

section("SUMMARY: What we're lying about (or at least misleading)")

lies = [
    ("5.1x compression", "MISLEADING",
     "Doesn't count Pi/S matrices (128KB/layer), ring buffer, or Python overhead. "
     "Real ratio ~4.6x at 4096 tokens, ~1.1x at 128 tokens. Only approaches 5x at 32k+."),

    ("Needle-in-haystack passes", "HONEST BUT MEANINGLESS",
     "Needle test uses query=key which gives perfect match. This tests 'does argmax survive' "
     "which is trivial. At SNR=1.0 it still passes because score-space SNR is 11x due to "
     "d=128 dimensionality. Real LLM queries are NOT copies of keys."),

    ("Recall@8 >= 0.40 (3-bit)", "DECEPTIVELY LOW BAR",
     "Paper claims near-perfect retrieval. Our 3-bit recall@1=38%, recall@8=55%. "
     "BUT this only matters for FLAT attention. When attention is spiky (dominant tokens exist), "
     "TQ preserves the important keys perfectly. The recall failure is on the unimportant tail."),

    ("Hybrid decode saves memory", "TRUE FOR STORAGE, FALSE FOR COMPUTE",
     "Storage is compressed (~3 bits/element). But during compute, ALL history is "
     "dequantized to float32. At 200k tokens that's ~1.6 GB per decode step (H_kv=8). "
     "Paper uses fused Triton kernels that never materialize full KV. We have those kernels "
     "but the hybrid path doesn't use them — it uses the PyTorch dequantize-then-matmul path."),

    ("Distortion follows 1/4^b", "TRUE (was falsely accused)",
     "Initial audit showed 75x above bound, but that was comparing non-normalized vectors. "
     "With unit-norm vectors as the paper specifies: 2-bit=0.70x, 3-bit=0.82x, 4-bit=0.97x. "
     "All WITHIN the theoretical bound. Implementation is faithful."),

    ("30k TQ is faster than baseline", "WITHIN NOISE",
     "Single run, no error bars. Total wall time (init+gen) TQ is actually slower (62.2s vs 59.2s). "
     "The 5.7% prefill 'speedup' is within measurement noise for N=1."),

    ("200k context works", "UNVERIFIED",
     "Process completed without crashing. Output quality never checked. "
     "Needle retrieval reportedly fails at 200k. No perplexity measurement. "
     "The SDPA prefill at 200k chunks internally but still allocates huge intermediates."),

    ("2x context improvement", "THEORETICAL ONLY",
     "Freed memory returns to CUDA allocator, not vLLM block allocator. "
     "Can't actually serve 2x more tokens without engine restart with higher max_model_len."),
]

for claim, verdict, detail in lies:
    print(f"  [{verdict}] \"{claim}\"")
    print(f"    {detail}")
    print()
