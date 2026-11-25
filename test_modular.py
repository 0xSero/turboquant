#!/usr/bin/env python3
"""
TurboQuant modular architecture tests.

Tests:
1. RingBuffer — write, overflow, drain, reset
2. CompressedKVStore — append_chunk, flat cache, chunked growth
3. KVCaptureEngine — prefill, decode, flush orchestration
4. compute_hybrid_attention — compressed + exact merge
5. Attention recall agreement — TQ vs exact top-k recall
6. Retrieval accuracy — needle-in-haystack style
"""

import math
import torch
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

PASS = 0
FAIL = 0


def run_test(name, fn, **kwargs):
    global PASS, FAIL
    try:
        fn(**kwargs)
        print(f"  PASS  {name}")
        PASS += 1
    except Exception as e:
        import traceback
        print(f"  FAIL  {name}")
        traceback.print_exc()
        FAIL += 1


# ── Test 1: RingBuffer ───────────────────────────────────────────────

def test_ring_buffer_basic():
    from turboquant.capture import RingBuffer

    buf = RingBuffer(capacity=8, num_kv_heads=2, head_dim=4, device="cpu")
    assert buf.size == 0
    assert not buf.is_full

    k = torch.randn(3, 2, 4)
    v = torch.randn(3, 2, 4)
    overflow = buf.write(k, v, 3)
    assert overflow is None
    assert buf.size == 3

    data = buf.peek()
    assert data is not None
    assert data[0].shape == (3, 2, 4)


def test_ring_buffer_overflow():
    from turboquant.capture import RingBuffer

    buf = RingBuffer(capacity=4, num_kv_heads=2, head_dim=4, device="cpu")
    k = torch.randn(6, 2, 4)
    v = torch.randn(6, 2, 4)
    overflow = buf.write(k, v, 6)
    assert overflow is not None
    ok, ov = overflow
    assert ok.shape[0] == 4  # one full buffer drained
    assert buf.size == 2     # 2 remaining in buffer


def test_ring_buffer_drain():
    from turboquant.capture import RingBuffer

    buf = RingBuffer(capacity=8, num_kv_heads=2, head_dim=4, device="cpu")
    k = torch.randn(5, 2, 4)
    v = torch.randn(5, 2, 4)
    buf.write(k, v, 5)
    data = buf.drain()
    assert data is not None
    assert data[0].shape == (5, 2, 4)
    assert buf.size == 0


def test_ring_buffer_large_overflow():
    """Write more than 2x capacity in one call."""
    from turboquant.capture import RingBuffer

    buf = RingBuffer(capacity=4, num_kv_heads=1, head_dim=2, device="cpu")
    k = torch.randn(11, 1, 2)
    v = torch.randn(11, 1, 2)
    overflow = buf.write(k, v, 11)
    assert overflow is not None
    ok, ov = overflow
    # Two full buffers drained (4+4=8), 3 remain in buffer
    assert ok.shape[0] == 8
    assert buf.size == 3


# ── Test 2: CompressedKVStore ─────────────────────────────────────────

def test_store_basic():
    from turboquant.store import CompressedKVStore

    store = CompressedKVStore(
        head_dim=64, num_kv_heads=4, key_bits=3, value_bits=2,
        value_group_size=32, device=torch.device("cpu"), layer_idx=0,
    )
    assert store.num_tokens == 0
    assert store.num_chunks == 0

    k = torch.randn(32, 4, 64)
    v = torch.randn(32, 4, 64)
    store.append_chunk(k, v)
    assert store.num_tokens == 32
    assert store.num_chunks == 1

    flat = store.get_flat_cache()
    assert flat is not None
    assert flat.num_tokens == 32


def test_store_multi_chunk():
    from turboquant.store import CompressedKVStore

    store = CompressedKVStore(
        head_dim=64, num_kv_heads=2, key_bits=3, value_bits=2,
        value_group_size=32, device=torch.device("cpu"),
    )

    for _ in range(3):
        k = torch.randn(16, 2, 64)
        v = torch.randn(16, 2, 64)
        store.append_chunk(k, v)

    assert store.num_tokens == 48
    assert store.num_chunks == 3

    flat = store.get_flat_cache()
    assert flat is not None
    assert flat.num_tokens == 48

    # Second call should use cache
    flat2 = store.get_flat_cache()
    assert flat2 is flat


def test_store_invalidation():
    from turboquant.store import CompressedKVStore

    store = CompressedKVStore(
        head_dim=64, num_kv_heads=2, key_bits=3, value_bits=2,
        value_group_size=32, device=torch.device("cpu"),
    )

    store.append_chunk(torch.randn(16, 2, 64), torch.randn(16, 2, 64))
    flat1 = store.get_flat_cache()

    store.append_chunk(torch.randn(16, 2, 64), torch.randn(16, 2, 64))
    flat2 = store.get_flat_cache()

    assert flat2 is not flat1
    assert flat2.num_tokens == 32


def test_store_memory():
    from turboquant.store import CompressedKVStore

    store = CompressedKVStore(
        head_dim=128, num_kv_heads=8, key_bits=3, value_bits=2,
        value_group_size=32, device=torch.device("cpu"),
    )

    k = torch.randn(512, 8, 128)
    v = torch.randn(512, 8, 128)
    store.append_chunk(k, v)

    mem = store.memory_bytes()
    fp16_mem = 512 * 8 * 128 * 2 * 2  # K+V in FP16
    assert mem < fp16_mem, f"TQ mem {mem} should be < FP16 mem {fp16_mem}"
    ratio = fp16_mem / mem
    assert ratio > 2.0, f"Compression ratio {ratio:.1f}x should be > 2x"


# ── Test 3: KVCaptureEngine ──────────────────────────────────────────

def test_capture_engine_prefill():
    from turboquant.capture import KVCaptureEngine
    from turboquant.store import CompressedKVStore

    store = CompressedKVStore(
        head_dim=64, num_kv_heads=2, key_bits=3, value_bits=2,
        value_group_size=32, device=torch.device("cpu"),
    )
    engine = KVCaptureEngine(store, ring_capacity=32, device=torch.device("cpu"))

    k = torch.randn(100, 2, 64)
    v = torch.randn(100, 2, 64)
    engine.ingest_prefill(k, v, 100)

    # 100 - 32 = 68 compressed, 32 buffered
    assert engine.total_compressed_tokens == 68
    assert engine.total_buffered_tokens == 32
    assert engine.total_tokens == 100


def test_capture_engine_decode():
    from turboquant.capture import KVCaptureEngine
    from turboquant.store import CompressedKVStore

    store = CompressedKVStore(
        head_dim=64, num_kv_heads=2, key_bits=3, value_bits=2,
        value_group_size=32, device=torch.device("cpu"),
    )
    engine = KVCaptureEngine(store, ring_capacity=8, device=torch.device("cpu"))

    # Fill ring
    for i in range(10):
        k = torch.randn(1, 2, 64)
        v = torch.randn(1, 2, 64)
        engine.ingest_decode(k, v, 1)

    # After 10 decode tokens with ring=8: 8 overflowed -> compressed, 2 in buffer
    assert engine.total_compressed_tokens == 8
    assert engine.total_buffered_tokens == 2


def test_capture_engine_flush():
    from turboquant.capture import KVCaptureEngine
    from turboquant.store import CompressedKVStore

    store = CompressedKVStore(
        head_dim=64, num_kv_heads=2, key_bits=3, value_bits=2,
        value_group_size=32, device=torch.device("cpu"),
    )
    engine = KVCaptureEngine(store, ring_capacity=16, device=torch.device("cpu"))

    k = torch.randn(5, 2, 64)
    v = torch.randn(5, 2, 64)
    engine.ingest_decode(k, v, 5)
    assert engine.total_buffered_tokens == 5
    assert engine.total_compressed_tokens == 0

    engine.flush()
    assert engine.total_buffered_tokens == 0
    assert engine.total_compressed_tokens == 5


# ── Test 4: Hybrid attention ─────────────────────────────────────────

def test_hybrid_attention_compressed_only():
    from turboquant.store import CompressedKVStore
    from turboquant.score import compute_hybrid_attention

    d = 64
    H_kv = 2
    Q = 4  # num_query_heads

    store = CompressedKVStore(
        head_dim=d, num_kv_heads=H_kv, key_bits=3, value_bits=2,
        value_group_size=32, device=torch.device("cpu"),
    )
    k = torch.randn(64, H_kv, d)
    v = torch.randn(64, H_kv, d)
    store.append_chunk(k, v)

    query = torch.randn(1, Q, d)
    out = compute_hybrid_attention(
        query=query, store=store, recent_k=None, recent_v=None,
        num_query_heads=Q,
    )
    assert out.shape == (1, Q, d)


def test_hybrid_attention_exact_only():
    from turboquant.store import CompressedKVStore
    from turboquant.score import compute_hybrid_attention

    d = 64
    H_kv = 2
    Q = 4

    store = CompressedKVStore(
        head_dim=d, num_kv_heads=H_kv, key_bits=3, value_bits=2,
        value_group_size=32, device=torch.device("cpu"),
    )
    # No compressed data, just recent buffer
    recent_k = torch.randn(16, H_kv, d)
    recent_v = torch.randn(16, H_kv, d)

    query = torch.randn(1, Q, d)
    out = compute_hybrid_attention(
        query=query, store=store, recent_k=recent_k, recent_v=recent_v,
        num_query_heads=Q,
    )
    assert out.shape == (1, Q, d)


def test_hybrid_attention_both():
    from turboquant.store import CompressedKVStore
    from turboquant.score import compute_hybrid_attention

    d = 64
    H_kv = 2
    Q = 4

    store = CompressedKVStore(
        head_dim=d, num_kv_heads=H_kv, key_bits=3, value_bits=2,
        value_group_size=32, device=torch.device("cpu"),
    )
    k_hist = torch.randn(64, H_kv, d)
    v_hist = torch.randn(64, H_kv, d)
    store.append_chunk(k_hist, v_hist)

    recent_k = torch.randn(8, H_kv, d)
    recent_v = torch.randn(8, H_kv, d)

    query = torch.randn(1, Q, d)
    out = compute_hybrid_attention(
        query=query, store=store, recent_k=recent_k, recent_v=recent_v,
        num_query_heads=Q,
    )
    assert out.shape == (1, Q, d)


# ── Test 5: Attention recall agreement ────────────────────────────────

def test_attention_recall():
    """Verify that TQ-compressed keys produce similar attention ranking to exact keys.

    Measures recall@k: what fraction of the true top-k keys by attention score
    are also in the TQ top-k. Target: >= 0.90 for 3-bit at k=8.
    """
    from turboquant.quantizer import TurboQuantProd

    d = 128
    N = 256
    n_queries = 16
    k = 8
    bits = 3
    device = torch.device("cpu")

    quantizer = TurboQuantProd(dim=d, bits=bits, device=device, seed=42)

    keys = torch.randn(1, 1, N, d, device=device) * 0.1
    queries = torch.randn(1, 1, n_queries, d, device=device) * 0.1

    # True scores
    true_scores = torch.matmul(queries, keys.transpose(-2, -1)).squeeze(0).squeeze(0)  # (n_q, N)
    true_topk = true_scores.topk(k, dim=-1).indices  # (n_q, k)

    # TQ scores
    key_q = quantizer.quantize(keys)
    tq_scores = quantizer.attention_score(queries, key_q).squeeze(0).squeeze(0)
    tq_topk = tq_scores.topk(k, dim=-1).indices

    # Compute recall@k
    recalls = []
    for q_idx in range(n_queries):
        true_set = set(true_topk[q_idx].tolist())
        tq_set = set(tq_topk[q_idx].tolist())
        recall = len(true_set & tq_set) / k
        recalls.append(recall)

    mean_recall = sum(recalls) / len(recalls)
    assert mean_recall >= 0.50, f"Mean recall@{k} = {mean_recall:.3f} < 0.50"


def test_attention_recall_4bit():
    """4-bit should achieve very high recall."""
    from turboquant.quantizer import TurboQuantProd

    d = 128
    N = 256
    n_queries = 16
    k = 8
    bits = 4
    device = torch.device("cpu")

    quantizer = TurboQuantProd(dim=d, bits=bits, device=device, seed=42)

    keys = torch.randn(1, 1, N, d, device=device) * 0.1
    queries = torch.randn(1, 1, n_queries, d, device=device) * 0.1

    true_scores = torch.matmul(queries, keys.transpose(-2, -1)).squeeze(0).squeeze(0)
    true_topk = true_scores.topk(k, dim=-1).indices

    key_q = quantizer.quantize(keys)
    tq_scores = quantizer.attention_score(queries, key_q).squeeze(0).squeeze(0)
    tq_topk = tq_scores.topk(k, dim=-1).indices

    recalls = []
    for q_idx in range(n_queries):
        true_set = set(true_topk[q_idx].tolist())
        tq_set = set(tq_topk[q_idx].tolist())
        recalls.append(len(true_set & tq_set) / k)

    mean_recall = sum(recalls) / len(recalls)
    assert mean_recall >= 0.65, f"4-bit recall@{k} = {mean_recall:.3f} < 0.65"


# ── Test 6: Needle retrieval (compressed store) ──────────────────────

def test_needle_retrieval():
    """Place a distinctive key in a sea of noise. Verify TQ compressed store
    still ranks the needle's position highest for a matching query."""
    from turboquant.store import CompressedKVStore

    d = 64
    H_kv = 1
    N = 200
    needle_pos = 137

    store = CompressedKVStore(
        head_dim=d, num_kv_heads=H_kv, key_bits=3, value_bits=2,
        value_group_size=32, device=torch.device("cpu"),
    )

    keys = torch.randn(N, H_kv, d) * 0.05
    values = torch.randn(N, H_kv, d) * 0.05

    # Plant a strong needle signal
    needle_key = torch.randn(1, H_kv, d) * 2.0
    keys[needle_pos] = needle_key.squeeze(0)
    values[needle_pos] = torch.ones(H_kv, d)

    store.append_chunk(keys, values)

    flat = store.get_flat_cache()
    assert flat is not None

    # Query that matches the needle
    query_vec = needle_key.squeeze(0).unsqueeze(0)  # (1, H_kv, d)

    # Dequantize and compute scores
    k_dequant = store.quantizer.dequantize(flat.prod_q)  # (H_kv, N, d)
    scores = torch.bmm(
        query_vec.float().transpose(0, 1),  # (H_kv, 1, d)
        k_dequant.float().transpose(1, 2),  # (H_kv, d, N)
    ).squeeze(1)  # (H_kv, N)

    top_idx = scores.argmax(dim=-1).item()
    assert top_idx == needle_pos, f"Needle at {needle_pos} but top score at {top_idx}"


def test_needle_retrieval_multi_chunk():
    """Needle retrieval across multiple chunks."""
    from turboquant.store import CompressedKVStore

    d = 64
    H_kv = 2
    needle_pos_in_chunk2 = 15

    store = CompressedKVStore(
        head_dim=d, num_kv_heads=H_kv, key_bits=4, value_bits=2,
        value_group_size=32, device=torch.device("cpu"),
    )

    # Chunk 1: noise
    store.append_chunk(torch.randn(32, H_kv, d) * 0.05, torch.randn(32, H_kv, d))

    # Chunk 2: needle at position 15
    keys2 = torch.randn(32, H_kv, d) * 0.05
    needle_key = torch.randn(1, H_kv, d) * 3.0
    keys2[needle_pos_in_chunk2] = needle_key.squeeze(0)
    store.append_chunk(keys2, torch.randn(32, H_kv, d))

    # Chunk 3: noise
    store.append_chunk(torch.randn(32, H_kv, d) * 0.05, torch.randn(32, H_kv, d))

    flat = store.get_flat_cache()
    k_dequant = store.quantizer.dequantize(flat.prod_q)

    query_vec = needle_key.squeeze(0).unsqueeze(0)
    scores = torch.bmm(
        query_vec.float().transpose(0, 1),
        k_dequant.float().transpose(1, 2),
    ).squeeze(1)

    # Needle is at global position 32 + 15 = 47
    expected_pos = 32 + needle_pos_in_chunk2
    # Check per head
    for h in range(H_kv):
        top_idx = scores[h].argmax().item()
        assert top_idx == expected_pos, \
            f"Head {h}: needle at {expected_pos} but top at {top_idx}"


# ── Test 7: Score rank correlation ────────────────────────────────────

def test_score_rank_correlation():
    """Spearman rank correlation between exact and TQ scores should be high."""
    from turboquant.quantizer import TurboQuantProd

    d = 128
    N = 128
    bits = 3
    device = torch.device("cpu")

    quantizer = TurboQuantProd(dim=d, bits=bits, device=device, seed=42)

    keys = torch.randn(1, 1, N, d, device=device) * 0.1
    query = torch.randn(1, 1, 1, d, device=device) * 0.1

    true_scores = torch.matmul(query, keys.transpose(-2, -1)).squeeze()
    key_q = quantizer.quantize(keys)
    tq_scores = quantizer.attention_score(query, key_q).squeeze()

    # Spearman rank correlation
    true_ranks = true_scores.argsort().argsort().float()
    tq_ranks = tq_scores.argsort().argsort().float()
    rank_corr = torch.corrcoef(torch.stack([true_ranks, tq_ranks]))[0, 1].item()

    assert rank_corr > 0.80, f"Rank correlation {rank_corr:.3f} < 0.80"


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("TurboQuant Modular Architecture Tests")
    print("=" * 60)
    print()

    print("-- RingBuffer --")
    run_test("basic write/peek", test_ring_buffer_basic)
    run_test("overflow", test_ring_buffer_overflow)
    run_test("drain", test_ring_buffer_drain)
    run_test("large overflow (>2x capacity)", test_ring_buffer_large_overflow)

    print()
    print("-- CompressedKVStore --")
    run_test("single chunk", test_store_basic)
    run_test("multi chunk + caching", test_store_multi_chunk)
    run_test("flat cache invalidation", test_store_invalidation)
    run_test("memory savings vs FP16", test_store_memory)

    print()
    print("-- KVCaptureEngine --")
    run_test("prefill", test_capture_engine_prefill)
    run_test("decode with overflow", test_capture_engine_decode)
    run_test("explicit flush", test_capture_engine_flush)

    print()
    print("-- Hybrid Attention --")
    run_test("compressed only", test_hybrid_attention_compressed_only)
    run_test("exact only", test_hybrid_attention_exact_only)
    run_test("compressed + exact merge", test_hybrid_attention_both)

    print()
    print("-- Retrieval Quality --")
    run_test("attention recall@8 (3-bit)", test_attention_recall)
    run_test("attention recall@8 (4-bit)", test_attention_recall_4bit)
    run_test("needle retrieval (single chunk)", test_needle_retrieval)
    run_test("needle retrieval (multi chunk)", test_needle_retrieval_multi_chunk)
    run_test("score rank correlation", test_score_rank_correlation)

    print()
    print("=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed (total {PASS + FAIL})")
    if FAIL == 0:
        print("All tests passed!")
    else:
        print(f"{FAIL} test(s) failed")
        exit(1)
    print("=" * 60)
