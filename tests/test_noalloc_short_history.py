"""Reproduction for the no_alloc decode zeros fallback on short history.

When the hybrid gate (``store.get_flat_cache().num_tokens >= 16``) routes a
decode step to the no_alloc fallback and the compressed store is still empty,
the fallback used to return an all-zero tensor instead of attending over the
exact tokens still held by the ring buffer. A prompt within the ring capacity
(default 128) never leaves the ring, so every such session computed attention
from zeros for its first decode steps. The fallback also ignored the caller's
preallocated ``output`` tensor.

The tests drive the REAL patched forward closure built by ``install_hooks`` on
stub vLLM objects (CPU only, no vLLM import) and compare against an
independent SDPA oracle over the ring's stored tokens. They are red on master
(zeros) and green once the fallback routes through ``compute_hybrid_attention``.

Bootstrap note: the integration module needs the whole ``turboquant`` package
and ``turboquant/__init__`` imports ``codebook`` -> scipy. When scipy cannot be
imported in the host environment, it is stubbed in ``sys.modules`` before the
package import; the codebook for this config (d=64, MSE bits 2) is served from
the tracked ``codebook_d64_b2.json``, so scipy is never exercised either way.
"""
import sys
import types

import torch
import torch.nn.functional as F


def _bootstrap_scipy_stub():
    """Stub scipy out of sys.modules when it cannot be imported.

    ``codebook.py`` only uses scipy inside ``compute_lloyd_max_codebook``;
    everything this file touches is served from the tracked codebook JSON.
    """
    try:
        import scipy.integrate  # noqa: F401
        import scipy.special  # noqa: F401
        return
    except Exception:
        pass
    for name in ("scipy", "scipy.integrate", "scipy.special"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    scipy_mod = sys.modules["scipy"]
    scipy_mod.integrate = sys.modules["scipy.integrate"]
    scipy_mod.special = sys.modules["scipy.special"]


_bootstrap_scipy_stub()

from turboquant.integration import vllm as tqv  # noqa: E402

DEVICE = torch.device("cpu")
NUM_KV_HEADS = 2
NUM_Q_HEADS = 4  # GQA ratio 2
HEAD_DIM = 64  # tracked codebook_d64_b2.json via initial key bits 3 -> MSE bits 2
MODE_HYBRID = tqv.MODE_HYBRID


# --- stub vLLM objects ------------------------------------------------------


class StubAttnImpl:
    """Stand-in for the vLLM attention impl object bound by install_hooks."""

    def __init__(self, num_kv_heads, num_heads, head_size):
        self.num_kv_heads = num_kv_heads
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = 1.0 / (head_size ** 0.5)

    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output=None, output_scale=None, output_block_scale=None):
        raise AssertionError(
            "original flash forward must not run on the TQ decode paths"
        )


class StubAttnModule:
    def __init__(self, impl):
        self.impl = impl
        self.num_heads = impl.num_heads


class StubCompilationConfig:
    def __init__(self, ctx):
        self.static_forward_context = ctx


class StubModelRunner:
    def __init__(self, ctx, device):
        self.compilation_config = StubCompilationConfig(ctx)
        self.device = device


class StubAttnMetadata:
    def __init__(self, num_actual_tokens, max_query_len):
        self.num_actual_tokens = num_actual_tokens
        self.max_query_len = max_query_len


def _install_patched_forward(ring_capacity=128):
    """Build the real patched forward closure via install_hooks (mode=hybrid,
    no_alloc=True). The stub has no do_kv_cache_update, so capture happens in
    forward via _capture_kv — the same wiring used for vLLM 0.16 FlashInfer."""
    impl = StubAttnImpl(NUM_KV_HEADS, NUM_Q_HEADS, HEAD_DIM)
    module = StubAttnModule(impl)
    runner = StubModelRunner({"layer0": module}, DEVICE)
    states = tqv.install_hooks(
        runner,
        key_bits=2,
        value_bits=2,
        ring_capacity=ring_capacity,
        mode=MODE_HYBRID,
        no_alloc=True,
    )
    return states["layer0"], impl


def _exact_ring_reference(q, ring_k, ring_v, scale):
    """Independent oracle: SDPA over the ring's stored tokens.

    q: (T, num_q_heads, head_dim); ring_k/ring_v: (N, num_kv_heads, head_dim).
    GQA mapping: query head h attends kv head h // gqa_ratio, matching the
    repeat_interleave expansion used by the exact-only attention path.
    """
    gqa_ratio = NUM_Q_HEADS // NUM_KV_HEADS
    k = ring_k.repeat_interleave(gqa_ratio, dim=1).transpose(0, 1).float()
    v = ring_v.repeat_interleave(gqa_ratio, dim=1).transpose(0, 1).float()
    q_t = q.transpose(0, 1).float()
    out = F.scaled_dot_product_attention(q_t, k, v, scale=scale)
    return out.transpose(0, 1)  # (T, num_q_heads, head_dim)


def _prefill_into_ring(impl, state, prefill_len):
    """Run a prefill step that lands entirely in the exact ring."""
    k = torch.randn(prefill_len, NUM_KV_HEADS, HEAD_DIM)
    v = torch.randn(prefill_len, NUM_KV_HEADS, HEAD_DIM)
    q = torch.randn(prefill_len, NUM_Q_HEADS, HEAD_DIM)
    md = StubAttnMetadata(num_actual_tokens=prefill_len, max_query_len=prefill_len)
    impl.forward(0, q, k, v, None, md)

    assert state.engine.ring.size == prefill_len
    assert state.store.num_tokens == 0
    assert state.store.get_flat_cache() is None


def test_decode_short_history_matches_exact_ring_oracle_with_output():
    """Red on master: a decode step right after a short prefill must attend
    over the ring's exact tokens instead of returning zeros, and must fill
    the caller's preallocated output."""
    torch.manual_seed(0)
    state, impl = _install_patched_forward()
    _prefill_into_ring(impl, state, prefill_len=6)

    # Decode step: 2D flat query (as vLLM no_alloc passes it) with a
    # preallocated output. Capture ingests this token's KV before attention,
    # so the ring holds 7 exact tokens when it runs.
    dk = torch.randn(1, NUM_KV_HEADS, HEAD_DIM)
    dv = torch.randn(1, NUM_KV_HEADS, HEAD_DIM)
    dq = torch.randn(1, NUM_Q_HEADS * HEAD_DIM)
    md = StubAttnMetadata(num_actual_tokens=1, max_query_len=1)
    output = torch.zeros(1, NUM_Q_HEADS * HEAD_DIM, dtype=dq.dtype)

    returned = impl.forward(0, dq, dk, dv, None, md, output=output)

    assert state.engine.ring.size == 7
    assert state.store.get_flat_cache() is None  # exact-only path, no dequantization
    assert returned is output
    ret = returned.view(1, NUM_Q_HEADS, HEAD_DIM)
    assert ret.abs().max() > 0, (
        "no_alloc decode fallback returned all-zeros and left the "
        "preallocated output unfilled despite exact ring tokens"
    )

    ring_k, ring_v = state.engine.ring.peek()
    expected = _exact_ring_reference(
        dq.view(1, NUM_Q_HEADS, HEAD_DIM), ring_k, ring_v, impl.scale
    )
    assert torch.allclose(ret, expected, atol=1e-4, rtol=1e-4)


def test_decode_short_history_matches_exact_ring_oracle_without_output():
    """Same scenario, 3D query and no preallocated output: the fallback must
    return the attention result itself, not zeros."""
    torch.manual_seed(1)
    state, impl = _install_patched_forward()
    _prefill_into_ring(impl, state, prefill_len=7)

    dk = torch.randn(1, NUM_KV_HEADS, HEAD_DIM)
    dv = torch.randn(1, NUM_KV_HEADS, HEAD_DIM)
    dq = torch.randn(1, NUM_Q_HEADS, HEAD_DIM)  # 3D query
    md = StubAttnMetadata(num_actual_tokens=1, max_query_len=1)

    returned = impl.forward(0, dq, dk, dv, None, md)

    assert state.engine.ring.size == 8
    assert returned.abs().max() > 0, (
        "no_alloc decode fallback returned all-zeros despite exact ring tokens"
    )

    ring_k, ring_v = state.engine.ring.peek()
    expected = _exact_ring_reference(dq, ring_k, ring_v, impl.scale)
    assert returned.shape == expected.shape
    assert torch.allclose(returned, expected, atol=1e-4, rtol=1e-4)


def test_cold_decode_single_ring_token_not_zeroed():
    """Even with no prior prefill, capture ingests the decode token's own KV
    before attention, so the ring holds exactly 1 token — it must be attended
    to, not zeroed. Uses a bf16 output to exercise the dtype cast on copy."""
    torch.manual_seed(2)
    state, impl = _install_patched_forward()

    dq = torch.randn(1, NUM_Q_HEADS * HEAD_DIM)
    dk = torch.randn(1, NUM_KV_HEADS, HEAD_DIM)
    dv = torch.randn(1, NUM_KV_HEADS, HEAD_DIM)
    md = StubAttnMetadata(num_actual_tokens=1, max_query_len=1)
    output = torch.zeros(1, NUM_Q_HEADS * HEAD_DIM, dtype=torch.bfloat16)

    returned = impl.forward(0, dq, dk, dv, None, md, output=output)

    assert state.engine.ring.size == 1
    assert returned is output
    assert returned.abs().max() > 0

    rk, rv = state.engine.ring.peek()
    expected = _exact_ring_reference(
        dq.view(1, NUM_Q_HEADS, HEAD_DIM), rk, rv, impl.scale
    )
    assert torch.allclose(
        returned.view(1, NUM_Q_HEADS, HEAD_DIM).float(), expected,
        atol=2e-2, rtol=2e-2,
    )


def test_hybrid_path_with_compressed_history_still_works():
    """Regression guard for the sibling branch: once the store crosses the
    16-token threshold, decode goes through the compressed+exact hybrid path
    and still produces a finite, non-zero result in the preallocated output."""
    torch.manual_seed(3)
    ring_capacity = 32
    state, impl = _install_patched_forward(ring_capacity=ring_capacity)

    # Prefill 2x capacity: the first half is compressed into the store, the
    # rest fills the ring.
    prefill_len = 2 * ring_capacity
    k = torch.randn(prefill_len, NUM_KV_HEADS, HEAD_DIM)
    v = torch.randn(prefill_len, NUM_KV_HEADS, HEAD_DIM)
    q = torch.randn(prefill_len, NUM_Q_HEADS, HEAD_DIM)
    md = StubAttnMetadata(num_actual_tokens=prefill_len, max_query_len=prefill_len)
    impl.forward(0, q, k, v, None, md)

    assert state.store.num_tokens == ring_capacity
    assert state.engine.ring.size == ring_capacity

    # Decode: capture overflows the full ring into the store, pushing it well
    # past the hybrid threshold.
    dk = torch.randn(1, NUM_KV_HEADS, HEAD_DIM)
    dv = torch.randn(1, NUM_KV_HEADS, HEAD_DIM)
    dq = torch.randn(1, NUM_Q_HEADS * HEAD_DIM)
    md = StubAttnMetadata(num_actual_tokens=1, max_query_len=1)
    output = torch.zeros(1, NUM_Q_HEADS * HEAD_DIM, dtype=dq.dtype)

    returned = impl.forward(0, dq, dk, dv, None, md, output=output)

    flat = state.store.get_flat_cache()
    assert flat is not None and flat.num_tokens >= 16
    assert returned is output
    assert output.shape == (1, NUM_Q_HEADS * HEAD_DIM)
    assert output.abs().max() > 0
    assert torch.isfinite(output).all()
