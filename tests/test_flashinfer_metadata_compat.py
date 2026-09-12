"""FlashInfer metadata compatibility for the patched vLLM attention forward.

Real FlashInferMetadata (vllm/v1/attention/backends/flashinfer.py, the
backend forced by FP8 KV caches) has no ``max_query_len`` attribute, so
classifying prefill vs decode with a bare attribute access crashed the
patched forward with AttributeError. These tests drive the real
``patched()`` closure built by ``_make_patched_forward`` (no vLLM install,
no GPU) against stub metadata objects, plus direct unit tests on the
extracted ``_is_prefill`` helper. Only the surrounding objects (metadata,
orig_fn) are stubbed — the code under fix always runs for real.

``turboquant.integration.vllm`` is imported directly (with a repo-root
sys.path bootstrap) because it imports the rest of the ``turboquant``
package anyway; scipy is available in this environment.

head_dim=64 is deliberate: it matches the group_size=32 value quantization
used on the hybrid read path, and its Lloyd-Max codebook ships with the repo.
"""
import os
import sys
from dataclasses import dataclass

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import turboquant.integration.vllm as tq_vllm

HEAD_DIM = 64
NUM_KV_HEADS = 2
NUM_QUERY_HEADS = 2

_ORIG_SENTINEL = object()


class StubMetadata:
    """Attention-metadata stand-in exposing only the attributes set on it."""

    def __init__(self, **attrs):
        self.__dict__.update(attrs)


@dataclass
class FlashInferStubMetadata:
    """Field-for-field mirror of the real FlashInferMetadata.

    Source of truth: vllm/v1/attention/backends/flashinfer.py::
    FlashInferMetadata — field shape stable across vllm v0.16.0 through
    v0.25-dev. The real class has NEITHER ``max_query_len`` NOR
    ``query_start_loc``; its classification signal is the prefill/decode
    token counts. The ``prefill``/``decode`` sub-objects are None iff the
    matching token count is 0, and both are None for cascade attention.
    """

    num_actual_tokens: int
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    prefill: object = None
    decode: object = None
    use_cascade: bool = False


class RecordingOrig:
    """Stand-in for the original attention forward; records every call."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        return _ORIG_SENTINEL


@pytest.fixture(autouse=True)
def _hybrid_mode():
    """Run every test in hybrid mode; restore the module default after."""
    tq_vllm.set_mode(tq_vllm.MODE_HYBRID)
    yield
    tq_vllm.set_mode(tq_vllm.MODE_CAPTURE_ONLY)


@pytest.fixture
def sdpa_calls(monkeypatch):
    """Record scaled_dot_product_attention calls while still computing them."""
    calls = []
    real = torch.nn.functional.scaled_dot_product_attention

    def recording(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(
        torch.nn.functional, "scaled_dot_product_attention", recording
    )
    return calls


def _make_state(ring_capacity=8):
    cfg = tq_vllm.LayerConfig(
        head_dim=HEAD_DIM,
        num_kv_heads=NUM_KV_HEADS,
        num_query_heads=NUM_QUERY_HEADS,
        ring_capacity=ring_capacity,
        device=torch.device("cpu"),
    )
    return tq_vllm._create_layer_state(cfg)


def _make_qkv(num_tokens):
    query = torch.randn(num_tokens, NUM_QUERY_HEADS, HEAD_DIM)
    key = torch.randn(num_tokens, NUM_KV_HEADS, HEAD_DIM)
    value = torch.randn(num_tokens, NUM_KV_HEADS, HEAD_DIM)
    return query, key, value


def _run_forward(state, orig, metadata, num_tokens, no_alloc=True):
    """Drive the real patched() closure; returns (result, query)."""
    patched = tq_vllm._make_patched_forward(orig, state, no_alloc=no_alloc)
    query, key, value = _make_qkv(num_tokens)
    result = patched(object(), object(), query, key, value, object(), metadata)
    return result, query


def test_flashinfer_metadata_no_crash(sdpa_calls):
    # THE reproduction: real FlashInferMetadata (forced by FP8 KV caches)
    # has no max_query_len, so master's bare `attn_metadata.max_query_len
    # > 1` raised AttributeError inside the patched forward.
    state = _make_state()
    orig = RecordingOrig()
    metadata = FlashInferStubMetadata(
        num_actual_tokens=7,
        num_prefills=1,
        num_prefill_tokens=7,
        num_decodes=0,
        num_decode_tokens=0,
    )

    result, query = _run_forward(state, orig, metadata, num_tokens=7)

    assert result.shape == (7, NUM_QUERY_HEADS, HEAD_DIM)
    assert result.dtype == query.dtype
    assert torch.isfinite(result).all()
    assert not torch.all(result == 0)  # real SDPA math, not a zero fallback
    assert len(sdpa_calls) == 1  # classified as prefill: one causal SDPA
    assert orig.calls == []  # flash backend not consulted


def test_multi_sequence_decode_batch_is_not_prefill(sdpa_calls):
    # Three concurrent decode sequences: num_prefill_tokens=0 with decode
    # counts > 0 means decode. This must NOT enter the prefill SDPA path
    # (is_causal SDPA across sequences would contaminate the batch).
    state = _make_state()
    orig = RecordingOrig()
    metadata = FlashInferStubMetadata(
        num_actual_tokens=3,
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=3,
        num_decode_tokens=3,
    )

    result, _ = _run_forward(state, orig, metadata, num_tokens=3)

    assert result.shape == (3, NUM_QUERY_HEADS, HEAD_DIM)
    assert torch.all(result == 0)  # decode fallback zeros, never SDPA output
    assert sdpa_calls == []  # prefill math never entered
    assert orig.calls == []


def test_mixed_prefill_decode_batch_is_prefill(sdpa_calls):
    # Mixed prefill+decode batch: num_prefill_tokens=5 > 0 means prefill.
    # This deliberately mirrors FlashAttention's `max_query_len > 1`
    # handling, which also routes mixed batches down the prefill path.
    state = _make_state()
    orig = RecordingOrig()
    metadata = FlashInferStubMetadata(
        num_actual_tokens=7,
        num_prefills=1,
        num_prefill_tokens=5,
        num_decodes=2,
        num_decode_tokens=2,
    )

    result, _ = _run_forward(state, orig, metadata, num_tokens=7)

    assert result.shape == (7, NUM_QUERY_HEADS, HEAD_DIM)
    assert not torch.all(result == 0)
    assert len(sdpa_calls) == 1  # prefill path taken
    assert orig.calls == []


def test_cascade_attention_batch_routes_to_orig_fn(sdpa_calls):
    # Cascade attention / empty batch: all counts 0 and both the prefill
    # and decode sub-objects None — unclassifiable, so defer to the
    # original backend rather than guess the batch phase.
    state = _make_state()
    orig = RecordingOrig()
    metadata = FlashInferStubMetadata(
        num_actual_tokens=0,
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0,
        num_decode_tokens=0,
        use_cascade=True,
    )

    result, query = _run_forward(state, orig, metadata, num_tokens=1)

    assert result is _ORIG_SENTINEL
    assert len(orig.calls) == 1
    call = orig.calls[0]
    assert len(call) == 10
    assert call[2] is query  # query tensor passed through untouched
    assert call[6] is metadata
    assert sdpa_calls == []  # no TQ math entered


def test_max_query_len_prefill_unchanged(sdpa_calls):
    # FlashAttention-style metadata: max_query_len still decides, even when a
    # misleading query_start_loc is also present — master behavior unchanged.
    state = _make_state()
    orig = RecordingOrig()
    metadata = StubMetadata(
        max_query_len=5, query_start_loc=[0, 1, 1], num_actual_tokens=5
    )

    result, _ = _run_forward(state, orig, metadata, num_tokens=5)

    assert result.shape == (5, NUM_QUERY_HEADS, HEAD_DIM)
    assert len(sdpa_calls) == 1
    assert orig.calls == []


def test_max_query_len_decode_unchanged(sdpa_calls):
    state = _make_state()
    orig = RecordingOrig()
    metadata = StubMetadata(max_query_len=1, num_actual_tokens=1)

    result, _ = _run_forward(state, orig, metadata, num_tokens=1)

    assert result.shape == (1, NUM_QUERY_HEADS, HEAD_DIM)
    assert torch.all(result == 0)
    assert sdpa_calls == []
    assert orig.calls == []


def test_unclassifiable_metadata_routes_to_orig_fn(sdpa_calls):
    # Neither max_query_len nor FlashInfer count fields: never guess the
    # batch phase from token counts — defer to the original flash backend.
    state = _make_state()
    orig = RecordingOrig()
    metadata = StubMetadata(num_actual_tokens=3)

    result, query = _run_forward(state, orig, metadata, num_tokens=3)

    assert result is _ORIG_SENTINEL
    assert len(orig.calls) == 1
    call = orig.calls[0]
    assert len(call) == 10
    assert call[2] is query  # query tensor passed through untouched
    assert call[6] is metadata
    assert call[7:] == (None, None, None)
    assert sdpa_calls == []  # no TQ math entered


def test_query_start_loc_only_metadata_routes_to_orig_fn(sdpa_calls):
    # Honest fallback, documented: no real vLLM metadata class carries
    # query_start_loc without also carrying max_query_len, so a
    # query_start_loc-only object is unclassifiable under the real
    # mechanism and defers to the original backend.
    state = _make_state()
    orig = RecordingOrig()
    metadata = StubMetadata(query_start_loc=[0, 5, 7])

    result, _ = _run_forward(state, orig, metadata, num_tokens=7)

    assert result is _ORIG_SENTINEL
    assert len(orig.calls) == 1
    assert sdpa_calls == []


def test_prefill_no_alloc_without_num_actual_tokens(sdpa_calls):
    # Metadata without num_actual_tokens on the prefill no_alloc path: the
    # token count falls back to the leading tensor dim, no crash.
    state = _make_state()
    orig = RecordingOrig()
    metadata = StubMetadata(max_query_len=7)

    result, _ = _run_forward(state, orig, metadata, num_tokens=7)

    assert result.shape == (7, NUM_QUERY_HEADS, HEAD_DIM)
    assert not torch.all(result == 0)
    assert len(sdpa_calls) == 1
    assert orig.calls == []


def test_hybrid_decode_without_num_actual_tokens(sdpa_calls):
    # Same guard on the hybrid decode path, with real captured history so
    # compute_hybrid_attention actually runs.
    state = _make_state()
    _, key, value = _make_qkv(20)
    state.engine.ingest_prefill(key, value, 20)
    state.engine.ingest_prefill(key, value, 20)  # store holds 24 tokens
    orig = RecordingOrig()
    metadata = StubMetadata(max_query_len=1)  # decode, no num_actual_tokens

    result, _ = _run_forward(state, orig, metadata, num_tokens=1)

    assert result.shape == (1, NUM_QUERY_HEADS, HEAD_DIM)
    assert torch.isfinite(result).all()
    assert not torch.all(result == 0)  # real hybrid attention math ran
    assert sdpa_calls == []  # decode: prefill SDPA never entered
    assert orig.calls == []


def test_is_prefill_max_query_len():
    assert tq_vllm._is_prefill(StubMetadata(max_query_len=5)) is True
    assert tq_vllm._is_prefill(StubMetadata(max_query_len=1)) is False
    # max_query_len wins even when other fields disagree
    assert (
        tq_vllm._is_prefill(StubMetadata(max_query_len=5, query_start_loc=[0, 1, 2]))
        is True
    )
    assert (
        tq_vllm._is_prefill(
            StubMetadata(max_query_len=1, num_prefill_tokens=5, num_decode_tokens=2)
        )
        is False
    )
    # explicit None falls through to the FlashInfer count fields
    assert (
        tq_vllm._is_prefill(StubMetadata(max_query_len=None, num_prefill_tokens=7))
        is True
    )


def test_is_prefill_flashinfer_counts():
    # prefill
    assert (
        tq_vllm._is_prefill(
            FlashInferStubMetadata(
                num_actual_tokens=7,
                num_prefills=1,
                num_prefill_tokens=7,
                num_decodes=0,
                num_decode_tokens=0,
            )
        )
        is True
    )
    # mixed batch counts as prefill (mirrors max_query_len > 1 semantics)
    assert (
        tq_vllm._is_prefill(
            FlashInferStubMetadata(
                num_actual_tokens=7,
                num_prefills=1,
                num_prefill_tokens=5,
                num_decodes=2,
                num_decode_tokens=2,
            )
        )
        is True
    )
    # pure decode
    assert (
        tq_vllm._is_prefill(
            FlashInferStubMetadata(
                num_actual_tokens=3,
                num_prefills=0,
                num_prefill_tokens=0,
                num_decodes=3,
                num_decode_tokens=3,
            )
        )
        is False
    )
    # all counts 0 (empty batch / cascade attention)
    assert (
        tq_vllm._is_prefill(
            FlashInferStubMetadata(
                num_actual_tokens=0,
                num_prefills=0,
                num_prefill_tokens=0,
                num_decodes=0,
                num_decode_tokens=0,
                use_cascade=True,
            )
        )
        is None
    )
    # request-count fallbacks when the token counters are absent
    assert tq_vllm._is_prefill(StubMetadata(num_prefills=1)) is True
    assert tq_vllm._is_prefill(StubMetadata(num_prefills=0, num_decodes=3)) is False


def test_is_prefill_unclassifiable():
    assert tq_vllm._is_prefill(StubMetadata()) is None
    assert tq_vllm._is_prefill(StubMetadata(num_actual_tokens=9)) is None
    # query_start_loc alone is NOT a classification signal: no real vLLM
    # metadata class has it without max_query_len.
    assert tq_vllm._is_prefill(StubMetadata(query_start_loc=[0, 5, 7])) is None
