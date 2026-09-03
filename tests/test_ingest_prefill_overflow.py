"""Reproduction for silent KV data loss in ``KVCaptureEngine.ingest_prefill``.

``RingBuffer.write`` returns overflow tokens when it is full; ``ingest_prefill``
must forward that overflow to the store (mirroring ``ingest_decode``) or tokens
are silently dropped. These tests exercise the real code path -- they do not
mock ``ingest_prefill`` or ``ring.write`` -- and assert no token is lost across
chunked prefill calls.

The module is loaded with ``importlib.util`` because the ``turboquant`` package
``__init__`` imports scipy (an optional dep), while ``capture.py`` itself only
needs torch.
"""
import importlib.util
import os

import torch


def _load_capture_module():
    here = os.path.dirname(__file__)
    repo_root = os.path.abspath(os.path.join(here, ".."))
    capture_path = os.path.join(repo_root, "turboquant", "capture.py")
    spec = importlib.util.spec_from_file_location("tq_capture", capture_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


capture = _load_capture_module()
KVCaptureEngine = capture.KVCaptureEngine


class MockStore:
    """Minimal stand-in for ``CompressedKVStore``; tracks appended tokens."""

    def __init__(self, num_kv_heads, head_dim, device):
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self._num_tokens = 0

    @property
    def num_tokens(self):
        return self._num_tokens

    def append_chunk(self, key, value):
        self._num_tokens += key.shape[0]

    def reset(self):
        self._num_tokens = 0


def _make_engine(ring_capacity, num_tokens, num_kv_heads=2, head_dim=4):
    device = torch.device("cpu")
    store = MockStore(num_kv_heads, head_dim, device)
    engine = KVCaptureEngine(store, ring_capacity=ring_capacity, device=device)
    k = torch.randn(num_tokens, num_kv_heads, head_dim, device=device)
    v = torch.randn(num_tokens, num_kv_heads, head_dim, device=device)
    return engine, k, v


def test_chunked_prefill_no_data_loss():
    engine, k, v = _make_engine(ring_capacity=8, num_tokens=12)
    engine.ingest_prefill(k, v, 12)  # 4 -> store, 8 -> ring: total 12
    engine.ingest_prefill(k, v, 12)  # 4 -> store, ring overflows 8, 8 -> ring
    assert engine.total_tokens == 24  # master drops the overflow (== 16)


def test_single_prefill_no_regression():
    engine, k, v = _make_engine(ring_capacity=8, num_tokens=12)
    engine.ingest_prefill(k, v, 12)
    assert engine.total_tokens == 12


def test_small_prefill_no_regression():
    engine, k, v = _make_engine(ring_capacity=8, num_tokens=4)
    engine.ingest_prefill(k, v, 4)
    assert engine.total_tokens == 4
