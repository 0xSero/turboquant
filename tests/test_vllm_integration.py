"""Tests for vLLM attention backend integration: head inference, TP sharding, and slice copying."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import torch

from turboquant.integration.vllm import _infer_num_query_heads


class DummyImpl:
    def __init__(self, num_heads=None, num_query_heads=None, num_kv_heads=8):
        if num_heads is not None:
            self.num_heads = num_heads
        if num_query_heads is not None:
            self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads


class DummyAttnModule:
    def __init__(self, num_heads=32, num_attention_heads=None, num_query_heads=None):
        self.num_heads = num_heads
        if num_attention_heads is not None:
            self.num_attention_heads = num_attention_heads
        if num_query_heads is not None:
            self.num_query_heads = num_query_heads


def test_infer_num_query_heads_tp_sharded():
    """Under Tensor Parallelism (e.g. TP=2), impl has partitioned heads while attn_module has global heads."""
    attn_module = DummyAttnModule(num_heads=32)
    impl = DummyImpl(num_heads=16, num_kv_heads=4)
    # Must prioritize impl.num_heads (per-rank: 16) over global attn_module.num_heads (32)
    assert _infer_num_query_heads(attn_module, impl) == 16


def test_infer_num_query_heads_explicit_query_heads():
    """impl.num_query_heads takes precedence if explicitly set."""
    attn_module = DummyAttnModule(num_heads=32)
    impl = DummyImpl(num_query_heads=8, num_heads=16, num_kv_heads=4)
    assert _infer_num_query_heads(attn_module, impl) == 8


def test_infer_num_query_heads_fallback_to_attn_module():
    """When impl lacks num_heads, falls back to attn_module."""
    attn_module = DummyAttnModule(num_heads=32)
    impl = DummyImpl(num_heads=None, num_kv_heads=8)
    assert _infer_num_query_heads(attn_module, impl) == 32


def test_infer_num_query_heads_fallback_to_kv_heads():
    """When neither specifies query heads, falls back to impl.num_kv_heads."""
    attn_module = DummyAttnModule(num_heads=None)
    impl = DummyImpl(num_heads=None, num_kv_heads=8)
    assert _infer_num_query_heads(attn_module, impl) == 8


def test_output_slice_view_as_copy():
    """Verify that view_as allows copying between 2D flat buffer and 3D head layout."""
    num_actual = 2
    num_heads = 4
    head_dim = 16

    result = torch.randn(num_actual, num_heads, head_dim)
    result_flat = result.reshape(num_actual, num_heads * head_dim)

    # Case 1: output is 2D buffer
    output_2d = torch.zeros(5, num_heads * head_dim)
    out_slice_2d = output_2d[:num_actual]
    out_slice_2d.copy_(result_flat.view_as(out_slice_2d))
    assert torch.allclose(out_slice_2d, result_flat)

    # Case 2: output is 3D buffer
    output_3d = torch.zeros(5, num_heads, head_dim)
    out_slice_3d = output_3d[:num_actual]
    out_slice_3d.copy_(result.view_as(out_slice_3d))
    assert torch.allclose(out_slice_3d, result)
