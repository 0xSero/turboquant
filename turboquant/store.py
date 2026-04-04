"""
TurboQuant compressed KV store — owns the quantized historical segment.

Design rules:
  - Chunks are stored in lists; concatenation is deferred (lazy flatten).
  - Flat cache is materialized on first read and invalidated on write.
  - No per-token overhead; all writes are chunk-based.
"""

from __future__ import annotations

import torch
from typing import Optional, NamedTuple

from turboquant.quantizer import TurboQuantProd, ProdQuantized
from turboquant.kv_cache import quantize_values, ValueQuantized


class FlatCache(NamedTuple):
    """Flattened view of compressed KV for fast read access."""
    prod_q: ProdQuantized       # (num_kv_heads, total_tokens, ...)
    value_q: ValueQuantized     # (num_kv_heads, total_tokens, ...)
    num_tokens: int


class CompressedKVStore:
    """Chunked compressed KV store with lazy flattening.

    Keys are quantized via TurboQuantProd (unbiased inner-product estimator).
    Values use symmetric group quantization.
    Chunks are kept in lists until a flat view is requested.
    """

    def __init__(
        self,
        head_dim: int,
        num_kv_heads: int,
        key_bits: int = 3,
        value_bits: int = 2,
        value_group_size: int = 32,
        device: torch.device = None,
        layer_idx: int = 0,
    ):
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.value_group_size = min(value_group_size, head_dim)
        self.device = device or torch.device("cuda")
        self.layer_idx = layer_idx

        self.quantizer = TurboQuantProd(
            dim=head_dim,
            bits=key_bits,
            device=self.device,
            seed=42 + layer_idx * 7,
        )

        self._key_chunks: list[ProdQuantized] = []
        self._value_chunks: list[ValueQuantized] = []
        self._chunk_lengths: list[int] = []

        self._flat: Optional[FlatCache] = None

    @property
    def num_tokens(self) -> int:
        return sum(self._chunk_lengths)

    @property
    def num_chunks(self) -> int:
        return len(self._chunk_lengths)

    def append_chunk(self, key: torch.Tensor, value: torch.Tensor):
        """Quantize and store a chunk of KV pairs.

        key/value: (chunk_len, num_kv_heads, head_dim)
        """
        chunk_len = key.shape[0]

        # Reshape to (1, num_kv_heads, chunk_len, head_dim) for quantizer
        k = key.transpose(0, 1).unsqueeze(0)  # (1, H, T, D)
        v = value.transpose(0, 1).unsqueeze(0)

        key_q = self.quantizer.quantize(k)
        val_q = quantize_values(v, bits=self.value_bits, group_size=self.value_group_size)

        self._key_chunks.append(key_q)
        self._value_chunks.append(val_q)
        self._chunk_lengths.append(chunk_len)
        self._flat = None  # invalidate

    def get_flat_cache(self) -> Optional[FlatCache]:
        """Return a flattened view of all compressed tokens. Cached until next write."""
        if not self._key_chunks:
            return None

        if self._flat is not None:
            return self._flat

        if len(self._key_chunks) == 1:
            kq = self._key_chunks[0]
            vq = self._value_chunks[0]
            flat_kq = _flatten_prod_q(kq)
            flat_vq = _flatten_value_q(vq)
        else:
            flat_kq = _concat_prod_q([_flatten_prod_q(c) for c in self._key_chunks])
            flat_vq = _concat_value_q([_flatten_value_q(c) for c in self._value_chunks])

        self._flat = FlatCache(
            prod_q=flat_kq,
            value_q=flat_vq,
            num_tokens=self.num_tokens,
        )
        return self._flat

    def memory_bytes(self) -> int:
        """Estimate GPU memory used by compressed data."""
        total = 0
        for kq in self._key_chunks:
            total += kq.mse_indices.nelement() * kq.mse_indices.element_size()
            total += kq.qjl_signs.nelement() * kq.qjl_signs.element_size()
            total += kq.residual_norms.nelement() * kq.residual_norms.element_size()
            total += kq.norms.nelement() * kq.norms.element_size()
        for vq in self._value_chunks:
            total += vq.data.nelement() * vq.data.element_size()
            total += vq.scales.nelement() * vq.scales.element_size()
            total += vq.zeros.nelement() * vq.zeros.element_size()
        return total

    def reset(self):
        self._key_chunks.clear()
        self._value_chunks.clear()
        self._chunk_lengths.clear()
        self._flat = None


def _flatten_prod_q(pq: ProdQuantized) -> ProdQuantized:
    """Collapse batch dim: (1, H, T, ...) -> (H, T, ...)."""
    return ProdQuantized(
        mse_indices=pq.mse_indices.reshape(-1, pq.mse_indices.shape[-2], pq.mse_indices.shape[-1]).contiguous(),
        qjl_signs=pq.qjl_signs.reshape(-1, pq.qjl_signs.shape[-2], pq.qjl_signs.shape[-1]).contiguous(),
        residual_norms=pq.residual_norms.reshape(-1, pq.residual_norms.shape[-1]).contiguous(),
        norms=pq.norms.reshape(-1, pq.norms.shape[-1]).contiguous(),
        mse_bits=pq.mse_bits,
    )


def _flatten_value_q(vq: ValueQuantized) -> ValueQuantized:
    """Collapse batch dim: (1, H, T, ...) -> (H, T, ...)."""
    v_bits = vq.bits if len(vq) > 3 else 2
    return ValueQuantized(
        data=vq.data.reshape(-1, vq.data.shape[-2], vq.data.shape[-1]).contiguous(),
        scales=vq.scales.reshape(-1, vq.scales.shape[-2], vq.scales.shape[-1]).contiguous(),
        zeros=vq.zeros.reshape(-1, vq.zeros.shape[-2], vq.zeros.shape[-1]).contiguous(),
        bits=v_bits,
    )


def _concat_prod_q(chunks: list[ProdQuantized]) -> ProdQuantized:
    """Concatenate multiple flattened ProdQuantized along the token dimension."""
    return ProdQuantized(
        mse_indices=torch.cat([c.mse_indices for c in chunks], dim=-2),
        qjl_signs=torch.cat([c.qjl_signs for c in chunks], dim=-2),
        residual_norms=torch.cat([c.residual_norms for c in chunks], dim=-1),
        norms=torch.cat([c.norms for c in chunks], dim=-1),
        mse_bits=chunks[0].mse_bits,
    )


def _concat_value_q(chunks: list[ValueQuantized]) -> ValueQuantized:
    """Concatenate multiple flattened ValueQuantized along the token dimension."""
    v_bits = chunks[0].bits if len(chunks[0]) > 3 else 2
    return ValueQuantized(
        data=torch.cat([c.data for c in chunks], dim=-2),
        scales=torch.cat([c.scales for c in chunks], dim=-2),
        zeros=torch.cat([c.zeros for c in chunks], dim=-2),
        bits=v_bits,
    )
