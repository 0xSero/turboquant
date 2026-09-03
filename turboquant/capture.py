"""
TurboQuant capture module — bulk ingestion and ring-buffer management.

Handles the write path:
  - Bulk capture from paged KV cache or raw tensors (prefill)
  - Append decode tokens into a small exact ring buffer
  - Flush ring buffer to compressed store only when full or at phase boundaries

Design rule: no per-token quantization on the hot decode path.
"""

from __future__ import annotations

import torch
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from turboquant.store import CompressedKVStore


class RingBuffer:
    """Fixed-size ring buffer for recent exact KV tokens.

    Stores the most recent ``capacity`` tokens in bf16/fp16.
    When full, the oldest chunk is returned for compression.
    """

    __slots__ = (
        "capacity",
        "num_kv_heads",
        "head_dim",
        "device",
        "dtype",
        "_k",
        "_v",
        "_pos",
        "_total_written",
    )

    def __init__(
        self,
        capacity: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.capacity = capacity
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        self._k = torch.zeros(
            capacity, num_kv_heads, head_dim, device=device, dtype=dtype
        )
        self._v = torch.zeros(
            capacity, num_kv_heads, head_dim, device=device, dtype=dtype
        )
        self._pos = 0
        self._total_written = 0

    @property
    def size(self) -> int:
        return self._pos

    @property
    def is_full(self) -> bool:
        return self._pos >= self.capacity

    @property
    def total_written(self) -> int:
        return self._total_written

    def write(
        self, key: torch.Tensor, value: torch.Tensor, num_tokens: int
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Append tokens. Returns (overflow_k, overflow_v) if buffer overflows, else None.

        key/value shapes: (num_tokens, num_kv_heads, head_dim)
        """
        space = self.capacity - self._pos
        overflow_k_parts = []
        overflow_v_parts = []

        offset = 0
        remaining = num_tokens

        while remaining > 0:
            space = self.capacity - self._pos
            if space <= 0:
                # Buffer is full — drain it
                overflow_k_parts.append(self._k[: self._pos].clone())
                overflow_v_parts.append(self._v[: self._pos].clone())
                self._pos = 0
                space = self.capacity

            n = min(remaining, space)
            self._k[self._pos : self._pos + n] = key[offset : offset + n]
            self._v[self._pos : self._pos + n] = value[offset : offset + n]
            self._pos += n
            offset += n
            remaining -= n

        self._total_written += num_tokens

        if overflow_k_parts:
            return (
                torch.cat(overflow_k_parts, dim=0),
                torch.cat(overflow_v_parts, dim=0),
            )
        return None

    def drain(self) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Return all buffered tokens and reset. Returns None if empty."""
        if self._pos == 0:
            return None
        k = self._k[: self._pos].clone()
        v = self._v[: self._pos].clone()
        self._pos = 0
        return k, v

    def peek(self) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Read current buffer contents without draining."""
        if self._pos == 0:
            return None
        return self._k[: self._pos], self._v[: self._pos]

    def reset(self):
        self._pos = 0
        self._total_written = 0


class KVCaptureEngine:
    """Orchestrates capture of KV pairs into a CompressedKVStore.

    Sits between the vLLM attention backend and the compressed store.
    Manages the ring buffer and decides when to flush to the store.
    """

    def __init__(
        self,
        store: "CompressedKVStore",
        ring_capacity: int = 128,
        device: torch.device = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.store = store
        self.ring = RingBuffer(
            capacity=ring_capacity,
            num_kv_heads=store.num_kv_heads,
            head_dim=store.head_dim,
            device=device or store.device,
            dtype=dtype,
        )
        self._prefill_done = False

    @property
    def total_compressed_tokens(self) -> int:
        return self.store.num_tokens

    @property
    def total_buffered_tokens(self) -> int:
        return self.ring.size

    @property
    def total_tokens(self) -> int:
        return self.total_compressed_tokens + self.total_buffered_tokens

    def ingest_prefill(self, key: torch.Tensor, value: torch.Tensor, num_tokens: int):
        """Bulk-capture prefill KV; flush ring overflow to the store.

        Mirrors ``ingest_decode`` so chunked-prefill overflow is never lost.
        key/value: (num_tokens, num_kv_heads, head_dim)
        """
        if num_tokens <= self.ring.capacity:
            overflow = self.ring.write(
                key[:num_tokens], value[:num_tokens], num_tokens
            )
        else:
            n_compress = num_tokens - self.ring.capacity
            self.store.append_chunk(key[:n_compress], value[:n_compress])
            overflow = self.ring.write(
                key[n_compress:num_tokens],
                value[n_compress:num_tokens],
                self.ring.capacity,
            )
        if overflow is not None:
            k_over, v_over = overflow
            self.store.append_chunk(k_over, v_over)
        self._prefill_done = True

    def ingest_prefill_from_paged_cache(
        self,
        kv_cache_tensor: torch.Tensor,
        num_tokens: int,
        block_table: torch.Tensor,
        block_size: int,
    ):
        """Bulk-capture prefill by reading from vLLM's paged KV cache tensor.

        kv_cache_tensor: (2, num_blocks, block_size, num_kv_heads, head_dim)
        block_table: (num_blocks_used,) int — maps logical block idx -> physical
        """
        num_blocks_needed = (num_tokens + block_size - 1) // block_size
        physical_blocks = block_table[:num_blocks_needed]

        keys_list = []
        vals_list = []
        collected = 0

        for i, phys_idx in enumerate(physical_blocks):
            start = 0
            end = min(block_size, num_tokens - collected)
            k_block = kv_cache_tensor[0, phys_idx, start:end]  # (end, heads, dim)
            v_block = kv_cache_tensor[1, phys_idx, start:end]
            keys_list.append(k_block)
            vals_list.append(v_block)
            collected += end

        all_k = torch.cat(keys_list, dim=0)  # (num_tokens, heads, dim)
        all_v = torch.cat(vals_list, dim=0)
        self.ingest_prefill(all_k, all_v, num_tokens)

    def ingest_decode(self, key: torch.Tensor, value: torch.Tensor, num_tokens: int):
        """Append decode tokens. Cheap: just writes to ring buffer.

        Overflow is automatically flushed to the compressed store.
        key/value: (num_tokens, num_kv_heads, head_dim)
        """
        overflow = self.ring.write(key[:num_tokens], value[:num_tokens], num_tokens)
        if overflow is not None:
            k_over, v_over = overflow
            self.store.append_chunk(k_over, v_over)

    def flush(self):
        """Force-flush ring buffer to compressed store."""
        data = self.ring.drain()
        if data is not None:
            k, v = data
            self.store.append_chunk(k, v)

    def reset(self):
        self.ring.reset()
        self.store.reset()
        self._prefill_done = False
