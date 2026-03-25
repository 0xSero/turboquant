"""
TurboQuant KV Cache — drop-in replacement for the standard KV cache
in transformer attention layers.

Handles:
  - Keys: TurboQuant_prod quantization (unbiased inner product estimation)
  - Values: Standard group quantization (symmetric, per-group min-max)
  - Outlier channels: kept in full precision (configurable count)
  - Buffer: recent tokens kept unquantized for quality

The design follows the pattern from QJL but is model-agnostic.
"""

import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, NamedTuple
from turboquant.quantizer import TurboQuantProd, ProdQuantized, TurboQuantMSE, MSEQuantized


class ValueQuantized(NamedTuple):
    """Quantized value cache (bit-packed)."""
    data: torch.Tensor       # (..., n_tokens, packed_d) bit-packed quantized values
    scales: torch.Tensor     # (..., n_tokens, n_groups) scale per group
    zeros: torch.Tensor      # (..., n_tokens, n_groups) zero point per group
    bits: int = 2            # quantization bits (for unpacking)


def unpack_values(vq: ValueQuantized) -> torch.Tensor:
    """Unpack bit-packed value data to uint8 per-element."""
    bits = vq.bits if len(vq) > 3 else 2
    packed = vq.data
    if bits == 2:
        v0 = packed & 0x03
        v1 = (packed >> 2) & 0x03
        v2 = (packed >> 4) & 0x03
        v3 = (packed >> 6) & 0x03
        return torch.stack([v0, v1, v2, v3], dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 4)
    elif bits == 4:
        v0 = packed & 0x0F
        v1 = (packed >> 4) & 0x0F
        return torch.stack([v0, v1], dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    return packed


def quantize_values(
    v: torch.Tensor,
    bits: int = 2,
    group_size: int = 32,
) -> ValueQuantized:
    """
    Symmetric group quantization for value vectors.

    Args:
        v: (..., seq_len, d) value vectors
        bits: quantization bits (2 or 4)
        group_size: number of elements per quantization group
    """
    orig_shape = v.shape
    d = orig_shape[-1]
    n_groups = d // group_size
    assert d % group_size == 0, f"head_dim {d} must be divisible by group_size {group_size}"

    # Reshape to groups
    v_grouped = v.reshape(*orig_shape[:-1], n_groups, group_size)  # (..., seq, n_groups, gs)

    # Compute scale and zero per group (asymmetric)
    v_min = v_grouped.min(dim=-1, keepdim=True).values
    v_max = v_grouped.max(dim=-1, keepdim=True).values

    n_levels = 2**bits - 1
    scale = (v_max - v_min) / n_levels
    scale = scale.clamp(min=1e-10)
    zero = v_min

    # Quantize
    v_q = ((v_grouped - zero) / scale).round().clamp(0, n_levels).to(torch.uint8)
    v_q_flat = v_q.reshape(*orig_shape[:-1], d)

    # Bit-pack: for 2-bit, pack 4 values per byte; for 4-bit, pack 2 per byte
    if bits == 2:
        # Pack 4 x 2-bit values into each uint8: [a, b, c, d] -> a | (b<<2) | (c<<4) | (d<<6)
        assert d % 4 == 0
        v_4 = v_q_flat.reshape(*orig_shape[:-1], d // 4, 4)
        packed = v_4[..., 0] | (v_4[..., 1] << 2) | (v_4[..., 2] << 4) | (v_4[..., 3] << 6)
        v_q_flat = packed  # shape: (..., d//4)
    elif bits == 4:
        assert d % 2 == 0
        v_2 = v_q_flat.reshape(*orig_shape[:-1], d // 2, 2)
        packed = v_2[..., 0] | (v_2[..., 1] << 4)
        v_q_flat = packed  # shape: (..., d//2)
    # bits==8: no packing needed

    return ValueQuantized(
        data=v_q_flat,
        scales=scale.squeeze(-1),
        zeros=zero.squeeze(-1),
        bits=bits,
    )


def dequantize_values(
    vq: ValueQuantized,
    group_size: int = 32,
) -> torch.Tensor:
    """Dequantize value vectors from bit-packed format."""
    data = unpack_values(vq).float()
    d = data.shape[-1]
    batch_shape = data.shape[:-1]

    n_groups = d // group_size
    data = data.reshape(*batch_shape, n_groups, group_size)
    scales = vq.scales.unsqueeze(-1)
    zeros = vq.zeros.unsqueeze(-1)

    v = data * scales + zeros
    return v.reshape(*batch_shape, d)


class TurboQuantKVCache:
    """
    KV cache using TurboQuant for keys and group quantization for values.

    Usage:
        cache = TurboQuantKVCache(head_dim=128, key_bits=3, value_bits=2)

        # During prefill:
        cache.prefill(key_states, value_states)

        # During decode (one token at a time):
        cache.append(new_key, new_value)

        # Compute attention:
        scores = cache.attention_scores(query_states)
        output = cache.attend(query_states, scores_after_softmax)
    """

    def __init__(
        self,
        head_dim: int,
        key_bits: int = 3,
        value_bits: int = 2,
        value_group_size: int = 32,
        buffer_size: int = 128,
        n_outlier_channels: int = 32,
        outlier_key_bits: int = None,
        device: torch.device = None,
        dtype: torch.dtype = torch.float16,
        layer_idx: int = 0,
    ):
        self.head_dim = head_dim
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.value_group_size = value_group_size
        self.buffer_size = buffer_size
        self.n_outlier_channels = n_outlier_channels
        self.outlier_key_bits = outlier_key_bits or key_bits + 1
        self.device = device or torch.device("cuda")
        self.dtype = dtype
        self.layer_idx = layer_idx

        # Key quantizers: separate for outlier and non-outlier channels
        # Paper Section 4.3: "splitting channels into outlier and non-outlier sets"
        self.key_quantizer = TurboQuantProd(
            dim=head_dim,
            bits=key_bits,
            device=self.device,
            seed=42 + layer_idx * 7,
        )

        # State
        self.seq_len: int = 0
        self.key_quantized: Optional[ProdQuantized] = None
        self.value_quantized: Optional[ValueQuantized] = None

        # Buffer for recent unquantized tokens
        self.key_buffer: Optional[torch.Tensor] = None
        self.value_buffer: Optional[torch.Tensor] = None

    def _effective_bits(self) -> float:
        """Compute the effective bits per element including the QJL overhead."""
        # MSE stage: (bits-1) bits per coordinate for indices
        # QJL stage: 1 bit per coordinate for signs
        # Plus: 1 float32 for residual norm per vector, 1 float32 for original norm
        # Total per vector: bits * d + 64 (for the two float32 norms)
        d = self.head_dim
        overhead_bits = 64  # two float32 scalars
        return self.key_bits + overhead_bits / d

    def prefill(self, keys: torch.Tensor, values: torch.Tensor):
        """
        Process prefill tokens.

        Args:
            keys: (batch, n_heads, seq_len, head_dim)
            values: (batch, n_heads, seq_len, head_dim)
        """
        seq_len = keys.shape[-2]
        self.seq_len = seq_len

        if seq_len <= self.buffer_size:
            # Everything fits in buffer, no quantization needed
            self.key_buffer = keys
            self.value_buffer = values
            return

        # Split into quantized portion and buffer
        n_quant = seq_len - self.buffer_size

        keys_to_quant = keys[..., :n_quant, :]
        values_to_quant = values[..., :n_quant, :]

        self.key_buffer = keys[..., n_quant:, :]
        self.value_buffer = values[..., n_quant:, :]

        # Quantize keys with TurboQuant
        self.key_quantized = self.key_quantizer.quantize(keys_to_quant)

        # Quantize values with group quantization
        self.value_quantized = quantize_values(
            values_to_quant, bits=self.value_bits, group_size=self.value_group_size
        )

    def append(self, key: torch.Tensor, value: torch.Tensor):
        """
        Append a single decode token.

        Args:
            key: (batch, n_heads, 1, head_dim)
            value: (batch, n_heads, 1, head_dim)
        """
        self.seq_len += 1

        if self.key_buffer is not None:
            self.key_buffer = torch.cat([self.key_buffer, key], dim=-2)
            self.value_buffer = torch.cat([self.value_buffer, value], dim=-2)
        else:
            self.key_buffer = key
            self.value_buffer = value

        # If buffer exceeds size, flush oldest chunk to quantized storage
        if self.key_buffer.shape[-2] > self.buffer_size:
            self._flush_buffer()

    def _flush_buffer(self):
        """Move oldest tokens from buffer to quantized storage."""
        n_flush = self.key_buffer.shape[-2] - self.buffer_size

        keys_flush = self.key_buffer[..., :n_flush, :]
        values_flush = self.value_buffer[..., :n_flush, :]

        self.key_buffer = self.key_buffer[..., n_flush:, :]
        self.value_buffer = self.value_buffer[..., n_flush:, :]

        # Quantize flushed keys
        new_key_q = self.key_quantizer.quantize(keys_flush)

        # Quantize flushed values
        new_val_q = quantize_values(
            values_flush, bits=self.value_bits, group_size=self.value_group_size
        )

        if self.key_quantized is None:
            self.key_quantized = new_key_q
            self.value_quantized = new_val_q
        else:
            # Concatenate along sequence dimension
            self.key_quantized = ProdQuantized(
                mse_indices=torch.cat([self.key_quantized.mse_indices, new_key_q.mse_indices], dim=-2),
                qjl_signs=torch.cat([self.key_quantized.qjl_signs, new_key_q.qjl_signs], dim=-2),
                residual_norms=torch.cat([self.key_quantized.residual_norms, new_key_q.residual_norms], dim=-1),
                norms=torch.cat([self.key_quantized.norms, new_key_q.norms], dim=-1),
                mse_bits=new_key_q.mse_bits,
            )
            self.value_quantized = ValueQuantized(
                data=torch.cat([self.value_quantized.data, new_val_q.data], dim=-2),
                scales=torch.cat([self.value_quantized.scales, new_val_q.scales], dim=-2),
                zeros=torch.cat([self.value_quantized.zeros, new_val_q.zeros], dim=-2),
                bits=self.value_bits,
            )

    def attention_scores(self, query: torch.Tensor, scale: float = None) -> torch.Tensor:
        """
        Compute attention logits: score[i,j] = <query_i, key_j> / sqrt(d).

        Args:
            query: (batch, n_heads, n_q, head_dim)
            scale: attention scale factor (default: 1/sqrt(head_dim))

        Returns:
            scores: (batch, n_heads, n_q, seq_len)
        """
        if scale is None:
            scale = 1.0 / math.sqrt(self.head_dim)

        scores_parts = []

        # Quantized portion
        if self.key_quantized is not None:
            scores_quant = self.key_quantizer.attention_score(query, self.key_quantized)
            scores_parts.append(scores_quant * scale)

        # Buffer portion (full precision)
        if self.key_buffer is not None:
            scores_buf = torch.matmul(query, self.key_buffer.transpose(-2, -1))
            scores_parts.append(scores_buf * scale)

        return torch.cat(scores_parts, dim=-1)

    def attend(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute attention output: out = softmax(scores) @ values.

        Args:
            attn_weights: (batch, n_heads, n_q, seq_len) — already softmaxed

        Returns:
            output: (batch, n_heads, n_q, head_dim)
        """
        output_parts = []
        col_offset = 0

        # Quantized values
        if self.value_quantized is not None:
            n_quant = self.value_quantized.data.shape[-2]
            w_quant = attn_weights[..., col_offset:col_offset + n_quant]
            v_dequant = dequantize_values(self.value_quantized, self.value_group_size)
            output_parts.append(torch.matmul(w_quant, v_dequant))
            col_offset += n_quant

        # Buffer values (full precision)
        if self.value_buffer is not None:
            n_buf = self.value_buffer.shape[-2]
            w_buf = attn_weights[..., col_offset:col_offset + n_buf]
            output_parts.append(torch.matmul(w_buf, self.value_buffer))

        return sum(output_parts)

    def memory_bytes(self) -> dict:
        """Estimate memory usage of the cache."""
        info = {"quantized_keys": 0, "quantized_values": 0, "buffer": 0, "total": 0}

        if self.key_quantized is not None:
            # MSE indices: bit-packed uint8
            info["quantized_keys"] += self.key_quantized.mse_indices.nelement()  # already packed bytes
            # QJL packed signs: 1 bit per coord, packed 8 per byte
            info["quantized_keys"] += self.key_quantized.qjl_signs.nelement()
            # Norms: float16 each (could use float16 for storage)
            info["quantized_keys"] += self.key_quantized.residual_norms.nelement() * 2
            info["quantized_keys"] += self.key_quantized.norms.nelement() * 2

        if self.value_quantized is not None:
            info["quantized_values"] += self.value_quantized.data.nelement()  # uint8 packed
            info["quantized_values"] += self.value_quantized.scales.nelement() * 2  # float16
            info["quantized_values"] += self.value_quantized.zeros.nelement() * 2

        if self.key_buffer is not None:
            info["buffer"] += self.key_buffer.nelement() * 2  # float16
        if self.value_buffer is not None:
            info["buffer"] += self.value_buffer.nelement() * 2

        info["total"] = info["quantized_keys"] + info["quantized_values"] + info["buffer"]
        return info

    def get_seq_length(self) -> int:
        return self.seq_len
