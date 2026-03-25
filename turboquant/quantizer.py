"""
TurboQuant quantizers — Algorithm 1 (MSE) and Algorithm 2 (inner product).

These operate on tensors of shape (..., d) where d is the embedding dimension
(typically head_dim = 128 for modern LLMs).
"""

import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, NamedTuple

from turboquant.codebook import get_codebook_tensors
from turboquant.rotation import (
    generate_rotation_matrix,
    generate_qjl_matrix,
    rotate_forward,
    rotate_backward,
)


class MSEQuantized(NamedTuple):
    """Output of TurboQuant MSE quantization."""
    indices: torch.Tensor       # (..., packed_len) uint8 bit-packed indices
    norms: torch.Tensor         # (...,) original L2 norms
    bits: int                   # number of bits per index (for unpacking)


class ProdQuantized(NamedTuple):
    """Output of TurboQuant inner-product quantization."""
    mse_indices: torch.Tensor   # (..., packed_len) uint8 bit-packed MSE indices
    qjl_signs: torch.Tensor    # (..., packed_len) uint8 packed sign bits
    residual_norms: torch.Tensor  # (...,) L2 norms of residual vectors
    norms: torch.Tensor         # (...,) original L2 norms
    mse_bits: int               # bits per MSE index (for unpacking)


def _pack_indices(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Bit-pack integer indices (0..2^bits-1) into uint8 bytes.

    For bits=1: 8 values per byte
    For bits=2: 4 values per byte
    For bits=3: stored as 4-bit (2 per byte) for simplicity
    For bits=4: 2 values per byte
    """
    d = indices.shape[-1]
    batch_shape = indices.shape[:-1]

    if bits == 1:
        vals_per_byte = 8
    elif bits == 2:
        vals_per_byte = 4
    elif bits <= 4:
        vals_per_byte = 2
        bits = 4  # round up to 4-bit packing
    else:
        # Just store as uint8
        return indices.to(torch.uint8)

    # Pad to multiple of vals_per_byte
    padded_d = ((d + vals_per_byte - 1) // vals_per_byte) * vals_per_byte
    if padded_d > d:
        indices = F.pad(indices.to(torch.uint8), (0, padded_d - d), value=0)

    reshaped = indices.to(torch.uint8).reshape(*batch_shape, -1, vals_per_byte)
    shifts = torch.arange(vals_per_byte, device=indices.device, dtype=torch.uint8) * bits
    packed = (reshaped << shifts).sum(dim=-1, dtype=torch.uint8)
    return packed


def _unpack_indices(packed: torch.Tensor, bits: int, d: int) -> torch.Tensor:
    """Unpack bit-packed indices back to integer tensor."""
    batch_shape = packed.shape[:-1]

    if bits == 1:
        vals_per_byte = 8
    elif bits == 2:
        vals_per_byte = 4
    elif bits <= 4:
        vals_per_byte = 2
        bits = 4
    else:
        return packed.long()

    mask = (1 << bits) - 1
    shifts = torch.arange(vals_per_byte, device=packed.device, dtype=torch.uint8) * bits
    unpacked = ((packed.unsqueeze(-1) >> shifts) & mask)
    unpacked = unpacked.reshape(*batch_shape, -1)
    return unpacked[..., :d].long()


class TurboQuantMSE(torch.nn.Module):
    """
    TurboQuant optimized for MSE (Algorithm 1).

    Quantize: y = Π·(x/||x||), then find nearest centroid per coordinate.
    Dequantize: look up centroids, rotate back, rescale by ||x||.
    """

    def __init__(
        self,
        dim: int,
        bits: int = 3,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
        seed: int = 42,
    ):
        super().__init__()
        self.dim = dim
        self.bits = bits
        self.n_clusters = 2**bits
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Precompute rotation matrix
        self.register_buffer(
            "Pi", generate_rotation_matrix(dim, self.device, dtype, seed=seed)
        )

        # Precompute codebook
        centroids, boundaries = get_codebook_tensors(dim, bits, self.device, dtype)
        self.register_buffer("centroids", centroids)      # (2^b,)
        self.register_buffer("boundaries", boundaries)    # (2^b + 1,)

        # Precompute interior boundaries for fast searchsorted
        # boundaries[1:-1] are the decision boundaries between clusters
        self.register_buffer("decision_boundaries", boundaries[1:-1].contiguous())

    def quantize(self, x: torch.Tensor) -> MSEQuantized:
        """
        Quantize vectors x of shape (..., d).

        Returns MSEQuantized with bit-packed indices and norms.
        """
        # Store norms for rescaling
        norms = x.norm(dim=-1, keepdim=False)
        # Normalize to unit sphere
        x_unit = x / (norms.unsqueeze(-1) + 1e-10)

        # Apply random rotation
        y = rotate_forward(x_unit.float(), self.Pi)  # (..., d)

        # Quantize each coordinate: find bucket via searchsorted
        indices = torch.searchsorted(self.decision_boundaries, y.contiguous())

        # Bit-pack the indices
        packed = _pack_indices(indices, self.bits)

        return MSEQuantized(indices=packed, norms=norms, bits=self.bits)

    def dequantize(self, q: MSEQuantized) -> torch.Tensor:
        """Reconstruct vectors from quantized representation."""
        # Unpack indices
        indices = _unpack_indices(q.indices, q.bits, self.dim)

        # Look up centroids
        y_hat = self.centroids[indices]  # (..., d)

        # Rotate back
        x_hat = rotate_backward(y_hat, self.Pi)  # (..., d)

        # Rescale by original norms
        x_hat = x_hat * q.norms.unsqueeze(-1)

        return x_hat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize and immediately dequantize (for testing)."""
        return self.dequantize(self.quantize(x))


class TurboQuantProd(torch.nn.Module):
    """
    TurboQuant optimized for inner products (Algorithm 2).

    Two-stage:
      1. Apply TurboQuant_MSE at (b-1) bits → get residual r = x - x̃
      2. Apply QJL to residual: sign(S·r) → 1 bit per coordinate
      3. Store ||r||₂ for rescaling

    The dequantized inner product estimate is:
      <y, x̃_mse> + ||r|| * sqrt(π/2)/d * <S^T · qjl_signs, y>
    which is unbiased: E[estimate] = <y, x>
    """

    def __init__(
        self,
        dim: int,
        bits: int = 3,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
        seed: int = 42,
    ):
        super().__init__()
        self.dim = dim
        self.bits = bits
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        assert bits >= 2, "Inner product TurboQuant requires at least 2 bits (1 for MSE + 1 for QJL)"

        # Stage 1: MSE quantizer at (b-1) bits
        self.mse_quantizer = TurboQuantMSE(
            dim=dim, bits=bits - 1, device=self.device, dtype=dtype, seed=seed
        )

        # Stage 2: QJL projection matrix S ∈ R^{d×d}
        self.register_buffer(
            "S", generate_qjl_matrix(dim, self.device, dtype, seed=seed + 1000)
        )

        # QJL dequantization constant
        self.qjl_scale = math.sqrt(math.pi / 2.0) / dim

    def _pack_qjl_signs(self, projected: torch.Tensor) -> torch.Tensor:
        """Pack sign bits into uint8 (8 signs per byte)."""
        signs = (projected > 0).to(torch.uint8)
        d = signs.shape[-1]
        if d % 8 != 0:
            signs = F.pad(signs, (0, 8 - d % 8), value=0)
        signs_reshaped = signs.reshape(*signs.shape[:-1], -1, 8)
        powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=signs.device, dtype=torch.uint8)
        return (signs_reshaped * powers).sum(dim=-1, dtype=torch.uint8)

    def _unpack_qjl_signs(self, packed: torch.Tensor) -> torch.Tensor:
        """Unpack sign bits from uint8 to float {-1, +1}."""
        powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=packed.device, dtype=torch.uint8)
        unpacked = ((packed.unsqueeze(-1) & powers) > 0).float()
        signs = unpacked.reshape(*packed.shape[:-1], -1)[..., :self.dim]
        return 2.0 * signs - 1.0

    def quantize(self, x: torch.Tensor) -> ProdQuantized:
        """
        Quantize vectors x of shape (..., d).

        Returns ProdQuantized with bit-packed MSE indices, QJL signs, and norms.
        """
        # Stage 1: MSE quantize at (b-1) bits
        mse_q = self.mse_quantizer.quantize(x)

        # Reconstruct MSE approximation
        x_hat = self.mse_quantizer.dequantize(mse_q)

        # Compute residual
        residual = x - x_hat
        residual_norms = residual.norm(dim=-1)

        # Stage 2: QJL on residual
        projected = torch.matmul(residual.float(), self.S.T)
        packed_signs = self._pack_qjl_signs(projected)

        return ProdQuantized(
            mse_indices=mse_q.indices,
            qjl_signs=packed_signs,
            residual_norms=residual_norms,
            norms=mse_q.norms,
            mse_bits=mse_q.bits,
        )

    def dequantize(self, q: ProdQuantized) -> torch.Tensor:
        """Reconstruct vectors from quantized representation."""
        # Stage 1: MSE dequantize
        mse_q = MSEQuantized(indices=q.mse_indices, norms=q.norms, bits=q.mse_bits)
        x_mse = self.mse_quantizer.dequantize(mse_q)

        # Stage 2: QJL dequantize
        signs = self._unpack_qjl_signs(q.qjl_signs)

        # x̃_qjl = sqrt(π/2)/d * ||r|| * S^T @ signs
        x_qjl = torch.matmul(signs, self.S)
        x_qjl = x_qjl * (self.qjl_scale * q.residual_norms.unsqueeze(-1))

        return x_mse + x_qjl

    def attention_score(
        self,
        query: torch.Tensor,
        quantized_key: ProdQuantized,
    ) -> torch.Tensor:
        """
        Compute attention scores <query, key> using quantized keys.

        Args:
            query: (..., n_q, d)  — the query vectors
            quantized_key: ProdQuantized with shapes (..., n_k, ...) etc.

        Returns:
            scores: (..., n_q, n_k) — the attention logits
        """
        # Stage 1: MSE contribution
        mse_q = MSEQuantized(indices=quantized_key.mse_indices, norms=quantized_key.norms,
                             bits=quantized_key.mse_bits)
        k_mse = self.mse_quantizer.dequantize(mse_q)
        scores_mse = torch.matmul(query.float(), k_mse.float().transpose(-2, -1))

        # Stage 2: QJL contribution — asymmetric estimator
        q_sketched = torch.matmul(query.float(), self.S.T)
        signs = self._unpack_qjl_signs(quantized_key.qjl_signs)

        scores_qjl = torch.matmul(q_sketched, signs.transpose(-2, -1))
        scores_qjl = scores_qjl * (self.qjl_scale * quantized_key.residual_norms.unsqueeze(-2))

        return scores_mse + scores_qjl.to(scores_mse.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize and immediately dequantize (for testing)."""
        return self.dequantize(self.quantize(x))
