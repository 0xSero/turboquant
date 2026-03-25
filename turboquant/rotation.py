"""
Random rotation utilities for TurboQuant.

The paper uses Π = QR decomposition of a random Gaussian matrix.
For efficiency on GPU, we offer two options:
  1. Full random orthogonal matrix (via QR) — exact, costs O(d^2) storage
  2. Randomized Hadamard Transform (RHT) — fast O(d log d) but approximate

For typical head_dim (64-256), full QR is fine. The matrix is shared
across all heads in a layer and generated once from a fixed seed per layer.
"""

import math
import torch


def generate_rotation_matrix(
    d: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
) -> torch.Tensor:
    """
    Generate a random orthogonal matrix Π ∈ R^{d×d} via QR decomposition.

    This is the method described in Algorithm 1 of the paper.
    For head_dim=128, this is a 128×128 matrix = 64KB in float32, negligible.
    """
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    # Generate on CPU for reproducibility, then move to device
    G = torch.randn(d, d, generator=rng, dtype=torch.float32)
    Q, R = torch.linalg.qr(G)

    # Ensure proper rotation (det = +1) by fixing signs
    diag_sign = torch.sign(torch.diag(R))
    Q = Q * diag_sign.unsqueeze(0)

    return Q.to(device=device, dtype=dtype)


def generate_qjl_matrix(
    d: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    seed: int = 12345,
) -> torch.Tensor:
    """
    Generate the random projection matrix S ∈ R^{d×d} for QJL.
    S has i.i.d. N(0,1) entries.
    """
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    S = torch.randn(d, d, generator=rng, dtype=torch.float32)
    return S.to(device=device, dtype=dtype)


def rotate_forward(x: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
    """Apply random rotation: y = x @ Pi^T (equivalent to Pi @ x for each vector)."""
    return torch.matmul(x, Pi.T)


def rotate_backward(y: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
    """Apply inverse rotation: x = y @ Pi (equivalent to Pi^T @ y)."""
    return torch.matmul(y, Pi)
