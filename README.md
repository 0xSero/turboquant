# TurboQuant

KV cache quantization for vLLM. Compresses keys to 3 bits and values to 2 bits using the method from [TurboQuant (Zandieh et al., ICLR 2026)](https://arxiv.org/abs/2504.19874).

## How it works

For each key vector **k** in R^d:
1. Store ‖k‖. Normalize to unit sphere.
2. Apply random orthogonal rotation (makes coordinates ~Beta distributed).
3. Quantize each coordinate with a Lloyd-Max optimal codebook at (b-1) bits.
4. Compute residual, project with a random Gaussian matrix, store signs (1 bit per coord).

Attention scores are computed directly from the compressed representation without dequantization:
```
score = <q_rot, centroids[indices]> · ‖k‖  +  qjl_correction
```

Values use standard asymmetric group quantization at 2 bits, packed 4 per byte.

Three fused Triton kernels do MSE score + QJL correction + online softmax + value aggregation in a single pass.

## Memory savings

Per-token storage (1 KV head, measured from actual tensor sizes):

| head_dim | BF16 | FP8 | TurboQuant | vs FP8 |
|----------|------|-----|------------|--------|
| 128 | 512 B | 256 B | 102 B | 2.5x smaller |
| 256 | 1024 B | 512 B | 198 B | 2.6x smaller |

At scale (per GPU, including fixed overhead from rotation matrices):

| Model config | Context | FP8 KV | TQ KV | Saved |
|---|---|---|---|---|
| 94 layers, d=128 | 32K | 789 MB | 327 MB | 462 MB |
| 94 layers, d=128 | 131K | 3,154 MB | 1,269 MB | 1,885 MB |
| 80 layers, d=128 | 131K | 2,684 MB | 1,080 MB | 1,604 MB |
| 16 layers, d=256 | 131K | 1,074 MB | 424 MB | 650 MB |

Run `python -m turboquant proof` to see the full table.

## Install

```bash
pip install -e .
```

Requires PyTorch >= 2.1, scipy, and optionally triton >= 3.0 for fused kernels.

## Usage

### Pre-compute codebooks

```bash
python -m turboquant codebook --dim 128
python -m turboquant codebook --dim 256
```

### vLLM integration

```python
from vllm import LLM, SamplingParams
from turboquant.vllm_attn_backend import install_turboquant_hooks, MODE_ACTIVE

llm = LLM(model="meta-llama/Llama-3-8B", tensor_parallel_size=1)

# Install hooks on each worker via collective_rpc
def _install(worker):
    return install_turboquant_hooks(
        worker.model_runner,
        key_bits=3, value_bits=2, mode=MODE_ACTIVE,
    )

executor = llm.llm_engine.engine_core.engine_core.model_executor
executor.collective_rpc(_install)

output = llm.generate(["Hello"], SamplingParams(max_tokens=100))
```

### Standalone (no vLLM)

```python
import torch
from turboquant import TurboQuantKVCache

cache = TurboQuantKVCache(head_dim=128, key_bits=3, value_bits=2)

keys = torch.randn(1, 8, 1024, 128, device="cuda")
values = torch.randn(1, 8, 1024, 128, device="cuda")
cache.prefill(keys, values)

query = torch.randn(1, 8, 1, 128, device="cuda")
scores = cache.attention_scores(query)
```

### A/B Benchmark

```bash
python benchmark.py --model meta-llama/Llama-3-8B --tp 1
```

## Tests

```bash
pip install pytest
pytest test_turboquant.py -v        # Core algorithm tests (needs CUDA)
pytest test_triton_kernels.py -v    # Triton kernel tests (needs CUDA + triton)
```

## Architecture

```
turboquant/
├── codebook.py           # Lloyd-Max optimal scalar quantizer for Beta(d/2, d/2)
├── rotation.py           # Random orthogonal rotation (QR) + Gaussian projection
├── quantizer.py          # TurboQuantMSE (Alg 1) + TurboQuantProd (Alg 2) + bit-packing
├── kv_cache.py           # KV cache manager with quantized key/value stores
├── triton_kernels.py     # 3 fused Triton kernels for compressed attention
├── vllm_attn_backend.py  # vLLM FlashAttention monkey-patching (v0.16-0.17)
├── patch.py              # HuggingFace model patching (standalone, no vLLM)
├── proof.py              # Memory savings calculator
└── codebooks/            # Pre-computed Lloyd-Max codebooks (JSON)
```

## References

- [TurboQuant](https://arxiv.org/abs/2504.19874) -- Zandieh et al., ICLR 2026
- [QJL](https://arxiv.org/abs/2406.03482) -- Zandieh et al., AAAI 2025
