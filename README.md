# TurboQuant: KV Cache Compression for vLLM

Implementation of the TurboQuant KV cache compression method (ICLR 2026) integrated with vLLM.

## Results

Tested on **Qwen3.5-27B** with 4x RTX 3090 (24GB each), TP=4, bf16, `gpu_memory_utilization=0.90`.

| Metric | Baseline vLLM | TurboQuant |
|--------|--------------|------------|
| KV cache blocks | 583 | 583 (freed after prefill) |
| Max token capacity | 457,072 | **914,144** |
| VRAM/GPU after gen | 21,535 MB | 21,553 MB |
| VRAM/GPU after free | - | **21,297 MB** |
| KV tensor freed/GPU | - | **7,489 MB** |
| Total KV freed | - | **30.0 GB** |
| **Context improvement** | 1.0x | **2.0x** |

Both produce identical output for the same prompt.

## How It Works

TurboQuant compresses KV cache entries using:
1. **Random orthogonal rotation** to spread information across dimensions
2. **Lloyd-Max optimal scalar quantization** (3-bit keys) on Beta-distributed rotated values
3. **QJL (Quantized Johnson-Lindenstrauss) projection** for residual sign bits
4. **Group quantization** (2-bit values) with per-group scales and zeros
5. **Bit-packing**: 4 values per byte for 2-bit quantization

After prefill, the compressed TQ store holds all token data. The original paged KV cache tensors are freed, and decode uses TQ's fused Triton kernels to reconstruct attention on-the-fly.

**Compression ratio**: ~198 bytes/token (TQ) vs ~512 bytes/token (bf16) = **2.6x compression** per full-attention layer.

Qwen3.5-27B has 16 full-attention layers (out of 64 total, rest are linear-attention/Mamba). TQ hooks onto all 16, freeing 7.5 GB/GPU of KV cache.

## Architecture

```
turboquant/
  codebook.py          # Lloyd-Max optimal scalar quantizer for Beta distribution
  codebooks/           # Pre-generated codebook files (d=256, bits 2/3/4)
  rotation.py          # Random orthogonal rotation + QJL projection
  quantizer.py         # TurboQuantMSE + TurboQuantProd (rotation + quantize pipeline)
  kv_cache.py          # KV cache manager with value bit-packing
  triton_kernels.py    # 3 fused Triton kernels for decode attention
  vllm_attn_backend.py # vLLM monkey-patching (hooks, free_kv_cache, enable_no_alloc)
  __init__.py
proof.py               # Definitive A/B benchmark (baseline vs TQ, separate processes)
setup.py               # pip install -e .
```

## Usage

```bash
pip install -e .

# Run the proof benchmark
CUDA_VISIBLE_DEVICES=0,1,4,6 python proof.py
```

## What Was Tried

This project went through many iterations to arrive at real, measurable VRAM savings:

### Iteration 1: Shadow mode (accumulate-only)
Ran TQ alongside vLLM's normal attention. Proved quantization accuracy but no VRAM benefit -- both caches existed simultaneously.

### Iteration 2: gpu_memory_utilization trick
Lowered `gpu_memory_utilization` for the TQ run to show less VRAM. This was **fake** -- the savings came from telling vLLM to use less memory, not from actual compression.

### Iteration 3: Active mode (skip flash attention on decode)
TQ computes decode attention entirely from its compressed store, skipping flash attention. Correct approach, but didn't free the paged cache.

### Iteration 4: Analytical proof
Computed tensor sizes directly: TQ uses 198 bytes/token vs 512 for bf16 = 2.6x compression. Mathematically correct but not a runtime measurement.

### Iteration 5: free_kv_cache (current approach)
After prefill, replace the paged KV cache tensors for TQ-hooked layers with 1-byte dummy tensors. This actually frees GPU memory. `torch.cuda.empty_cache()` returns a portion to the OS; the rest stays available in CUDA's memory pool for new allocations.

**Result**: 30 GB freed across 4 GPUs, 2x effective context capacity.

### Iteration 6: Zero-allocation (attempted, not yet working)
Tried to exclude TQ layers from `get_kv_cache_spec()` so vLLM never allocates KV cache for them. Failed because vLLM's compiled graph hardcodes `unified_kv_cache_update` calls per layer, and `get_attention_context` throws KeyError for layers missing from `attn_metadata`. Would require patching `torch.ops.vllm.unified_attention` and `unified_kv_cache_update` at the custom op level, or forking vLLM.

### Qwen3.5-262B attempts
Multiple OOM crashes on 8x RTX 3090 (24GB each). The model is multimodal (vision encoder uses extra VRAM), weights are 136GB (W4A16 quantized), and margins were too tight. Leaked GPU memory from killed processes accumulated across attempts. Switched to 27B for the proof.

## Limitations

- **Prefill still uses paged cache**: The KV cache is allocated at engine init and used during prefill. TQ frees it after. True zero-allocation requires deeper vLLM integration.
- **Only full-attention layers**: TQ hooks onto standard multi-head/GQA attention layers. Linear-attention/Mamba layers are not compressed.
- **Lossy compression**: TQ is lossy. 3-bit keys + 2-bit values lose information. Quality impact depends on the task and model.
- **Single sequence tested**: The proof runs 1 sequence. Production benefits scale with concurrent requests.

## Community Ports

| Project | Hardware | Backend | Link |
|---------|----------|---------|------|
| **turboquant-mac** | Apple Silicon (M1–M4) | MLX Metal kernels + PyTorch CPU | [yzamari/turboQuantPlayground](https://github.com/yzamari/turboQuantPlayground) |
| **turboquant-pytorch** | Any CPU/GPU | PyTorch | [tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch) |
| **turbo-quant** | Any CPU | Rust | [RecursiveIntell/turbo-quant](https://github.com/RecursiveIntell/turbo-quant) |

## Environment

- vLLM 0.17.0
- 4x NVIDIA RTX 3090 (24GB each)
- Qwen3.5-27B (bf16, TP=4)
- CUDA 12.x, PyTorch 2.x
