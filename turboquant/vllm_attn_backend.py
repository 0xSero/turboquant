"""
TurboQuant attention backend integration for vLLM v0.17.x.

This module keeps the vLLM monkey-patching logic readable by splitting the
implementation into small helpers around three concerns:
  1. cache update interception
  2. active/shadow decode execution for FlashAttention / GQA
  3. explicit gating for MLA backends until a TurboQuant MLA path exists

Modes:
  - accumulate: collect TurboQuant side-cache only, return vLLM output
  - shadow: compute TurboQuant decode for validation, return vLLM output
  - active: return TurboQuant decode output when supported, otherwise fallback
"""

from __future__ import annotations

import logging
import types
from dataclasses import dataclass
from typing import Optional

import torch

from turboquant.kv_cache import TurboQuantKVCache, ValueQuantized
from turboquant.quantizer import ProdQuantized, TurboQuantProd

logger = logging.getLogger("turboquant.attn")

MODE_SHADOW = "shadow"
MODE_ACCUMULATE = "accumulate"
MODE_ACTIVE = "active"
_VALID_MODES = (MODE_SHADOW, MODE_ACCUMULATE, MODE_ACTIVE)

_GLOBAL_MODE = MODE_ACCUMULATE


def set_mode(mode: str):
    global _GLOBAL_MODE
    assert mode in _VALID_MODES
    _GLOBAL_MODE = mode
    logger.info(f"[TurboQuant] Mode set to: {mode}")


def reset_decode_flag():
    pass


def get_mode() -> str:
    return _GLOBAL_MODE


@dataclass
class DecodeBatch:
    query_heads: torch.Tensor
    prod_q_flat: ProdQuantized
    value_q_flat: ValueQuantized
    num_tokens: int
    num_query_heads: int
    head_dim: int
    quantized_tokens: int


class TurboQuantLayerState:
    """Per-layer TurboQuant state for standard vLLM attention backends."""

    def __init__(
        self,
        head_dim,
        num_kv_heads,
        num_query_heads=None,
        key_bits=3,
        value_bits=2,
        value_group_size=32,
        buffer_size=128,
        device=None,
        layer_idx=0,
        backend_kind="flash",
    ):
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.num_query_heads = num_query_heads or num_kv_heads
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.value_group_size = value_group_size
        self.buffer_size = buffer_size
        self.device = device or torch.device("cuda")
        self.layer_idx = layer_idx
        self.backend_kind = backend_kind

        self.quantizer = TurboQuantProd(
            dim=head_dim,
            bits=key_bits,
            device=self.device,
            seed=42 + layer_idx * 7,
        )

        self.seq_caches: dict[str, TurboQuantKVCache] = {}
        self._log_count = 0
        self._warned_mla_active = False

        self._flat_cache: Optional[tuple[ProdQuantized, ValueQuantized, int]] = None
        self._flat_cache_seq_len: int = -1

        self._ring_k = torch.zeros(
            buffer_size,
            num_kv_heads,
            head_dim,
            device=self.device,
            dtype=torch.bfloat16,
        )
        self._ring_v = torch.zeros(
            buffer_size,
            num_kv_heads,
            head_dim,
            device=self.device,
            dtype=torch.bfloat16,
        )
        self._ring_pos = 0

    def get_or_create_cache(self, seq_id: str = "default") -> TurboQuantKVCache:
        cache = self.seq_caches.get(seq_id)
        if cache is None:
            cache = TurboQuantKVCache(
                head_dim=self.head_dim,
                key_bits=self.key_bits,
                value_bits=self.value_bits,
                value_group_size=self.value_group_size,
                buffer_size=0,
                device=self.device,
                layer_idx=self.layer_idx,
            )
            self.seq_caches[seq_id] = cache
        return cache

    def ring_write(self, key: torch.Tensor, value: torch.Tensor, num_tokens: int):
        space = self.buffer_size - self._ring_pos
        if num_tokens <= space:
            self._ring_k[self._ring_pos:self._ring_pos + num_tokens] = key[:num_tokens]
            self._ring_v[self._ring_pos:self._ring_pos + num_tokens] = value[:num_tokens]
            self._ring_pos += num_tokens
            return

        if space > 0:
            self._ring_k[self._ring_pos:] = key[:space]
            self._ring_v[self._ring_pos:] = value[:space]
            self._ring_pos = self.buffer_size

        self.flush_default_sequence()

        remaining = num_tokens - space
        while remaining > self.buffer_size:
            offset = num_tokens - remaining
            self._ring_k[:] = key[offset:offset + self.buffer_size]
            self._ring_v[:] = value[offset:offset + self.buffer_size]
            self._ring_pos = self.buffer_size
            self.flush_default_sequence()
            remaining -= self.buffer_size

        if remaining > 0:
            offset = num_tokens - remaining
            self._ring_k[:remaining] = key[offset:offset + remaining]
            self._ring_v[:remaining] = value[offset:offset + remaining]
            self._ring_pos = remaining

    def flush_default_sequence(self):
        if self._ring_pos == 0:
            return

        all_k = self._ring_k[:self._ring_pos].transpose(0, 1).unsqueeze(0)
        all_v = self._ring_v[:self._ring_pos].transpose(0, 1).unsqueeze(0)

        cache = self.get_or_create_cache("default")
        if cache.seq_len == 0:
            cache.prefill(all_k, all_v)
        else:
            cache.append(all_k, all_v)

        self._ring_pos = 0
        self.invalidate_flat_cache()

    def invalidate_flat_cache(self):
        self._flat_cache = None
        self._flat_cache_seq_len = -1

    def get_flat_cache(self, cache: TurboQuantKVCache):
        if cache.key_quantized is None or cache.value_quantized is None:
            return None
        current_len = cache.key_quantized.norms.shape[-1]
        if self._flat_cache is not None and self._flat_cache_seq_len == current_len:
            return self._flat_cache
        self._flat_cache = _flatten_quantized_cache(cache.key_quantized, cache.value_quantized)
        self._flat_cache_seq_len = current_len
        return self._flat_cache

    def reset(self):
        self.seq_caches.clear()
        self._ring_pos = 0
        self._log_count = 0
        self._warned_mla_active = False
        self.invalidate_flat_cache()

    @property
    def supports_active_decode(self) -> bool:
        return self.backend_kind == "flash"


class PatchedFlashKVCacheUpdate:
    """Intercepts do_kv_cache_update.

    In NO_ALLOC mode: skip writing to paged cache entirely (it doesn't exist).
    K/V capture happens in PatchedFlashForward instead.
    In other modes: write to paged cache normally, then capture for TQ.
    """
    __slots__ = ("orig_fn", "state", "no_alloc")

    def __init__(self, orig_fn, state: TurboQuantLayerState, no_alloc: bool = False):
        self.orig_fn = orig_fn
        self.state = state
        self.no_alloc = no_alloc

    def __call__(self, self_impl, layer, key, value, kv_cache, slot_mapping):
        if self.no_alloc:
            return

        self.orig_fn(self_impl, layer, key, value, kv_cache, slot_mapping)

        num_tokens = slot_mapping.shape[0]
        if num_tokens <= 1:
            return

        self.state.ring_write(key, value, num_tokens)
        self.state.flush_default_sequence()


class PatchedMLAKVCacheUpdate:
    __slots__ = ("orig_fn", "state")

    def __init__(self, orig_fn, state: TurboQuantLayerState):
        self.orig_fn = orig_fn
        self.state = state

    def __call__(self, self_impl, kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale):
        self.orig_fn(self_impl, kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale)
        if not getattr(self.state, "_mla_logged", False):
            logger.info(
                f"[TurboQuant] MLA cache update observed on layer {self.state.layer_idx}; "
                "TurboQuant MLA path is gated pending MLA-specific quantization support."
            )
            self.state._mla_logged = True


class PatchedFlashForward:
    __slots__ = ("orig_fn", "state", "no_alloc")

    def __init__(self, orig_fn, state: TurboQuantLayerState, no_alloc: bool = False):
        self.orig_fn = orig_fn
        self.state = state
        self.no_alloc = no_alloc

    def __call__(
        self,
        self_impl,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=None,
        output_scale=None,
        output_block_scale=None,
    ):
        # Fast path: when paged cache exists and not in accumulate-only mode,
        # the only extra work is capturing K/V during prefill. For decode
        # (the hot path), we fall through to flash attention immediately.
        if not self.no_alloc:
            mode = _GLOBAL_MODE
            if mode != MODE_ACCUMULATE and attn_metadata is not None:
                is_prefill = attn_metadata.max_query_len > 1
                if is_prefill and mode == MODE_ACTIVE:
                    # Prefill: capture K/V into TQ, but still use flash attention
                    pass  # capture already happens in do_kv_cache_update
            return self.orig_fn(
                self_impl, layer, query, key, value, kv_cache,
                attn_metadata, output, output_scale, output_block_scale,
            )

        # no_alloc path below -- paged cache was freed
        mode = _GLOBAL_MODE

        if mode == MODE_ACCUMULATE:
            return self.orig_fn(
                self_impl, layer, query, key, value, kv_cache,
                attn_metadata, output, output_scale, output_block_scale,
            )

        if attn_metadata is None:
            return self.orig_fn(
                self_impl, layer, query, key, value, kv_cache,
                attn_metadata, output, output_scale, output_block_scale,
            )

        is_prefill = attn_metadata.max_query_len > 1

        # --- PREFILL (no_alloc only) ---
        if is_prefill:
            result = _tq_prefill_attention(
                self.state, self_impl, query, key, value, attn_metadata
            )
            if output is not None:
                output[:result.shape[0]].copy_(result.view_as(output[:result.shape[0]]))
                return output
            return result

        # --- DECODE (no_alloc only): paged cache was freed, must use TQ decode ---
        cache = self.state.seq_caches.get("default")
        flat = self.state.get_flat_cache(cache) if cache is not None else None
        has_tq_data = flat is not None and flat[2] >= 16

        if mode == MODE_ACTIVE and has_tq_data:
            decode_batch = _build_flash_decode_batch(self.state, query, attn_metadata, cache)
            if decode_batch is not None:
                tq_out = _run_flash_turboquant_decode(self.state, self_impl, decode_batch)
                if tq_out is not None:
                    num_actual = attn_metadata.num_actual_tokens
                    expected_shape = (num_actual, decode_batch.num_query_heads * decode_batch.head_dim)
                    result = tq_out.reshape(expected_shape).to(query.dtype)

                    if self.state._log_count < 3 and self.state.layer_idx == 0:
                        logger.info(
                            f"[TQ ACTIVE L{self.state.layer_idx}] "
                            f"out={tuple(result.shape)} q_heads={decode_batch.num_query_heads} "
                            f"kv_heads={self.state.num_kv_heads} TQ_tokens={decode_batch.quantized_tokens} "
                            f"(flash skipped)"
                        )
                        self.state._log_count += 1

                    if output is not None:
                        output.copy_(result.view_as(output))
                        return output
                    return result

        logger.warning(f"[TQ] no_alloc decode without TQ data on L{self.state.layer_idx}")
        num_actual = attn_metadata.num_actual_tokens
        return torch.zeros(num_actual, query.shape[-1], dtype=query.dtype, device=query.device)


class PatchedMLAForwardMQA:
    __slots__ = ("orig_fn", "state")

    def __init__(self, orig_fn, state: TurboQuantLayerState):
        self.orig_fn = orig_fn
        self.state = state

    def __call__(self, self_impl, q, kv_c_and_k_pe_cache, attn_metadata, layer):
        mode = _GLOBAL_MODE
        if mode == MODE_ACTIVE and not self.state._warned_mla_active:
            logger.warning(
                f"[TurboQuant] MLA active decode is not implemented yet for layer {self.state.layer_idx}; "
                "falling back to native vLLM MLA output."
            )
            self.state._warned_mla_active = True
        return self.orig_fn(self_impl, q, kv_c_and_k_pe_cache, attn_metadata, layer)


def _infer_num_query_heads(attn_module, impl) -> int:
    for candidate in (
        getattr(attn_module, "num_heads", None),
        getattr(attn_module, "num_attention_heads", None),
        getattr(impl, "num_heads", None),
    ):
        if candidate:
            return int(candidate)
    return int(impl.num_kv_heads)


def _is_mla_impl(impl) -> bool:
    return hasattr(impl, "forward_mqa") and hasattr(impl, "do_kv_cache_update") and not hasattr(impl, "forward")


def _flatten_quantized_cache(prod_q: ProdQuantized, value_q: ValueQuantized) -> tuple[ProdQuantized, ValueQuantized, int]:
    mse_indices = prod_q.mse_indices.reshape(-1, prod_q.mse_indices.shape[-2], prod_q.mse_indices.shape[-1]).contiguous()
    qjl_signs = prod_q.qjl_signs.reshape(-1, prod_q.qjl_signs.shape[-2], prod_q.qjl_signs.shape[-1]).contiguous()
    norms = prod_q.norms.reshape(-1, prod_q.norms.shape[-1]).contiguous()
    residual_norms = prod_q.residual_norms.reshape(-1, prod_q.residual_norms.shape[-1]).contiguous()

    v_data = value_q.data.reshape(-1, value_q.data.shape[-2], value_q.data.shape[-1]).contiguous()
    v_scales = value_q.scales.reshape(-1, value_q.scales.shape[-2], value_q.scales.shape[-1]).contiguous()
    v_zeros = value_q.zeros.reshape(-1, value_q.zeros.shape[-2], value_q.zeros.shape[-1]).contiguous()

    flat_prod_q = ProdQuantized(
        mse_indices=mse_indices,
        qjl_signs=qjl_signs,
        residual_norms=residual_norms,
        norms=norms,
        mse_bits=prod_q.mse_bits,
    )
    v_bits = value_q.bits if len(value_q) > 3 else 2
    flat_value_q = ValueQuantized(data=v_data, scales=v_scales, zeros=v_zeros, bits=v_bits)
    quantized_tokens = int(norms.shape[-1])
    return flat_prod_q, flat_value_q, quantized_tokens


def _tq_prefill_attention(
    state: TurboQuantLayerState,
    self_impl,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_metadata,
) -> torch.Tensor:
    """Compute prefill attention without paged KV cache, capture K/V into TQ."""
    import torch.nn.functional as F

    num_actual = attn_metadata.num_actual_tokens
    q = query[:num_actual]   # (N, num_q_heads, head_dim)
    k = key[:num_actual]     # (N, num_kv_heads, head_dim)
    v = value[:num_actual]   # (N, num_kv_heads, head_dim)

    # Capture K/V into TQ compressed store
    state.ring_write(k, v, num_actual)
    state.flush_default_sequence()

    head_dim = state.head_dim
    num_kv_heads = state.num_kv_heads
    num_q_heads = q.shape[1] if q.dim() == 3 else q.shape[-1] // head_dim

    if q.dim() == 2:
        q = q.view(num_actual, num_q_heads, head_dim)
    if k.dim() == 2:
        k = k.view(num_actual, num_kv_heads, head_dim)
        v = v.view(num_actual, num_kv_heads, head_dim)

    # GQA: repeat K/V heads to match Q heads
    if num_q_heads != num_kv_heads:
        repeats = num_q_heads // num_kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)

    # (1, heads, seq, dim) for SDPA
    q_t = q.unsqueeze(0).transpose(1, 2)
    k_t = k.unsqueeze(0).transpose(1, 2)
    v_t = v.unsqueeze(0).transpose(1, 2)

    scale = getattr(self_impl, 'scale', 1.0 / (head_dim ** 0.5))
    attn_out = F.scaled_dot_product_attention(
        q_t, k_t, v_t, is_causal=True, scale=scale
    )  # (1, heads, seq, dim)

    # Back to (num_tokens, num_heads * head_dim) -- matches vLLM output format
    result = attn_out.squeeze(0).transpose(0, 1).reshape(num_actual, num_q_heads * head_dim)

    if state._log_count < 3 and state.layer_idx == 0:
        logger.info(
            f"[TQ PREFILL L{state.layer_idx}] tokens={num_actual} "
            f"q_heads={num_q_heads} kv_heads={num_kv_heads} head_dim={head_dim}"
        )
        state._log_count += 1

    return result.to(query.dtype)


def _expand_cache_for_gqa(tensor: torch.Tensor, num_kv_heads: int, num_query_heads: int) -> torch.Tensor:
    """Expand a (num_kv_heads, ...) cache tensor to (num_query_heads, ...) by repeating."""
    if num_query_heads == num_kv_heads or tensor.shape[0] != num_kv_heads:
        return tensor
    repeats = num_query_heads // num_kv_heads
    return tensor.repeat_interleave(repeats, dim=0)


def _build_flash_decode_batch(
    state: TurboQuantLayerState,
    query: torch.Tensor,
    attn_metadata,
    cache: TurboQuantKVCache,
) -> Optional[DecodeBatch]:
    num_actual = attn_metadata.num_actual_tokens
    q = query[:num_actual]
    if q.dim() == 2:
        q = q.unsqueeze(1)

    if q.shape[-1] != state.head_dim:
        return None

    flat = state.get_flat_cache(cache)
    if flat is None:
        return None
    flat_prod_q, flat_value_q, quantized_tokens = flat

    num_query_heads = q.shape[1]
    num_kv_heads = state.num_kv_heads

    # Flatten query: (num_actual, num_query_heads, D) -> (num_query_heads, 1, D)
    flat_query = q.squeeze(0).unsqueeze(1).contiguous()  # (num_query_heads, 1, D)

    # For GQA: expand cache from num_kv_heads to num_query_heads by repeating
    if num_query_heads != num_kv_heads:
        exp = lambda t: _expand_cache_for_gqa(t, num_kv_heads, num_query_heads)
        flat_prod_q = ProdQuantized(
            mse_indices=exp(flat_prod_q.mse_indices),
            qjl_signs=exp(flat_prod_q.qjl_signs),
            norms=exp(flat_prod_q.norms),
            residual_norms=exp(flat_prod_q.residual_norms),
            mse_bits=flat_prod_q.mse_bits,
        )
        flat_value_q = ValueQuantized(
            data=exp(flat_value_q.data),
            scales=exp(flat_value_q.scales),
            zeros=exp(flat_value_q.zeros),
            bits=flat_value_q.bits,
        )

    return DecodeBatch(
        query_heads=flat_query,
        prod_q_flat=flat_prod_q,
        value_q_flat=flat_value_q,
        num_tokens=num_actual,
        num_query_heads=num_query_heads,
        head_dim=state.head_dim,
        quantized_tokens=quantized_tokens,
    )


def _run_flash_turboquant_decode(
    state: TurboQuantLayerState,
    self_impl,
    decode_batch: DecodeBatch,
) -> Optional[torch.Tensor]:
    try:
        from turboquant.triton_kernels import turboquant_fused_decode

        tq_flat = turboquant_fused_decode(
            decode_batch.query_heads,
            decode_batch.prod_q_flat,
            decode_batch.value_q_flat,
            state.quantizer.mse_quantizer.Pi,
            state.quantizer.S,
            state.quantizer.mse_quantizer.centroids,
            decode_batch.prod_q_flat.mse_bits,
            state.quantizer.qjl_scale,
            self_impl.scale,
            state.value_group_size,
        )
        return tq_flat.reshape(decode_batch.num_tokens, decode_batch.num_query_heads, decode_batch.head_dim)
    except Exception as exc:
        if not hasattr(state, "_err_logged"):
            logger.warning(f"[TQ {_GLOBAL_MODE}] Triton error L{state.layer_idx}: {exc}")
            state._err_logged = True
        return None


def install_turboquant_hooks(
    model_runner,
    key_bits: int = 3,
    value_bits: int = 2,
    value_group_size: int = 32,
    buffer_size: int = 128,
    initial_layers_count: int = 4,
    initial_layers_key_bits: int | None = None,
    mode: str = MODE_ACCUMULATE,
    no_alloc: bool = False,
):
    global _GLOBAL_MODE
    _GLOBAL_MODE = mode

    if initial_layers_key_bits is None:
        initial_layers_key_bits = min(key_bits + 1, 4)

    static_ctx = model_runner.compilation_config.static_forward_context
    device = model_runner.device

    tq_states = {}
    layer_idx = 0

    for layer_name, attn_module in static_ctx.items():
        if not hasattr(attn_module, "impl"):
            continue

        impl = attn_module.impl
        num_kv_heads = getattr(impl, "num_kv_heads", None)
        if num_kv_heads is None:
            continue

        if hasattr(impl, "head_size"):
            head_dim = int(impl.head_size)
        elif hasattr(impl, "kv_lora_rank"):
            head_dim = int(impl.kv_lora_rank)
        else:
            continue

        bits = initial_layers_key_bits if layer_idx < initial_layers_count else key_bits
        backend_kind = "mla" if _is_mla_impl(impl) else "flash"
        num_query_heads = _infer_num_query_heads(attn_module, impl)

        state = TurboQuantLayerState(
            head_dim=head_dim,
            num_kv_heads=int(num_kv_heads),
            num_query_heads=num_query_heads,
            key_bits=bits,
            value_bits=value_bits,
            value_group_size=min(value_group_size, head_dim),
            buffer_size=buffer_size,
            device=device,
            layer_idx=layer_idx,
            backend_kind=backend_kind,
        )
        tq_states[layer_name] = state

        if backend_kind == "flash":
            patched_forward = PatchedFlashForward(impl.forward.__func__, state, no_alloc=no_alloc)
            impl.forward = types.MethodType(patched_forward, impl)
            if no_alloc:
                patched_update = PatchedFlashKVCacheUpdate(impl.do_kv_cache_update.__func__, state, no_alloc=True)
                impl.do_kv_cache_update = types.MethodType(patched_update, impl)
        else:
            patched_forward_mqa = PatchedMLAForwardMQA(impl.forward_mqa.__func__, state)
            impl.forward_mqa = types.MethodType(patched_forward_mqa, impl)
            if no_alloc:
                patched_update = PatchedMLAKVCacheUpdate(impl.do_kv_cache_update.__func__, state)
                impl.do_kv_cache_update = types.MethodType(patched_update, impl)

        impl._tq_state = state
        layer_idx += 1

    model_runner._tq_states = tq_states
    model_runner._tq_no_alloc = no_alloc
    logger.info(f"[TurboQuant] Hooks on {len(tq_states)} layers (mode={mode}, no_alloc={no_alloc})")
    return tq_states


_TQ_NO_ALLOC_CONFIG = None


def enable_no_alloc(
    key_bits: int = 3,
    value_bits: int = 2,
    buffer_size: int = 128,
    initial_layers_count: int = 4,
):
    """
    Call BEFORE creating vllm.LLM(). Patches the executor so that during
    engine initialization, TQ hooks are installed on all workers automatically.

    After prefill, call free_kv_cache() via collective_rpc to release the
    paged KV cache for TQ layers -- TQ's compressed store holds all old tokens.
    """
    global _TQ_NO_ALLOC_CONFIG
    _TQ_NO_ALLOC_CONFIG = dict(
        key_bits=key_bits,
        value_bits=value_bits,
        buffer_size=buffer_size,
        initial_layers_count=initial_layers_count,
    )

    from vllm.v1.executor.abstract import Executor

    if hasattr(Executor, "_tq_patched"):
        return

    orig_get_specs = Executor.get_kv_cache_specs

    def patched_get_kv_cache_specs(self):
        cfg = _TQ_NO_ALLOC_CONFIG
        if cfg is None:
            return orig_get_specs(self)

        def _worker_install_tq(worker):
            from turboquant.vllm_attn_backend import (
                install_turboquant_hooks, MODE_ACTIVE
            )
            tq_states = install_turboquant_hooks(
                worker.model_runner,
                key_bits=cfg["key_bits"],
                value_bits=cfg["value_bits"],
                buffer_size=cfg["buffer_size"],
                initial_layers_count=cfg["initial_layers_count"],
                mode=MODE_ACTIVE,
                no_alloc=True,
            )
            return len(tq_states)

        hooks = self.collective_rpc(_worker_install_tq)
        logger.info(f"[TurboQuant] Installed hooks: {hooks} layers per worker")

        return orig_get_specs(self)

    Executor.get_kv_cache_specs = patched_get_kv_cache_specs
    Executor._tq_patched = True
    logger.info("[TurboQuant] Patched Executor for auto TQ hook installation")


def capture_kv_cache(model_runner, num_tokens: int = 0):
    """
    Read K/V data from vLLM's paged KV cache and compress into TQ format.

    Call this AFTER generation (prefill + decode) but BEFORE free_kv_cache().
    Processes one layer at a time to minimize peak memory.

    Args:
        model_runner: the vLLM model runner
        num_tokens: number of tokens to capture. If 0, uses all allocated blocks.
    Returns: number of tokens captured per layer.
    """
    tq_states = getattr(model_runner, "_tq_states", None)
    if not tq_states:
        return 0

    static_ctx = model_runner.compilation_config.static_forward_context
    captured = 0

    for layer_name, state in tq_states.items():
        if layer_name not in static_ctx:
            continue
        attn_module = static_ctx[layer_name]
        kv_list = getattr(attn_module, "kv_cache", None)
        if not kv_list or len(kv_list) == 0:
            continue
        kv_tensor = kv_list[0]
        if kv_tensor.numel() <= 1:
            continue

        # kv_tensor shape: (2, num_blocks, block_size, num_kv_heads, head_size)
        num_blocks, block_size, num_kv_heads, head_size = kv_tensor[0].shape

        if num_tokens > 0:
            n_blocks_used = min((num_tokens + block_size - 1) // block_size, num_blocks)
            total_tokens = min(num_tokens, n_blocks_used * block_size)
        else:
            n_blocks_used = num_blocks
            total_tokens = num_blocks * block_size

        # Read keys and values from paged cache -- reshape in-place, no copy
        # (num_blocks, block_size, kv_heads, dim) -> (total_tok, kv_heads, dim)
        k_flat = kv_tensor[0, :n_blocks_used].reshape(-1, num_kv_heads, head_size)[:total_tokens]
        v_flat = kv_tensor[1, :n_blocks_used].reshape(-1, num_kv_heads, head_size)[:total_tokens]

        # TQ cache API expects (1, kv_heads, seq_len, dim)
        # Use contiguous views to avoid large allocations
        k_for_tq = k_flat.permute(1, 0, 2).unsqueeze(0)  # (1, kv_heads, tok, dim)
        v_for_tq = v_flat.permute(1, 0, 2).unsqueeze(0)

        # Reset any existing TQ cache and compress
        state.seq_caches.clear()
        state._ring_pos = 0
        state.invalidate_flat_cache()

        cache = state.get_or_create_cache("default")
        cache.prefill(k_for_tq, v_for_tq)
        captured = total_tokens

        # Free intermediate refs immediately
        del k_flat, v_flat, k_for_tq, v_for_tq
        torch.cuda.empty_cache()

    logger.info(f"[TurboQuant] Captured {captured} tokens from paged KV cache into TQ ({len(tq_states)} layers)")
    return captured


def free_kv_cache(model_runner):
    """
    Replace KV cache tensors for TQ-hooked layers with tiny 1-byte tensors.
    Frees the GPU memory that was allocated for the standard paged KV cache.

    Automatically calls capture_kv_cache() first if TQ states have no data.
    After this call, only TQ ACTIVE mode decode works (flash attention will fail).

    Returns: bytes freed.
    """
    tq_states = getattr(model_runner, "_tq_states", None)
    if not tq_states:
        logger.warning("[TurboQuant] No TQ states found, nothing to free")
        return 0

    # Auto-capture from paged cache if TQ doesn't have data yet
    first_state = next(iter(tq_states.values()))
    if not first_state.seq_caches.get("default") or first_state.seq_caches["default"].seq_len == 0:
        capture_kv_cache(model_runner)

    static_ctx = model_runner.compilation_config.static_forward_context
    device = model_runner.device
    freed = 0
    tiny = torch.zeros(1, dtype=torch.int8, device=device)

    # Only replace KV cache tensors for TQ-hooked layers.
    # Other layers (e.g. linear/GDN attention in Qwen3.5) keep their caches.
    hooked_ptrs = set()
    for layer_name in tq_states:
        if layer_name not in static_ctx:
            continue
        attn_module = static_ctx[layer_name]
        kv_list = getattr(attn_module, "kv_cache", None)
        if kv_list and len(kv_list) > 0:
            old = kv_list[0]
            if old.numel() > 1:
                hooked_ptrs.add(old.data_ptr())
                freed += old.nelement() * old.element_size()
                kv_list[0] = tiny

    # Also replace matching entries in runner_kv_caches, but ONLY for
    # tensors that belong to hooked layers (identified by data_ptr).
    for i in range(len(model_runner.kv_caches)):
        entry = model_runner.kv_caches[i]
        if isinstance(entry, list):
            for j in range(len(entry)):
                if hasattr(entry[j], 'data_ptr') and entry[j].data_ptr() in hooked_ptrs:
                    entry[j] = tiny
        elif hasattr(entry, 'data_ptr') and entry.data_ptr() in hooked_ptrs:
            model_runner.kv_caches[i] = tiny

    torch.cuda.empty_cache()
    logger.info(f"[TurboQuant] Freed {freed/1e6:.0f} MB KV cache ({len(tq_states)} layers)")
    return freed
