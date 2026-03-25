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
    """Intercepts do_kv_cache_update to accumulate KV into TurboQuant side-cache.

    CRITICAL PERFORMANCE NOTE:
    In enforce_eager mode, this hook is called on EVERY token including decode.
    We MUST skip single-token decode calls (num_tokens==1) to avoid O(n^2)
    torch.cat growth on the quantized cache. Only prefill batches (num_tokens>1)
    are worth quantizing here. Decode tokens are already handled by vLLM's
    paged cache and the TQ forward hook reads from the quantized prefill data.
    """
    __slots__ = ("orig_fn", "state")

    def __init__(self, orig_fn, state: TurboQuantLayerState):
        self.orig_fn = orig_fn
        self.state = state

    def __call__(self, self_impl, layer, key, value, kv_cache, slot_mapping):
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
    __slots__ = ("orig_fn", "state")

    def __init__(self, orig_fn, state: TurboQuantLayerState):
        self.orig_fn = orig_fn
        self.state = state

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
        mode = _GLOBAL_MODE

        if mode == MODE_ACCUMULATE:
            return self.orig_fn(
                self_impl,
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        if attn_metadata is None or attn_metadata.max_query_len > 1:
            return self.orig_fn(
                self_impl,
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        cache = self.state.seq_caches.get("default")
        flat = self.state.get_flat_cache(cache) if cache is not None else None
        has_tq_data = flat is not None and flat[2] >= 16

        # In ACTIVE mode with TQ data, try to skip flash attention entirely.
        # This is the key to VRAM savings: we don't need the paged cache for
        # old tokens, only the TQ compressed store.
        if mode == MODE_ACTIVE and has_tq_data:
            decode_batch = _build_flash_decode_batch(self.state, query, attn_metadata, cache)
            if decode_batch is not None:
                tq_out = _run_flash_turboquant_decode(self.state, self_impl, decode_batch)
                if tq_out is not None:
                    num_actual = attn_metadata.num_actual_tokens
                    # Reshape to match expected output: (num_actual_tokens, num_query_heads * head_dim)
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
                        output.copy_(result)
                        return output
                    return result

        # Fallback: no TQ data, or TQ failed -- run flash attention
        if not has_tq_data:
            return self.orig_fn(
                self_impl,
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        # SHADOW mode: run both flash and TQ, return flash output
        flash_out = self.orig_fn(
            self_impl,
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        )

        decode_batch = _build_flash_decode_batch(self.state, query, attn_metadata, cache)
        if decode_batch is None:
            return flash_out

        tq_out = _run_flash_turboquant_decode(self.state, self_impl, decode_batch)
        if tq_out is None:
            return flash_out

        if self.state._log_count < 3 and self.state.layer_idx == 0:
            logger.info(
                f"[TQ {mode} L{self.state.layer_idx}] "
                f"out={tuple(tq_out.shape)} q_heads={decode_batch.num_query_heads} "
                f"kv_heads={self.state.num_kv_heads} TQ tokens={decode_batch.quantized_tokens}"
            )
            self.state._log_count += 1

        return flash_out


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


def _repeat_kv_heads_for_gqa(query_heads: torch.Tensor, num_kv_heads: int) -> torch.Tensor:
    num_tokens, num_query_heads, head_dim = query_heads.shape
    if num_query_heads == num_kv_heads:
        return query_heads
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            f"Unsupported GQA layout: query_heads={num_query_heads}, kv_heads={num_kv_heads}"
        )
    groups = num_query_heads // num_kv_heads
    grouped = query_heads.view(num_tokens, num_kv_heads, groups, head_dim)
    return grouped.reshape(num_tokens * groups * num_kv_heads, 1, head_dim)


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
    flat_query = _repeat_kv_heads_for_gqa(q.contiguous(), state.num_kv_heads)

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
            patched_update = PatchedFlashKVCacheUpdate(impl.do_kv_cache_update.__func__, state)  # noqa: E501
            patched_forward = PatchedFlashForward(impl.forward.__func__, state)
            impl.do_kv_cache_update = types.MethodType(
                lambda self, *a, _p=patched_update, **k: _p(self, *a, **k), impl
            )
            impl.forward = types.MethodType(
                lambda self, *a, _p=patched_forward, **k: _p(self, *a, **k), impl
            )
        else:
            patched_update = PatchedMLAKVCacheUpdate(impl.do_kv_cache_update.__func__, state)
            patched_forward_mqa = PatchedMLAForwardMQA(impl.forward_mqa.__func__, state)
            impl.do_kv_cache_update = types.MethodType(
                lambda self, *a, _p=patched_update, **k: _p(self, *a, **k), impl
            )
            impl.forward_mqa = types.MethodType(
                lambda self, *a, _p=patched_forward_mqa, **k: _p(self, *a, **k), impl
            )

        impl._tq_state = state
        layer_idx += 1

    model_runner._tq_states = tq_states
    logger.info(f"[TurboQuant] Hooks on {len(tq_states)} layers (mode={mode})")
    return tq_states
