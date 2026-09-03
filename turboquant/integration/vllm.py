"""
TurboQuant vLLM integration — thin adapter layer.

Responsibilities:
  - Detect layer/backend type (flash vs MLA/GDN)
  - Install minimal monkey-patches that delegate to capture/store/score
  - Expose clean modes: off | capture_only | hybrid | full_tq
  - Keep patching surface tiny; all real logic lives in capture/store/score

Modes:
  - off:          no TQ activity, passthrough
  - capture_only: capture KV into compressed store, always use flash output
  - hybrid:       use compressed history + exact recent for decode
  - full_tq:      (future) TQ handles everything including prefill
"""

from __future__ import annotations

import math
import logging
import types
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

from turboquant.capture import KVCaptureEngine
from turboquant.store import CompressedKVStore
from turboquant.score import compute_hybrid_attention

logger = logging.getLogger("turboquant.integration.vllm")

MODE_OFF = "off"
MODE_CAPTURE_ONLY = "capture_only"
MODE_HYBRID = "hybrid"
MODE_FULL_TQ = "full_tq"
_VALID_MODES = (MODE_OFF, MODE_CAPTURE_ONLY, MODE_HYBRID, MODE_FULL_TQ)

_GLOBAL_MODE = MODE_CAPTURE_ONLY


def set_mode(mode: str):
    global _GLOBAL_MODE
    assert mode in _VALID_MODES, f"Invalid mode: {mode}. Valid: {_VALID_MODES}"
    _GLOBAL_MODE = mode
    logger.info(f"[TurboQuant] Mode set to: {mode}")


def get_mode() -> str:
    return _GLOBAL_MODE


@dataclass
class LayerConfig:
    """Per-layer TQ configuration."""
    head_dim: int
    num_kv_heads: int
    num_query_heads: int
    key_bits: int = 3
    value_bits: int = 2
    value_group_size: int = 32
    ring_capacity: int = 128
    layer_idx: int = 0
    backend_kind: str = "flash"  # "flash" | "mla"
    device: torch.device = field(default_factory=lambda: torch.device("cuda"))


@dataclass
class LayerState:
    """Per-layer runtime state. Owns the capture engine and store."""
    config: LayerConfig
    store: CompressedKVStore
    engine: KVCaptureEngine
    _log_count: int = 0

    @property
    def supports_hybrid(self) -> bool:
        return self.config.backend_kind == "flash"

    def reset(self):
        self.engine.reset()
        self._log_count = 0


def _create_layer_state(cfg: LayerConfig) -> LayerState:
    store = CompressedKVStore(
        head_dim=cfg.head_dim,
        num_kv_heads=cfg.num_kv_heads,
        key_bits=cfg.key_bits,
        value_bits=cfg.value_bits,
        value_group_size=cfg.value_group_size,
        device=cfg.device,
        layer_idx=cfg.layer_idx,
    )
    engine = KVCaptureEngine(
        store=store,
        ring_capacity=cfg.ring_capacity,
        device=cfg.device,
    )
    return LayerState(config=cfg, store=store, engine=engine)


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
    return (
        hasattr(impl, "forward_mqa")
        and hasattr(impl, "do_kv_cache_update")
        and not hasattr(impl, "forward")
    )


# ---------------------------------------------------------------------------
# Patched methods — kept as thin as possible
# ---------------------------------------------------------------------------

def _make_patched_kv_update(orig_fn, state: LayerState, no_alloc: bool = False):
    """Intercept KV cache writes to capture into TQ store."""

    def patched(self_impl, layer, key, value, kv_cache, slot_mapping):
        if not no_alloc:
            # Standard mode: keep paged cache behavior.
            orig_fn(self_impl, layer, key, value, kv_cache, slot_mapping)

        mode = _GLOBAL_MODE
        if mode == MODE_OFF:
            return

        num_tokens = slot_mapping.shape[0]
        if num_tokens <= 1:
            # Decode token — append to ring buffer
            state.engine.ingest_decode(key, value, num_tokens)
        else:
            # Prefill — bulk capture
            state.engine.ingest_prefill(key, value, num_tokens)

    return patched


def _no_alloc_prefill_attention(
    state: LayerState,
    self_impl,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_metadata,
):
    num_actual = attn_metadata.num_actual_tokens
    q = query[:num_actual]
    k = key[:num_actual]
    v = value[:num_actual]

    if q.dim() == 2:
        q = q.view(num_actual, state.config.num_query_heads, state.config.head_dim)
    if k.dim() == 2:
        k = k.view(num_actual, state.config.num_kv_heads, state.config.head_dim)
        v = v.view(num_actual, state.config.num_kv_heads, state.config.head_dim)

    if state.config.num_query_heads != state.config.num_kv_heads:
        repeats = state.config.num_query_heads // state.config.num_kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)

    q_t = q.unsqueeze(0).transpose(1, 2)
    k_t = k.unsqueeze(0).transpose(1, 2)
    v_t = v.unsqueeze(0).transpose(1, 2)

    scale = getattr(self_impl, "scale", 1.0 / math.sqrt(state.config.head_dim))
    out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True, scale=scale)
    return out.squeeze(0).transpose(0, 1)


def _make_patched_forward(orig_fn, state: LayerState, no_alloc: bool = False,
                          capture_in_forward: bool = False):
    """Intercept forward to optionally use TQ decode.
    
    If capture_in_forward=True, also capture K/V from forward args
    (needed when the backend has no separate do_kv_cache_update method).
    """

    def _capture_kv(key, value, attn_metadata):
        """Capture K/V tensors into TQ store."""
        num_tokens = getattr(attn_metadata, 'num_actual_tokens', key.shape[0])
        if num_tokens <= 1:
            state.engine.ingest_decode(key[:num_tokens], value[:num_tokens], num_tokens)
        else:
            state.engine.ingest_prefill(key[:num_tokens], value[:num_tokens], num_tokens)

    def patched(
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

        # Capture K/V when no separate kv_update hook exists
        if capture_in_forward and mode not in (MODE_OFF,) and attn_metadata is not None:
            _capture_kv(key, value, attn_metadata)

        # Off or capture-only: always use flash
        if mode in (MODE_OFF, MODE_CAPTURE_ONLY):
            return orig_fn(
                self_impl, layer, query, key, value, kv_cache,
                attn_metadata, output, output_scale, output_block_scale,
            )

        # Profiling pass or prefill: use flash
        if attn_metadata is None:
            return orig_fn(
                self_impl, layer, query, key, value, kv_cache,
                attn_metadata, output, output_scale, output_block_scale,
            )

        is_prefill = attn_metadata.max_query_len > 1
        if is_prefill:
            if no_alloc:
                result = _no_alloc_prefill_attention(
                    state, self_impl, query, key, value, attn_metadata
                )
                num_actual = attn_metadata.num_actual_tokens
                result_flat = result.reshape(
                    num_actual, state.config.num_query_heads * state.config.head_dim
                ).to(query.dtype)
                if output is not None:
                    out_slice = output[:num_actual]
                    if out_slice.dim() == 3:
                        out_slice.copy_(result.to(out_slice.dtype))
                    else:
                        out_slice.copy_(result_flat.to(out_slice.dtype))
                    return output
                if query.dim() == 3:
                    return result.to(query.dtype)
                return result_flat
            return orig_fn(
                self_impl, layer, query, key, value, kv_cache,
                attn_metadata, output, output_scale, output_block_scale,
            )

        # --- Hybrid decode ---
        if mode == MODE_HYBRID and state.supports_hybrid:
            flat = state.store.get_flat_cache()
            if flat is not None and flat.num_tokens >= 16:
                num_actual = attn_metadata.num_actual_tokens
                q = query[:num_actual]
                if q.dim() == 2:
                    q = q.view(num_actual, state.config.num_query_heads, state.config.head_dim)

                recent = state.engine.ring.peek()
                recent_k = recent[0] if recent else None
                recent_v = recent[1] if recent else None

                result = compute_hybrid_attention(
                    query=q,
                    store=state.store,
                    recent_k=recent_k,
                    recent_v=recent_v,
                    num_query_heads=state.config.num_query_heads,
                    scale=getattr(self_impl, "scale", None),
                )

                result_flat = result.reshape(
                    num_actual, state.config.num_query_heads * state.config.head_dim
                ).to(query.dtype)

                if output is not None:
                    out_slice = output[:num_actual]
                    if out_slice.dim() == 3:
                        out_slice.copy_(result.to(out_slice.dtype))
                    else:
                        out_slice.copy_(result_flat.to(out_slice.dtype))
                    return output
                if query.dim() == 3:
                    return result.to(query.dtype)
                return result_flat

        # Fallback to flash
        if no_alloc:
            num_actual = getattr(attn_metadata, "num_actual_tokens", query.shape[0])
            if query.dim() == 3:
                return torch.zeros_like(query[:num_actual])
            return torch.zeros(
                num_actual,
                state.config.num_query_heads * state.config.head_dim,
                dtype=query.dtype,
                device=query.device,
            )
        return orig_fn(
            self_impl, layer, query, key, value, kv_cache,
            attn_metadata, output, output_scale, output_block_scale,
        )

    return patched


def _make_patched_mla_update(orig_fn, state: LayerState):
    """MLA KV update — log-only, no TQ capture yet."""

    def patched(self_impl, kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale):
        orig_fn(self_impl, kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale)
        if state._log_count < 1:
            logger.info(
                f"[TurboQuant] MLA update observed on layer {state.config.layer_idx}; "
                "TQ MLA path is deferred."
            )
            state._log_count += 1

    return patched


def _make_patched_mla_forward(orig_fn, state: LayerState):
    """MLA forward — passthrough (unsupported)."""

    def patched(self_impl, q, kv_c_and_k_pe_cache, attn_metadata, layer):
        return orig_fn(self_impl, q, kv_c_and_k_pe_cache, attn_metadata, layer)

    return patched


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def install_hooks(
    model_runner,
    key_bits: int = 3,
    value_bits: int = 2,
    value_group_size: int = 32,
    ring_capacity: int = 128,
    initial_layers_count: int = 4,
    initial_layers_key_bits: int | None = None,
    mode: str = MODE_CAPTURE_ONLY,
    no_alloc: bool = False,
) -> dict[str, LayerState]:
    """Install TurboQuant hooks on all attention layers in a vLLM model runner.

    Returns: dict mapping layer_name -> LayerState
    """
    global _GLOBAL_MODE
    _GLOBAL_MODE = mode

    if initial_layers_key_bits is None:
        initial_layers_key_bits = min(key_bits + 1, 4)

    static_ctx = model_runner.compilation_config.static_forward_context
    device = model_runner.device

    layer_states: dict[str, LayerState] = {}
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

        cfg = LayerConfig(
            head_dim=head_dim,
            num_kv_heads=int(num_kv_heads),
            num_query_heads=num_query_heads,
            key_bits=bits,
            value_bits=value_bits,
            value_group_size=min(value_group_size, head_dim),
            ring_capacity=ring_capacity,
            layer_idx=layer_idx,
            backend_kind=backend_kind,
            device=device,
        )

        state = _create_layer_state(cfg)
        layer_states[layer_name] = state

        if backend_kind == "flash":
            has_separate_kv_update = hasattr(impl, "do_kv_cache_update")
            needs_forward_capture = not has_separate_kv_update

            if has_separate_kv_update:
                patched_update = _make_patched_kv_update(
                    impl.do_kv_cache_update.__func__, state, no_alloc=no_alloc
                )
                impl.do_kv_cache_update = types.MethodType(
                    lambda self, *a, _p=patched_update, **kw: _p(self, *a, **kw), impl
                )

            patched_forward = _make_patched_forward(
                impl.forward.__func__, state, no_alloc=no_alloc,
                capture_in_forward=needs_forward_capture,
            )
            impl.forward = types.MethodType(
                lambda self, *a, _p=patched_forward, **kw: _p(self, *a, **kw), impl
            )

            if needs_forward_capture and layer_idx == 0:
                logger.info(
                    "[TurboQuant] No do_kv_cache_update found (vLLM 0.16 FlashInfer); "
                    "capturing K/V in forward()"
                )
        else:
            if hasattr(impl, "do_kv_cache_update"):
                patched_update = _make_patched_mla_update(impl.do_kv_cache_update.__func__, state)
                impl.do_kv_cache_update = types.MethodType(
                    lambda self, *a, _p=patched_update, **kw: _p(self, *a, **kw), impl
                )
            if hasattr(impl, "forward_mqa"):
                patched_fwd = _make_patched_mla_forward(impl.forward_mqa.__func__, state)
                impl.forward_mqa = types.MethodType(
                    lambda self, *a, _p=patched_fwd, **kw: _p(self, *a, **kw), impl
                )

        impl._tq_layer_state = state
        layer_idx += 1

    model_runner._tq_layer_states = layer_states
    model_runner._tq_no_alloc = no_alloc
    logger.info(
        f"[TurboQuant] Hooks on {len(layer_states)} layers "
        f"(mode={mode}, no_alloc={no_alloc})"
    )
    return layer_states


def free_kv_cache(model_runner) -> int:
    """Free paged KV cache for TQ-hooked layers. Returns bytes freed.

    Only frees layers that have TQ state. Non-TQ layers (MLA/GDN) keep their cache.
    """
    layer_states = getattr(model_runner, "_tq_layer_states", None)
    if not layer_states:
        logger.warning("[TurboQuant] No layer states found, nothing to free")
        return 0

    static_ctx = model_runner.compilation_config.static_forward_context
    device = model_runner.device
    freed = 0
    tiny = torch.zeros(1, dtype=torch.int8, device=device)

    ptrs_to_free = set()
    for layer_name, state in layer_states.items():
        if not state.supports_hybrid:
            continue
        if layer_name not in static_ctx:
            continue
        attn_module = static_ctx[layer_name]
        kv_list = getattr(attn_module, "kv_cache", None)
        if kv_list is not None and len(kv_list) > 0:
            ptrs_to_free.add(kv_list[0].data_ptr())

    for layer_name, state in layer_states.items():
        if not state.supports_hybrid:
            continue
        if layer_name not in static_ctx:
            continue
        attn_module = static_ctx[layer_name]
        kv_list = getattr(attn_module, "kv_cache", None)
        if kv_list is not None and len(kv_list) > 0:
            old = kv_list[0]
            freed += old.nelement() * old.element_size()
            kv_list[0] = tiny

    for i in range(len(model_runner.kv_caches)):
        entry = model_runner.kv_caches[i]
        if isinstance(entry, list):
            for j in range(len(entry)):
                if hasattr(entry[j], "data_ptr") and entry[j].data_ptr() in ptrs_to_free:
                    entry[j] = tiny
        elif hasattr(entry, "data_ptr") and entry.data_ptr() in ptrs_to_free:
            model_runner.kv_caches[i] = tiny

    torch.cuda.empty_cache()
    logger.info(f"[TurboQuant] Freed {freed / 1e6:.0f} MB KV cache ({len(layer_states)} layers)")
    return freed


def get_stats(model_runner) -> dict:
    """Return summary statistics for all TQ layer states."""
    layer_states = getattr(model_runner, "_tq_layer_states", None)
    if not layer_states:
        return {}

    stats = {}
    total_compressed = 0
    total_buffered = 0
    total_memory = 0

    for name, state in layer_states.items():
        compressed = state.store.num_tokens
        buffered = state.engine.ring.size
        mem = state.store.memory_bytes()
        total_compressed += compressed
        total_buffered += buffered
        total_memory += mem

    stats["num_layers"] = len(layer_states)
    stats["total_compressed_tokens"] = total_compressed // max(len(layer_states), 1)
    stats["total_buffered_tokens"] = total_buffered // max(len(layer_states), 1)
    stats["total_memory_bytes"] = total_memory
    stats["mode"] = _GLOBAL_MODE
    return stats
