"""
TurboQuant attention backend shim for vLLM v0.17+.

Delegates to turboquant.integration.vllm for all real logic.
Kept for backward compatibility with scripts that import from here.
"""

from __future__ import annotations

import logging
import torch

import turboquant.integration.vllm as _new_backend

logger = logging.getLogger("turboquant.attn")

MODE_SHADOW = "shadow"
MODE_ACCUMULATE = "accumulate"
MODE_ACTIVE = "active"
_VALID_MODES = (MODE_SHADOW, MODE_ACCUMULATE, MODE_ACTIVE)

_LEGACY_TO_NEW = {
    MODE_ACCUMULATE: _new_backend.MODE_CAPTURE_ONLY,
    MODE_SHADOW: _new_backend.MODE_CAPTURE_ONLY,
    MODE_ACTIVE: _new_backend.MODE_HYBRID,
}

_GLOBAL_MODE = MODE_ACCUMULATE


def set_mode(mode: str):
    global _GLOBAL_MODE
    assert mode in _VALID_MODES
    _GLOBAL_MODE = mode
    _new_backend.set_mode(_LEGACY_TO_NEW.get(mode, _new_backend.MODE_CAPTURE_ONLY))


def get_mode() -> str:
    return _GLOBAL_MODE


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
    new_mode = _LEGACY_TO_NEW.get(mode, _new_backend.MODE_CAPTURE_ONLY)

    layer_states = _new_backend.install_hooks(
        model_runner,
        key_bits=key_bits,
        value_bits=value_bits,
        value_group_size=value_group_size,
        ring_capacity=buffer_size,
        initial_layers_count=initial_layers_count,
        initial_layers_key_bits=initial_layers_key_bits,
        mode=new_mode,
        no_alloc=no_alloc,
    )

    _GLOBAL_MODE = mode
    model_runner._tq_states = layer_states
    model_runner._tq_no_alloc = no_alloc
    return layer_states


_TQ_NO_ALLOC_CONFIG = None


def enable_no_alloc(
    key_bits: int = 3,
    value_bits: int = 2,
    buffer_size: int = 128,
    initial_layers_count: int = 4,
):
    """Call BEFORE creating vllm.LLM(). Patches the executor so TQ hooks
    are installed automatically during engine initialization."""
    global _TQ_NO_ALLOC_CONFIG
    _TQ_NO_ALLOC_CONFIG = dict(
        key_bits=key_bits,
        value_bits=value_bits,
        buffer_size=buffer_size,
        initial_layers_count=initial_layers_count,
    )

    from vllm.v1.executor.abstract import Executor
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if hasattr(Executor, "_tq_patched"):
        return

    if not hasattr(GPUModelRunner, "_tq_layout_patch"):
        orig_layout_update = GPUModelRunner._update_hybrid_attention_mamba_layout

        def patched_layout_update(self, kv_caches):
            for layer_name, target_layer_name in getattr(
                self, "shared_kv_cache_layers", {}
            ).items():
                if layer_name not in kv_caches and target_layer_name in kv_caches:
                    kv_caches[layer_name] = kv_caches[target_layer_name]
            return orig_layout_update(self, kv_caches)

        GPUModelRunner._update_hybrid_attention_mamba_layout = patched_layout_update
        GPUModelRunner._tq_layout_patch = True

    orig_get_specs = Executor.get_kv_cache_specs

    def patched_get_kv_cache_specs(self):
        cfg = _TQ_NO_ALLOC_CONFIG
        with open("/tmp/tq_debug.log", "a") as f:
            f.write(f"patched_get_kv_cache_specs called pid={os.getpid()} cfg={cfg is not None}\n")
            f.flush()
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
            static_ctx = worker.model_runner.compilation_config.static_forward_context
            flash_layers = [
                name
                for name, state in tq_states.items()
                if getattr(state, "supports_hybrid", False)
            ]
            shared_layers = 0
            if len(flash_layers) > 1:
                target = flash_layers[0]
                target_attn = static_ctx.get(target)
                if target_attn is not None and hasattr(target_attn, "kv_sharing_target_layer_name"):
                    target_attn.kv_sharing_target_layer_name = None
                for name in flash_layers[1:]:
                    attn = static_ctx.get(name)
                    if attn is None or not hasattr(attn, "kv_sharing_target_layer_name"):
                        continue
                    attn.kv_sharing_target_layer_name = target
                    shared_layers += 1

            return {
                "hooks": len(tq_states),
                "flash_layers": len(flash_layers),
                "shared_layers": shared_layers,
            }

        try:
            hooks = self.collective_rpc(_worker_install_tq)
            print(f"[TurboQuant] Installed no_alloc hooks: {hooks}", flush=True)
        except Exception as e:
            import traceback
            print(f"[TurboQuant] collective_rpc FAILED: {e}", flush=True)
            traceback.print_exc()
        return orig_get_specs(self)

    Executor.get_kv_cache_specs = patched_get_kv_cache_specs
    Executor._tq_patched = True

    # Patch the worker's load_model (NOT decorated, so our patch won't be bypassed)
    try:
        from vllm.v1.worker.gpu_worker import GPUWorker as WorkerCls
    except ImportError:
        try:
            from vllm.v1.worker.gpu_worker import Worker as WorkerCls
        except ImportError:
            WorkerCls = None

    if WorkerCls is not None:
        orig_worker_load = WorkerCls.load_model

        def patched_worker_load(self_worker):
            orig_worker_load(self_worker)
            cfg = _TQ_NO_ALLOC_CONFIG
            if cfg:
                try:
                    import sys
                    sys.path.insert(0, '/tmp')
                    from turboquant.vllm_attn_backend import install_turboquant_hooks, MODE_ACCUMULATE
                    tq = install_turboquant_hooks(
                        self_worker.model_runner,
                        key_bits=cfg["key_bits"],
                        value_bits=cfg["value_bits"],
                        buffer_size=cfg["buffer_size"],
                        initial_layers_count=cfg["initial_layers_count"],
                        mode=MODE_ACCUMULATE,
                        no_alloc=False,
                    )
                    with open("/tmp/tq_debug.log", "a") as f:
                        f.write(f"TQ hooks: {len(tq)} layers pid={os.getpid()}\n")
                        f.flush()
                except Exception as e:
                    with open("/tmp/tq_debug.log", "a") as f:
                        import traceback
                        f.write(f"TQ FAIL pid={os.getpid()}: {e}\n")
                        traceback.print_exc(file=f)
                        f.flush()

        WorkerCls.load_model = patched_worker_load

    logger.info("[TurboQuant] Patched Executor for auto TQ hook installation")


def free_kv_cache(model_runner):
    """Free paged KV cache for TQ-hooked layers."""
    if getattr(model_runner, "_tq_layer_states", None):
        return _new_backend.free_kv_cache(model_runner)

    layer_states = getattr(model_runner, "_tq_states", None)
    if not layer_states:
        return 0

    static_ctx = model_runner.compilation_config.static_forward_context
    device = model_runner.device
    freed = 0
    tiny = torch.zeros(1, dtype=torch.int8, device=device)

    ptrs_to_free = set()
    for layer_name, state in layer_states.items():
        if not getattr(state, "supports_hybrid", False):
            continue
        attn_module = static_ctx.get(layer_name)
        if attn_module is None:
            continue
        kv_list = getattr(attn_module, "kv_cache", None)
        if kv_list is not None and len(kv_list) > 0 and hasattr(kv_list[0], "data_ptr"):
            ptrs_to_free.add(kv_list[0].data_ptr())

    for layer_name, state in layer_states.items():
        if not getattr(state, "supports_hybrid", False):
            continue
        attn_module = static_ctx.get(layer_name)
        if attn_module is None:
            continue
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
    return freed
