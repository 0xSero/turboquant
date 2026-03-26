#!/usr/bin/env python3
"""
TurboQuant definitive proof. Two separate subprocesses:
  1. Baseline vLLM
  2. TurboQuant + free_kv_cache
Hard numbers side by side.
"""
import os, sys, subprocess, json

MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-27B")
TP = int(os.environ.get("TP", "4"))
GPU_MEM = float(os.environ.get("GPU_MEM", "0.90"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "131072"))
GPUS = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,4,6")
PYTHON = sys.executable


def run_phase(name, script):
    path = f"/tmp/tq_{name}.py"
    with open(path, "w") as f:
        f.write(script)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPUS
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env["TOKENIZERS_PARALLELISM"] = "false"
    r = subprocess.run([PYTHON, path], capture_output=True, text=True, env=env, timeout=600)
    if r.returncode != 0:
        print(f"=== {name} FAILED ===")
        # Find the actual error
        for line in r.stderr.split("\n"):
            if "Error" in line or "error" in line:
                print(f"  {line.strip()}")
        return None
    for line in reversed(r.stdout.strip().split("\n")):
        try:
            return json.loads(line)
        except:
            continue
    return None


BASELINE = f'''
import os, json, subprocess
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

def main():
    import sys
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="{MODEL}", dtype="bfloat16",
        gpu_memory_utilization={GPU_MEM},
        max_model_len={MAX_MODEL_LEN},
        tensor_parallel_size={TP},
        trust_remote_code=True, max_num_seqs=1,
    )
    blocks = llm.llm_engine.vllm_config.cache_config.num_gpu_blocks

    r = subprocess.run(["nvidia-smi","--query-gpu=index,memory.used","--format=csv,noheader,nounits"],
        capture_output=True, text=True)
    vram = [int(l.split(",")[1].strip()) for l in r.stdout.strip().split("\\n") if l.strip()]

    out = llm.generate(["Explain KV cache compression in LLM inference."],
        SamplingParams(temperature=0, max_tokens=64))

    r2 = subprocess.run(["nvidia-smi","--query-gpu=index,memory.used","--format=csv,noheader,nounits"],
        capture_output=True, text=True)
    vram2 = [int(l.split(",")[1].strip()) for l in r2.stdout.strip().split("\\n") if l.strip()]

    print(json.dumps({{"blocks": blocks, "vram_load": vram, "vram_gen": vram2,
        "text": out[0].outputs[0].text[:100]}}))

if __name__ == "__main__":
    main()
'''

TQ = f'''
import os, json, subprocess
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

def main():
    import sys
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="{MODEL}", dtype="bfloat16",
        gpu_memory_utilization={GPU_MEM},
        max_model_len={MAX_MODEL_LEN},
        tensor_parallel_size={TP},
        trust_remote_code=True, max_num_seqs=1,
    )
    blocks = llm.llm_engine.vllm_config.cache_config.num_gpu_blocks

    engine = llm.llm_engine
    core = getattr(engine, "engine_core", engine)
    inner = getattr(core, "engine_core", core)
    executor = inner.model_executor

    def _install(worker):
        from turboquant.vllm_attn_backend import install_turboquant_hooks, MODE_ACTIVE
        return len(install_turboquant_hooks(worker.model_runner, key_bits=3, value_bits=2,
            buffer_size=128, mode=MODE_ACTIVE))
    hooks = executor.collective_rpc(_install)

    out = llm.generate(["Explain KV cache compression in LLM inference."],
        SamplingParams(temperature=0, max_tokens=64))

    r = subprocess.run(["nvidia-smi","--query-gpu=index,memory.used","--format=csv,noheader,nounits"],
        capture_output=True, text=True)
    vram_gen = [int(l.split(",")[1].strip()) for l in r.stdout.strip().split("\\n") if l.strip()]

    def _free(worker):
        from turboquant.vllm_attn_backend import free_kv_cache
        return free_kv_cache(worker.model_runner)
    freed = executor.collective_rpc(_free)

    r2 = subprocess.run(["nvidia-smi","--query-gpu=index,memory.used","--format=csv,noheader,nounits"],
        capture_output=True, text=True)
    vram_freed = [int(l.split(",")[1].strip()) for l in r2.stdout.strip().split("\\n") if l.strip()]

    print(json.dumps({{"blocks": blocks, "hooks": hooks[0], "vram_gen": vram_gen,
        "vram_freed": vram_freed, "freed_bytes": freed,
        "text": out[0].outputs[0].text[:100]}}))

if __name__ == "__main__":
    main()
'''


def main():
    print(f"Model: {MODEL}")
    print(f"TP={TP}, GPU_MEM={GPU_MEM}, MAX_MODEL_LEN={MAX_MODEL_LEN}")
    print(f"GPUs: {GPUS}")
    print()

    print(">>> Phase 1: Baseline ...", flush=True)
    bl = run_phase("baseline", BASELINE)
    if not bl:
        return

    print(">>> Phase 2: TurboQuant ...", flush=True)
    tq = run_phase("tq", TQ)
    if not tq:
        return

    n = len(GPUS.split(","))
    bl_v = bl["vram_gen"][:n]
    tq_v = tq["vram_gen"][:n]
    tq_f = tq["vram_freed"][:n]

    freed_total = sum(tq["freed_bytes"])
    freed_per = tq["freed_bytes"][0]

    block_size = 784  # Qwen3.5-27B: attention block aligned to mamba
    bl_tokens = bl["blocks"] * block_size
    # Extra capacity from freed KV cache
    # full_attn: 16 layers, kv_heads=1/gpu, head_dim=256, bf16=2, K+V=2
    bytes_per_block_full = 2 * 1 * 256 * 2 * block_size * tq["hooks"]
    extra_blocks = int(freed_per / max(bytes_per_block_full, 1))
    new_tokens = bl_tokens + extra_blocks * block_size

    print()
    print("=" * 70)
    print(f"  MODEL: {MODEL}")
    print(f"  TP={TP}, max_model_len={MAX_MODEL_LEN}, gpu_mem={GPU_MEM}")
    print()
    print(f"  BASELINE (vanilla vLLM)")
    print(f"    KV cache blocks:         {bl['blocks']}")
    print(f"    Max tokens:              {bl_tokens:,}")
    print(f"    VRAM/GPU after gen:      {bl_v} MB")
    print()
    print(f"  TURBOQUANT (3-bit key, 2-bit value, {tq['hooks']} full_attn layers)")
    print(f"    KV cache blocks:         {tq['blocks']}  (same initial alloc)")
    print(f"    VRAM/GPU after gen:      {tq_v} MB")
    print(f"    VRAM/GPU after free:     {tq_f} MB")
    print(f"    Tensor freed/GPU:        {freed_per/1e6:.0f} MB")
    print(f"    Total tensor freed:      {freed_total/1e6:.0f} MB ({freed_total/1e9:.1f} GB)")
    print()
    print(f"  RESULT")
    print(f"    KV VRAM saved/GPU:       {freed_per/1e6:.0f} MB")
    print(f"    Extra blocks possible:   {extra_blocks}")
    print(f"    Baseline capacity:       {bl_tokens:,} tokens")
    print(f"    With TQ capacity:        {new_tokens:,} tokens")
    print(f"    Improvement:             {new_tokens/bl_tokens:.2f}x context length")
    print()
    print(f"  OUTPUT COMPARISON")
    print(f"    Baseline: {bl['text']}")
    print(f"    TQ:       {tq['text']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
