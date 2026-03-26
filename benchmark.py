#!/usr/bin/env python3
"""
TurboQuant comprehensive benchmark.
Tests: VRAM, throughput (tok/s), quality, context capacity.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,4,6 MODEL=Qwen3.5-27B python benchmark.py
"""
import os, sys, subprocess, json, time

PYTHON = sys.executable
GPUS = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,4,6")

MODELS = {
    "Qwen2.5-7B-Instruct": {
        "path": "/mnt/llm_models/Qwen2.5-7B-Instruct",
        "tp": 2, "gpu_mem": 0.90, "max_model_len": 32768,
        "block_size": 16, "dtype": "bfloat16",
    },
    "Qwen3.5-27B": {
        "path": "/mnt/llm_models/Qwen3.5-27B",
        "tp": 4, "gpu_mem": 0.90, "max_model_len": 131072,
        "block_size": 784, "dtype": "bfloat16",
    },
}

PROMPT = "Explain how KV cache compression works in large language model inference."
QUALITY_PROMPT = "Answer precisely: 1) Capital of France? 2) 17*23? 3) Who wrote Romeo and Juliet? 4) Chemical formula for water? 5) Year WWII ended?"


def run_script(name, code):
    path = f"/tmp/tq_{name}.py"
    with open(path, "w") as f:
        f.write(code)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPUS
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env["TOKENIZERS_PARALLELISM"] = "false"
    r = subprocess.run([PYTHON, path], capture_output=True, text=True, env=env, timeout=600)
    if r.returncode != 0:
        print(f"  {name} FAILED")
        for l in r.stderr.split("\n"):
            if "Error" in l and "Warning" not in l and "Future" not in l:
                print(f"    {l.strip()}")
        return None
    for line in reversed(r.stdout.strip().split("\n")):
        try:
            return json.loads(line)
        except:
            pass
    return None


def baseline_code(m):
    return f'''
import os, json, subprocess, time
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

def main():
    from vllm import LLM, SamplingParams
    llm = LLM(model="{m['path']}", dtype="{m['dtype']}", gpu_memory_utilization={m['gpu_mem']},
        max_model_len={m['max_model_len']}, tensor_parallel_size={m['tp']},
        trust_remote_code=True, max_num_seqs=1)
    blocks = llm.llm_engine.vllm_config.cache_config.num_gpu_blocks

    # Throughput
    t0 = time.perf_counter()
    out = llm.generate(["{PROMPT}"], SamplingParams(temperature=0, max_tokens=256))
    t1 = time.perf_counter()
    toks = len(out[0].outputs[0].token_ids)
    text = out[0].outputs[0].text[:200]

    # Quality
    qout = llm.generate(["{QUALITY_PROMPT}"], SamplingParams(temperature=0, max_tokens=256))
    quality = qout[0].outputs[0].text[:300]

    r = subprocess.run(["nvidia-smi","--query-gpu=index,memory.used","--format=csv,noheader,nounits"],
        capture_output=True, text=True)
    vram = [int(l.split(",")[1].strip()) for l in r.stdout.strip().split("\\n") if l.strip()]

    print(json.dumps({{"blocks": blocks, "toks": toks, "elapsed": round(t1-t0,3),
        "tps": round(toks/(t1-t0),1), "vram": vram, "text": text, "quality": quality}}))

if __name__ == "__main__":
    main()
'''


def tq_code(m):
    return f'''
import os, json, subprocess, time
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

def main():
    from vllm import LLM, SamplingParams
    llm = LLM(model="{m['path']}", dtype="{m['dtype']}", gpu_memory_utilization={m['gpu_mem']},
        max_model_len={m['max_model_len']}, tensor_parallel_size={m['tp']},
        trust_remote_code=True, max_num_seqs=1)
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

    # Throughput
    t0 = time.perf_counter()
    out = llm.generate(["{PROMPT}"], SamplingParams(temperature=0, max_tokens=256))
    t1 = time.perf_counter()
    toks = len(out[0].outputs[0].token_ids)
    text = out[0].outputs[0].text[:200]

    # Quality (before freeing KV cache -- need paged cache for new prefill)
    def _reset(worker):
        tq_states = getattr(worker.model_runner, "_tq_states", {{}})
        for s in tq_states.values():
            s.reset()
        return len(tq_states)
    executor.collective_rpc(_reset)

    qout = llm.generate(["{QUALITY_PROMPT}"], SamplingParams(temperature=0, max_tokens=256))
    quality = qout[0].outputs[0].text[:300]

    r = subprocess.run(["nvidia-smi","--query-gpu=index,memory.used","--format=csv,noheader,nounits"],
        capture_output=True, text=True)
    vram_gen = [int(l.split(",")[1].strip()) for l in r.stdout.strip().split("\\n") if l.strip()]

    # Free KV cache (after all generation is done)
    def _free(worker):
        from turboquant.vllm_attn_backend import free_kv_cache
        return free_kv_cache(worker.model_runner)
    freed = executor.collective_rpc(_free)

    r2 = subprocess.run(["nvidia-smi","--query-gpu=index,memory.used","--format=csv,noheader,nounits"],
        capture_output=True, text=True)
    vram_freed = [int(l.split(",")[1].strip()) for l in r2.stdout.strip().split("\\n") if l.strip()]

    print(json.dumps({{"blocks": blocks, "hooks": hooks[0], "toks": toks,
        "elapsed": round(t1-t0,3), "tps": round(toks/(t1-t0),1),
        "vram_gen": vram_gen, "vram_freed": vram_freed, "freed": freed,
        "text": text, "quality": quality}}))

if __name__ == "__main__":
    main()
'''


def run_model(name, m):
    n = m["tp"]
    bs = m["block_size"]

    print(f"\\n{'#'*60}")
    print(f"# {name} (TP={n})")
    print(f"{'#'*60}")

    print("  Baseline ...", flush=True)
    bl = run_script(f"bl_{name}", baseline_code(m))
    if not bl:
        return None

    print("  TurboQuant ...", flush=True)
    tq = run_script(f"tq_{name}", tq_code(m))
    if not tq:
        return None

    freed_per = tq["freed"][0]
    freed_total = sum(tq["freed"])

    bl_tokens = bl["blocks"] * bs
    # Estimate extra capacity from freed bytes
    # Very rough: freed_per / (page_size_per_block * tq_layers)
    extra_blocks = int(freed_per / max(bl_tokens * 2, 1))  # rough estimate

    print(f"\\n  {'='*56}")
    print(f"  VRAM")
    print(f"    Baseline:      {bl['vram'][:n]} MB/GPU")
    print(f"    TQ after gen:  {tq['vram_gen'][:n]} MB/GPU")
    print(f"    TQ after free: {tq['vram_freed'][:n]} MB/GPU")
    print(f"    Freed/GPU:     {freed_per/1e6:.0f} MB")
    print(f"    Total freed:   {freed_total/1e6:.0f} MB ({freed_total/1e9:.1f} GB)")
    print(f"  THROUGHPUT")
    print(f"    Baseline:      {bl['tps']} tok/s ({bl['toks']} tokens, {bl['elapsed']}s)")
    print(f"    TQ:            {tq['tps']} tok/s ({tq['toks']} tokens, {tq['elapsed']}s)")
    print(f"    Ratio:         {tq['tps']/max(bl['tps'],0.1):.2f}x")
    print(f"  CONTEXT")
    print(f"    Baseline:      {bl_tokens:,} tokens ({bl['blocks']} blocks x {bs})")
    print(f"    TQ layers:     {tq['hooks']}")
    print(f"  QUALITY")
    print(f"    Baseline: {bl['quality'][:200]}")
    print(f"    TQ:       {tq['quality'][:200]}")
    print(f"  OUTPUT")
    print(f"    Baseline: {bl['text'][:150]}")
    print(f"    TQ:       {tq['text'][:150]}")
    print(f"  {'='*56}")

    return {"model": name, "bl_tps": bl["tps"], "tq_tps": tq["tps"],
            "freed_mb": round(freed_total/1e6), "hooks": tq["hooks"],
            "bl_blocks": bl["blocks"], "bl_tokens": bl_tokens}


def main():
    target = os.environ.get("MODEL")
    to_run = {}
    for name, m in MODELS.items():
        if target and target not in name and target != m["path"]:
            continue
        to_run[name] = m

    if not to_run:
        print(f"No matching model for MODEL={target}")
        print(f"Available: {list(MODELS.keys())}")
        return

    results = []
    for name, m in to_run.items():
        r = run_model(name, m)
        if r:
            results.append(r)

    if results:
        print(f"\\n{'='*60}")
        print("SUMMARY")
        print(f"{'Model':<25} {'Hooks':>6} {'BL tok/s':>9} {'TQ tok/s':>9} {'Freed':>8}")
        for r in results:
            print(f"{r['model']:<25} {r['hooks']:>6} {r['bl_tps']:>9} {r['tq_tps']:>9} {r['freed_mb']:>6} MB")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
