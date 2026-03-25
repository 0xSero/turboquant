#!/usr/bin/env python3
"""
TurboQuant A/B benchmark: baseline vLLM vs TurboQuant KV cache.

Runs two separate processes (clean GPU memory between phases):
  Phase 1: Baseline vLLM generation
  Phase 2: vLLM + TurboQuant hooks (ACTIVE mode)

Usage:
    # Two-phase runner (recommended):
    python benchmark.py --model /path/to/model --tp 4

    # Single phase only:
    python benchmark.py --model /path/to/model --tp 4 --phase baseline
    python benchmark.py --model /path/to/model --tp 4 --phase turboquant
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import time

import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"
if "VLLM_ENABLE_V1_MULTIPROCESSING" not in os.environ:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
logger = logging.getLogger("tq_bench")

PROMPTS = [
    "Write a detailed technical analysis of modern distributed consensus algorithms.",
    "Explain the mathematical foundations of transformer attention mechanisms.",
    "Describe the evolution of GPU computing from early graphics pipelines through modern tensor cores.",
]


def gpu_mem():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True)
    gpus = []
    for line in r.stdout.strip().split("\n"):
        if not line.strip():
            continue
        p = [x.strip() for x in line.split(",")]
        gpus.append({"idx": int(p[0]), "used": int(p[1]), "total": int(p[2])})
    return gpus


def get_executor(llm):
    engine = llm.llm_engine
    if hasattr(engine, "model_executor"):
        return engine.model_executor
    core = getattr(engine, "engine_core", None)
    if core is None:
        raise RuntimeError("No engine_core")
    inner = getattr(core, "engine_core", core)
    if hasattr(inner, "model_executor"):
        return inner.model_executor
    raise RuntimeError("Cannot find model_executor")


def _install_tq(self_worker):
    from turboquant.vllm_attn_backend import install_turboquant_hooks, MODE_ACTIVE
    states = install_turboquant_hooks(
        self_worker.model_runner,
        key_bits=3, value_bits=2, value_group_size=32,
        buffer_size=128, initial_layers_count=4, initial_layers_key_bits=4,
        mode=MODE_ACTIVE,
    )
    return len(states)


def run_phase(args, mode):
    from vllm import LLM, SamplingParams

    logger.info(f"Phase: {mode} | model={args.model} tp={args.tp} gpu_mem={args.gpu_mem}")

    llm = LLM(
        model=args.model, dtype="auto",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        max_num_seqs=args.max_num_seqs,
        kv_cache_dtype=args.kv_cache_dtype,
        enable_chunked_prefill=True,
    )

    if mode == "turboquant":
        executor = get_executor(llm)
        result = executor.collective_rpc(_install_tq)
        logger.info(f"TQ ACTIVE hooks installed: {result}")

    sampling = SamplingParams(temperature=0, max_tokens=args.max_tokens)

    # Warmup
    llm.generate(["Hello"], SamplingParams(max_tokens=10, temperature=0))

    torch.cuda.synchronize()
    times = []
    outputs = None
    total_tokens = 0

    for i in range(args.n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = llm.generate(PROMPTS, sampling)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        if i == 0:
            outputs = out
            for o in out:
                total_tokens += len(o.outputs[0].token_ids)

    mem = gpu_mem()
    avg = sum(times) / len(times)
    tps = total_tokens / times[0]

    result = {
        "mode": mode,
        "model": args.model,
        "tp": args.tp,
        "gpu_mem_util": args.gpu_mem,
        "max_model_len": args.max_model_len,
        "avg_time_s": round(avg, 3),
        "tokens_per_sec": round(tps, 1),
        "total_tokens": total_tokens,
        "times": [round(t, 3) for t in times],
        "gpu_mem_after": mem,
        "total_vram_mb": sum(g["used"] for g in mem),
        "outputs": [
            {"idx": i, "n_tok": len(o.outputs[0].token_ids),
             "text": o.outputs[0].text[:200]}
            for i, o in enumerate(outputs)
        ],
    }

    out_path = f"/tmp/tq_bench_{mode}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"Saved {out_path}")
    return result


def run_both(args):
    python = sys.executable
    script = os.path.abspath(__file__)
    base_cmd = [
        python, script,
        "--model", args.model,
        "--tp", str(args.tp),
        "--gpu-mem", str(args.gpu_mem),
        "--max-model-len", str(args.max_model_len),
        "--max-tokens", str(args.max_tokens),
        "--max-num-seqs", str(args.max_num_seqs),
        "--kv-cache-dtype", args.kv_cache_dtype,
        "--n-runs", str(args.n_runs),
    ]

    for phase in ["baseline", "turboquant"]:
        logger.info(f"\n{'#'*70}\n  PHASE: {phase.upper()}\n{'#'*70}")
        cmd = base_cmd + ["--phase", phase]
        proc = subprocess.run(cmd, timeout=900)
        if proc.returncode != 0:
            logger.error(f"Phase {phase} failed (exit {proc.returncode})")
            sys.exit(1)

    with open("/tmp/tq_bench_baseline.json") as f:
        b = json.load(f)
    with open("/tmp/tq_bench_turboquant.json") as f:
        t = json.load(f)

    bv = b["total_vram_mb"]
    tv = t["total_vram_mb"]

    logger.info(f"\n{'='*70}")
    logger.info(f"  A/B Results: {args.model}")
    logger.info(f"{'='*70}")
    logger.info(f"  {'Metric':<30} {'Baseline':>14} {'TurboQuant':>14} {'Delta':>10}")
    logger.info(f"  {'-'*30} {'-'*14} {'-'*14} {'-'*10}")
    logger.info(f"  {'Tok/s':<30} {b['tokens_per_sec']:14.1f} {t['tokens_per_sec']:14.1f} {t['tokens_per_sec']/max(b['tokens_per_sec'],0.001):9.2f}x")
    logger.info(f"  {'VRAM total (MB)':<30} {bv:14d} {tv:14d} {tv-bv:+9d}")

    n_match = sum(1 for bo, to in zip(b["outputs"], t["outputs"]) if bo["text"][:100] == to["text"][:100])
    logger.info(f"  {'Output match (100 chars)':<30} {n_match}/{len(b['outputs'])}")

    combined = {"baseline": b, "turboquant": t}
    with open("/tmp/tq_bench_results.json", "w") as f:
        json.dump(combined, f, indent=2, default=str)
    logger.info(f"\nFull results: /tmp/tq_bench_results.json")


def main():
    parser = argparse.ArgumentParser(description="TurboQuant A/B Benchmark")
    parser.add_argument("--model", required=True, help="Model path or HF name")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--gpu-mem", type=float, default=0.85, help="gpu_memory_utilization")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--n-runs", type=int, default=2)
    parser.add_argument("--phase", choices=["baseline", "turboquant", "both"], default="both")
    args = parser.parse_args()

    if args.phase == "both":
        run_both(args)
    else:
        run_phase(args, args.phase)


if __name__ == "__main__":
    main()
