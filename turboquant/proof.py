"""Memory proof: computes actual tensor sizes for TQ vs BF16 vs FP8."""


def _tq_bytes_per_token(head_dim, key_bits=3, val_bits=2, val_gs=32):
    mse_bits = key_bits - 1
    packed_d_mse = head_dim // (8 // mse_bits)
    packed_d_signs = head_dim // 8
    packed_d_v = head_dim // (8 // val_bits)
    n_groups = head_dim // val_gs
    return (
        packed_d_mse + packed_d_signs + 2 + 4  # key: indices + signs + norm(bf16) + res_norm(fp32)
        + packed_d_v + n_groups * 2 + n_groups * 2  # value: data + scales(bf16) + zeros(bf16)
    )


def _fixed_per_layer(head_dim):
    return head_dim * head_dim * 4 * 2  # Pi + S matrices, float32


def print_memory_proof():
    configs = [
        ("Qwen3.5-27B (16 full_attn, TP=4)", 16, 1, 256),
        ("Qwen3.5-262B (15 full_attn, TP=8)", 15, 1, 256),
        ("Qwen3-235B (94 layers, TP=8)", 94, 1, 128),
        ("Llama-3-70B (80 layers, TP=8)", 80, 1, 128),
    ]
    ctx_lengths = [4096, 8192, 32768, 65536, 131072]

    print(f"\n{'='*95}")
    print(f"  TurboQuant KV Cache Memory (actual tensor sizes, per GPU)")
    print(f"{'='*95}")

    for name, n_layers, n_kv, hdim in configs:
        tq_per_tok = _tq_bytes_per_token(hdim) * n_kv
        fp8_per_tok = 2 * n_kv * hdim  # K+V, 1 byte each
        bf16_per_tok = 2 * n_kv * hdim * 2  # K+V, 2 bytes each
        fixed = _fixed_per_layer(hdim) * n_layers

        print(f"\n  {name}")
        print(f"  Per-token: BF16={bf16_per_tok}B  FP8={fp8_per_tok}B  TQ={tq_per_tok}B  "
              f"Fixed overhead={fixed/1e6:.1f}MB")
        print(f"  {'Context':>10}  {'BF16':>10}  {'FP8':>10}  {'TQ':>10}  {'vs FP8':>8}  {'Saved':>10}")

        for ctx in ctx_lengths:
            bf16 = n_layers * bf16_per_tok * ctx
            fp8 = n_layers * fp8_per_tok * ctx
            tq = n_layers * tq_per_tok * ctx + fixed
            ratio = tq / max(fp8, 1)
            saved = fp8 - tq
            print(f"  {ctx:>10}  {bf16/1e6:>8.1f}MB  {fp8/1e6:>8.1f}MB  {tq/1e6:>8.1f}MB  "
                  f"{ratio:>7.2f}x  {saved/1e6:>+8.1f}MB")

    print(f"\n{'='*95}")
    print(f"  TQ: 3-bit keys + 2-bit values + metadata. ~2.5x smaller than FP8 at scale.")
    print(f"{'='*95}\n")
