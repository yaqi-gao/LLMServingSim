#!/usr/bin/env python3
"""
Generate synthetic profiling data for openai/gpt-oss-20b by scaling from
Mixtral-8x7B H100 measurements.

Scaling rationale: For linear layers, latency scales roughly with the number
of multiply-accumulate operations (proportional to weight matrix size).
For element-wise ops (norms, activations, RoPE), latency scales with the
activation tensor size.

Mixtral-8x7B dimensions:
  hidden=4096, intermediate=14336, num_heads=32, kv_heads=8,
  head_dim=128 (=4096/32), vocab=32000, num_experts=8

GPT-OSS-20B dimensions:
  hidden=2880, intermediate=2880, num_heads=64, kv_heads=8,
  head_dim=64 (explicit), vocab=201088, num_experts=32
"""

import csv
import os

# --- Model configs ---
MIXTRAL = {
    "hidden": 4096,
    "intermediate": 14336,
    "num_heads": 32,
    "kv_heads": 8,
    "head_dim": 128,  # 4096/32
    "vocab": 32000,
    "num_experts": 8,
    "q_out": 4096,       # num_heads * head_dim
    "kv_out": 1024,      # kv_heads * head_dim
}

GPT_OSS = {
    "hidden": 2880,
    "intermediate": 2880,
    "num_heads": 64,
    "kv_heads": 8,
    "head_dim": 64,
    "vocab": 201088,
    "num_experts": 32,
    "q_out": 4096,       # 64 * 64
    "kv_out": 512,       # 8 * 64
}

def compute_scale(layer_name):
    """Compute latency scale factor: gpt_oss / mixtral."""
    m = MIXTRAL
    g = GPT_OSS

    # Linear layers: FLOPs ~ 2*M*N*K, dominated by weight size (N*K)
    if layer_name == "q_proj":
        # weight: hidden x q_out
        return (g["hidden"] * g["q_out"]) / (m["hidden"] * m["q_out"])
    elif layer_name in ("k_proj", "v_proj"):
        # weight: hidden x kv_out
        return (g["hidden"] * g["kv_out"]) / (m["hidden"] * m["kv_out"])
    elif layer_name == "o_proj":
        # weight: q_out x hidden
        return (g["q_out"] * g["hidden"]) / (m["q_out"] * m["hidden"])
    elif layer_name == "gate":
        # weight: hidden x num_experts
        return (g["hidden"] * g["num_experts"]) / (m["hidden"] * m["num_experts"])
    elif layer_name == "expert.w1":
        # weight: hidden x intermediate (per expert, not scaled by num_experts)
        return (g["hidden"] * g["intermediate"]) / (m["hidden"] * m["intermediate"])
    elif layer_name == "expert.w2":
        # weight: intermediate x hidden
        return (g["intermediate"] * g["hidden"]) / (m["intermediate"] * m["hidden"])
    elif layer_name == "expert.w3":
        # weight: hidden x intermediate
        return (g["hidden"] * g["intermediate"]) / (m["hidden"] * m["intermediate"])
    elif layer_name == "embedding":
        # weight: vocab x hidden
        return (g["vocab"] * g["hidden"]) / (m["vocab"] * m["hidden"])
    elif layer_name == "lm_head":
        # weight: hidden x vocab
        return (g["hidden"] * g["vocab"]) / (m["hidden"] * m["vocab"])
    elif layer_name in ("input_layernorm", "post_layernorm", "final_layernorm"):
        # Element-wise: scales with hidden_size
        return g["hidden"] / m["hidden"]
    elif layer_name == "rope":
        # Scales with total head elements: (num_heads + kv_heads) * head_dim
        g_total = (g["num_heads"] + g["kv_heads"]) * g["head_dim"]
        m_total = (m["num_heads"] + m["kv_heads"]) * m["head_dim"]
        return g_total / m_total
    elif layer_name == "act_fn":
        # Element-wise on intermediate dim
        return g["intermediate"] / m["intermediate"]
    else:
        # Fallback: scale by hidden ratio
        return g["hidden"] / m["hidden"]


def generate_profile(src_path, dst_path):
    """Read Mixtral profile, scale latencies, write GPT-OSS profile."""
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    with open(src_path, "r") as fin:
        reader = csv.DictReader(fin)
        rows = list(reader)

    # Pre-compute scales
    scales = {}
    for row in rows:
        ln = row["layer_name"]
        if ln not in scales:
            scales[ln] = compute_scale(ln)

    with open(dst_path, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=["layer_name", "input", "kv_cache", "tp_size", "latency(ns)"])
        writer.writeheader()
        for row in rows:
            ln = row["layer_name"]
            latency = int(row["latency(ns)"])
            scaled = max(1, int(latency * scales[ln]))
            writer.writerow({
                "layer_name": ln,
                "input": row["input"],
                "kv_cache": row["kv_cache"],
                "tp_size": row["tp_size"],
                "latency(ns)": scaled,
            })

    print(f"Generated {dst_path} ({len(rows)} rows)")
    print("Scale factors:")
    for ln, s in sorted(scales.items()):
        print(f"  {ln:20s}: {s:.4f}")


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))

    for tp in [1, 2, 4]:
        src = os.path.join(base, f"perf_models/H100/mistralai/Mixtral-8x7B-v0.1/tp{tp}/layers.csv")
        dst = os.path.join(base, f"perf_models/H100/openai/gpt-oss-20b/tp{tp}/layers.csv")
        if os.path.exists(src):
            generate_profile(src, dst)
        else:
            print(f"Skipping tp{tp}: source {src} not found")
