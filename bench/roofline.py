"""Compare measured NxDI latencies with Trn1 hardware floors.

Decode reads every weight once per step  -> bound by HBM bandwidth.
Prefill does ~2*params FLOPs per token    -> bound by TensorE compute.

usage: python bench/roofline.py results/<run> <model_dir>
"""
import glob
import json
import re
import os
import sys

HBM_GBPS = 820       # Trainium1 chip HBM bandwidth (both NeuronCores share it)
BF16_TFLOPS = 190    # Trainium1 chip dense BF16 peak

run_dir, model_dir = sys.argv[1], sys.argv[2]

weight_bytes = sum(os.path.getsize(f) for f in glob.glob(os.path.join(model_dir, "*.safetensors")))
params = weight_bytes / 2  # bf16
env = dict(l.split("=", 1) for l in open(os.path.join(run_dir, "config.env"))
           if "=" in l and not l.lstrip().startswith("#"))
max_ctx = int(env["MAX_CTX"].split()[0])

decode_floor_ms = weight_bytes / (HBM_GBPS * 1e9) * 1e3
prefill_floor_ms = 2 * params * max_ctx / (BF16_TFLOPS * 1e12) * 1e3
print(f"weights {weight_bytes / 1e9:.2f} GB (~{params / 1e9:.2f}B params)")
print(f"decode floor  (batch 1, weights only): {decode_floor_ms:.2f} ms/token")
print(f"prefill floor ({max_ctx} tokens):          {prefill_floor_ms:.2f} ms\n")

for report in sorted(glob.glob(os.path.join(run_dir, "bench_*", "benchmark_report.json"))):
    m = re.match(r"bench_(?:tp(\d+)_)?b(\d+)$", os.path.basename(os.path.dirname(report)))
    tp, batch = m.group(1) or "?", int(m.group(2))
    prefill_floor_b_ms = prefill_floor_ms * batch  # batch x max_ctx tokens of prefill FLOPs
    print(f"## {os.path.relpath(report, run_dir)}  (TP {tp}, batch {batch})")
    for name, stats in json.load(open(report)).items():
        if not isinstance(stats, dict):
            continue
        p50 = stats.get("latency_ms_p50")
        line = f"  {name:28s} p50={p50} p99={stats.get('latency_ms_p99')} thrpt={stats.get('throughput')}"
        if p50 and "token_generation" in name:
            line += f"  -> {decode_floor_ms / p50:.0%} of HBM roofline"
        if p50 and "context_encoding" in name:
            line += f"  -> {prefill_floor_b_ms / p50:.0%} of compute roofline"
        print(line)
