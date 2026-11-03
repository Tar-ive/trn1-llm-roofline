"""Charts for one run: phase latency vs hardware floor, roofline efficiency, vLLM serving.

Reads only files committed with the run (bench_*/benchmark_report.json, roofline.log, vllm_*.json),
so it works on the box or later on a laptop:
    python bench/plots.py results/<run>
    uv run --with matplotlib bench/plots.py results/<run>
Writes results/<run>/plots/*.png.
"""
import glob
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PREFILL, DECODE = "#2a78d6", "#eb6834"  # categorical slots 1-2 (validated: CVD dE 24.7)

run = sys.argv[1]
out = os.path.join(run, "plots")
os.makedirs(out, exist_ok=True)
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})

# Hardware floors as computed by roofline.py (prefill floor is for batch 1; scales with batch).
roof = open(os.path.join(run, "roofline.log")).read() if os.path.exists(os.path.join(run, "roofline.log")) else ""
m_dec = re.search(r"decode floor.*?:\s*([\d.]+) ms", roof)
m_pre = re.search(r"prefill floor.*?:\s*([\d.]+) ms", roof)
decode_floor = float(m_dec.group(1)) if m_dec else None
prefill_floor_b1 = float(m_pre.group(1)) if m_pre else None

rows = []
for report in sorted(glob.glob(os.path.join(run, "bench_*", "benchmark_report.json"))):
    m = re.match(r"bench_(?:tp(\d+)_)?b(\d+)$", os.path.basename(os.path.dirname(report)))
    d = json.load(open(report))
    rows.append({
        "label": f"TP{m.group(1) or '?'} · b{m.group(2)}", "batch": int(m.group(2)),
        "prefill": d.get("context_encoding_model", {}).get("latency_ms_p50"),
        "decode": d.get("token_generation_model", {}).get("latency_ms_p50"),
    })
rows.sort(key=lambda r: (r["label"].split("·")[0], r["batch"]))


def bars(ax, labels, values, color, floors=None, fmt="{:.1f}"):
    xs = range(len(labels))
    ax.bar(xs, values, width=0.55, color=color, edgecolor=SURFACE, linewidth=2, zorder=2)
    for x, v in zip(xs, values):
        ax.annotate(fmt.format(v), (x, v), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", color=INK, fontsize=9)
    if floors:
        for x, f in zip(xs, floors):
            if f:
                ax.hlines(f, x - 0.34, x + 0.34, colors=INK_2, linestyles=(0, (3, 2)), linewidth=1.5, zorder=3)
        ax.annotate("hardware floor", (len(labels) - 1 + 0.34, floors[-1]), xytext=(4, 0),
                    textcoords="offset points", va="center", color=INK_2, fontsize=8)
    ax.set_xticks(list(xs), labels)
    ax.set_ylim(0, max(list(values) + [f or 0 for f in (floors or [])]) * 1.18)


if rows:
    labels = [r["label"] for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    bars(a1, labels, [r["prefill"] or 0 for r in rows], PREFILL,
         [prefill_floor_b1 * r["batch"] if prefill_floor_b1 else None for r in rows])
    a1.set_title("Prefill latency, p50 (ms)")
    bars(a2, labels, [r["decode"] or 0 for r in rows], DECODE, [decode_floor] * len(rows), fmt="{:.2f}")
    a2.set_title("Decode step latency, p50 (ms)")
    fig.suptitle(os.path.basename(run.rstrip("/")), x=0.01, ha="left", color=INK_2, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "phase_latency.png"), dpi=150)
    plt.close(fig)

    if decode_floor and prefill_floor_b1:
        eff_p = [100 * prefill_floor_b1 * r["batch"] / r["prefill"] if r["prefill"] else 0 for r in rows]
        eff_d = [100 * decode_floor / r["decode"] if r["decode"] else 0 for r in rows]
        fig, ax = plt.subplots(figsize=(7, 3.8))
        xs = range(len(rows))
        w = 0.36
        for off, vals, color, name in ((-w / 2, eff_p, PREFILL, "Prefill: % of BF16 compute peak"),
                                       (w / 2, eff_d, DECODE, "Decode: % of HBM bandwidth")):
            pos = [x + off for x in xs]
            ax.bar(pos, vals, width=w, color=color, edgecolor=SURFACE, linewidth=2, label=name, zorder=2)
            for x, v in zip(pos, vals):
                ax.annotate(f"{v:.0f}%", (x, v), xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=9)
        ax.set_xticks(list(xs), labels)
        ax.set_ylim(0, 100)
        ax.set_title("How close each phase gets to its hardware limit")
        ax.legend(frameon=False, loc="upper left", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "roofline_efficiency.png"), dpi=150)
        plt.close(fig)

vllm = []
for f in sorted(glob.glob(os.path.join(run, "vllm_*_c*.json"))):
    d = json.load(open(f))
    c = int(re.search(r"_c(\d+)\.json$", f).group(1))
    vllm.append((c, d.get("median_ttft_ms", 0), d.get("median_tpot_ms", 0), d.get("output_throughput", 0)))
if vllm:
    vllm.sort()
    labels = [f"concurrency {c}" for c, *_ in vllm]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    bars(axes[0], labels, [v[1] for v in vllm], PREFILL)
    axes[0].set_title("Time to first token, p50 (ms)")
    bars(axes[1], labels, [v[2] for v in vllm], DECODE, fmt="{:.2f}")
    axes[1].set_title("Time per output token, p50 (ms)")
    bars(axes[2], labels, [v[3] for v in vllm], INK_2, fmt="{:.0f}")
    axes[2].set_title("Output throughput (tok/s)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "vllm_serving.png"), dpi=150)
    plt.close(fig)

print("\n".join(sorted(glob.glob(os.path.join(out, "*.png")))) or "no data to plot")
