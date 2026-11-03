"""Cross-run figures for the technical report, read from committed results only.

    uv run --with matplotlib paper/figures.py
Writes paper/figs/*.pdf (for LaTeX) and *.png (for posts).
"""
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "figs")
os.makedirs(OUT, exist_ok=True)

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PREFILL, DECODE = "#2a78d6", "#eb6834"  # same slots as bench/plots.py
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})

RUNS = {"TinyLlama-1.1B": "20261029T003425Z", "Llama-3.1-8B": "20261031T032009Z-run"}


def load(run):
    d = os.path.join(ROOT, "results", run)
    roof = open(os.path.join(d, "roofline.log")).read()
    floors = {k: float(re.search(rf"{k} floor.*?:\s*([\d.]+) ms", roof).group(1)) for k in ("decode", "prefill")}
    rows = {}
    for name in os.listdir(d):
        m = re.match(r"bench_tp(\d+)_b(\d+)$", name)
        if m:
            r = json.load(open(os.path.join(d, name, "benchmark_report.json")))
            rows[(int(m.group(1)), int(m.group(2)))] = {
                "prefill": r["context_encoding_model"]["latency_ms_p50"],
                "decode": r["token_generation_model"]["latency_ms_p50"],
            }
    return floors, rows


def save(fig, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200)
    plt.close(fig)


def label_bars(ax, xs, vals, fmt):
    for x, v in zip(xs, vals):
        ax.annotate(fmt.format(v), (x, v), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)


data = {m: load(r) for m, r in RUNS.items()}

# 1. Efficiency vs hardware limit, TP=2, every batch size measured
labels, eff_p, eff_d = [], [], []
for model, (floors, rows) in data.items():
    for (tp, b), r in sorted(rows.items()):
        if tp != 2:
            continue
        labels.append(f"{model}\nbatch {b}")
        eff_p.append(100 * floors["prefill"] * b / r["prefill"])
        eff_d.append(100 * floors["decode"] / r["decode"])
fig, ax = plt.subplots(figsize=(8, 3.9))
xs, w = range(len(labels)), 0.36
for off, vals, color, name in ((-w / 2, eff_p, PREFILL, "Prefill: % of BF16 compute peak"),
                               (w / 2, eff_d, DECODE, "Decode: % of HBM bandwidth (weights only)")):
    pos = [x + off for x in xs]
    ax.bar(pos, vals, width=w, color=color, edgecolor=SURFACE, linewidth=2, label=name, zorder=2)
    label_bars(ax, pos, vals, "{:.0f}%")
ax.set_xticks(list(xs), labels)
ax.set_ylim(0, 100)
ax.set_title("How close each phase gets to the Trainium1 hardware limit (TP=2)")
ax.legend(frameon=False, loc="upper left", fontsize=9)
save(fig, "efficiency")

# 2. TinyLlama: TP=1 vs TP=2 decode step at batch 1 and 4, against the weights-only floor
floors, rows = data["TinyLlama-1.1B"]
fig, ax = plt.subplots(figsize=(6, 3.6))
batches = sorted({b for _, b in rows})
xs, w = range(len(batches)), 0.36
for off, tp, color in ((-w / 2, 1, INK_2), (w / 2, 2, DECODE)):
    vals = [rows[(tp, b)]["decode"] for b in batches]
    pos = [x + off for x in xs]
    ax.bar(pos, vals, width=w, color=color, edgecolor=SURFACE, linewidth=2, label=f"TP={tp}", zorder=2)
    label_bars(ax, pos, vals, "{:.2f}")
ax.axhline(floors["decode"], color=INK_2, linestyle=(0, (3, 2)), linewidth=1.2,
           label=f"weights-only floor ({floors['decode']:.2f} ms)")
ax.set_xticks(list(xs), [f"batch {b}" for b in batches])
ax.set_ylim(0, 12)
ax.set_title("TinyLlama-1.1B decode step, p50 (ms)")
ax.legend(frameon=False, loc="upper left", fontsize=9)
save(fig, "tp_decode")

# 3. Llama-3.1-8B: batching, step time vs decode tokens/s
floors, rows = data["Llama-3.1-8B"]
bs = sorted(b for tp, b in rows if tp == 2)
step = [rows[(2, b)]["decode"] for b in bs]
tps = [1000 * b / s for b, s in zip(bs, step)]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.4))
a1.bar(range(len(bs)), step, width=0.55, color=DECODE, edgecolor=SURFACE, linewidth=2, zorder=2)
label_bars(a1, range(len(bs)), step, "{:.1f}")
a1.axhline(floors["decode"], color=INK_2, linestyle=(0, (3, 2)), linewidth=1.2)
a1.set_xticks(range(len(bs)), [f"batch {b}" for b in bs])
a1.set_ylim(0, max(step) * 1.2)
a1.set_title("Decode step, p50 (ms)")
a2.bar(range(len(bs)), tps, width=0.55, color=INK_2, edgecolor=SURFACE, linewidth=2, zorder=2)
label_bars(a2, range(len(bs)), tps, "{:.0f}")
a2.set_xticks(range(len(bs)), [f"batch {b}" for b in bs])
a2.set_ylim(0, max(tps) * 1.2)
a2.set_title("Decode tokens/s")
save(fig, "batching_8b")

print("\n".join(sorted(os.listdir(OUT))))
