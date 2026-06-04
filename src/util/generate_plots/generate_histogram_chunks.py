"""
generate_histogram_chunks.py
----------------------------
Publication-quality figure: per-model IoU (or ΔIoU vs RGB) distributions split
into N equal-sized chunks ordered by RGB IoU rank (high → low).

Each panel shows:
  • KDE curves (smooth density) with a light fill — one per model
  • A dashed vertical mean line per model
  • A paired Wilcoxon significance table inside the panel

Inputs : 3 iou_results.json files (one per model)
Output : one PNG saved to  histogram_charts/

Usage:
    python generate_histogram_chunks.py

    Edit the CONFIG block below.
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
import numpy as np
from scipy.stats import wilcoxon, gaussian_kde

_util_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _util_dir not in sys.path:
    sys.path.insert(0, _util_dir)
from paths import FAILURE_RECREATION_OUTPUT_DIR

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these before running
# ──────────────────────────────────`────────────────────────────────────────────

TITLE      = "IoU Distribution by RGB Performance Quartile - BASELINE"  # title + filename
N_CHUNKS   = 4      # equal-sized rank bins  (3 or 4 recommended)
SHOW_DEPTH = True   # False → hide DEPTH model from all panels
SHOW_DIFF  = False  # True  → plot ΔIoU vs RGB  (DEPTH−RGB, RGD−RGB)
                    #         instead of raw IoU values


def _iou_results_path(base: Path, mode: str, variant: str, run: str) -> Path:
    return base / mode / variant / run / f"json_{run}" / "iou_results.json"


def parse_args():
    default_base = FAILURE_RECREATION_OUTPUT_DIR / "testing_output"
    p = argparse.ArgumentParser(description="Generate IoU distribution histogram by RGB quartile")
    p.add_argument("--base-dir", type=Path, default=default_base,
                   help=f"Base dir holding <mode>/<variant>/<run>/json_<run>/iou_results.json (default: {default_base})")
    p.add_argument("--variant", default="baseline", help="Variant subdir for all models (default: baseline)")
    p.add_argument("--rgb-run", default="RGB-2", help="RGB run name (default: RGB-2)")
    p.add_argument("--depth-run", default="DEPTH-5", help="DEPTH run name (default: DEPTH-5)")
    p.add_argument("--rgd-run", default="RGD-4", help="RGD run name (default: RGD-4)")
    return p.parse_args()


_args = parse_args()
INPUTS = {
    "RGB":   _iou_results_path(_args.base_dir, "rgb", _args.variant, _args.rgb_run),
    "DEPTH": _iou_results_path(_args.base_dir, "depth", _args.variant, _args.depth_run),
    "RGD":   _iou_results_path(_args.base_dir, "rgd", _args.variant, _args.rgd_run),
}

# ──────────────────────────────────────────────────────────────────────────────
# STYLE  (publication-quality)
# ──────────────────────────────────────────────────────────────────────────────

plt.rcParams.update(
    {
        "font.family":        "serif",
        "font.serif":         ["DejaVu Serif", "Times New Roman", "Georgia"],
        "font.size":          9,
        "axes.titlesize":     10,
        "axes.labelsize":     9,
        "xtick.labelsize":    8,
        "ytick.labelsize":    8,
        "legend.fontsize":    9,
        "figure.dpi":         150,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     0.8,
        "xtick.major.width":  0.6,
        "ytick.major.width":  0.6,
    }
)

# Colours — identical to existing line plots
_ALL_COLORS = {
    "RGB":   "#4E79A7",
    "DEPTH": "#F28E2B",
    "RGD":   "#B07AA1",
}

# KDE fill alphas
_FILL_ALPHA = 0.13
_LINE_WIDTH = 2.0

# Chunk labels — one-liner strings, no embedded newlines
_LABELS_4 = ["Q4 — Highest RGB", "Q3", "Q2", "Q1 — Lowest RGB"]
_LABELS_3 = ["High RGB",         "Mid RGB",  "Low RGB"]

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────


def load_iou(path: Path) -> dict:
    """iou_results.json → {image_name: mean_iou}."""
    with open(path) as f:
        data = json.load(f)
    return {name: v["mean_iou"] for name, v in data["per_image_iou"].items()}


def sig_stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def wilcoxon_safe(a, b):
    """
    Paired Wilcoxon signed-rank test on two aligned arrays.
    Returns (p, stars).
    """
    diffs = np.asarray(a) - np.asarray(b)
    if np.all(diffs == 0):
        return 1.0, "ns"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, p = wilcoxon(diffs)
    return p, sig_stars(p)


def wilcoxon_vs_zero(a):
    """
    One-sample Wilcoxon signed-rank test: is the distribution of `a` (ΔIoU)
    significantly different from zero?
    Returns (p, stars).
    """
    arr = np.asarray(a)
    if np.all(arr == 0):
        return 1.0, "ns"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, p = wilcoxon(arr)
    return p, sig_stars(p)


# ──────────────────────────────────────────────────────────────────────────────
# LOAD & ALIGN
# ──────────────────────────────────────────────────────────────────────────────

active_inputs = {k: v for k, v in INPUTS.items() if k != "DEPTH" or SHOW_DEPTH}
COLORS = {k: v for k, v in _ALL_COLORS.items() if k in active_inputs}

print("Loading data...")
raw = {label: load_iou(path) for label, path in active_inputs.items()}

common_ids = set(raw["RGB"].keys())
for d in raw.values():
    common_ids &= set(d.keys())

if not common_ids:
    raise ValueError("No images shared across all JSONs — check your paths.")

print(f"  {len(common_ids)} images shared across all models")

# Sort by RGB IoU high → low
ordered_ids = sorted(common_ids, key=lambda img: raw["RGB"][img], reverse=True)

# Raw aligned arrays
aligned_raw = {
    label: np.array([raw[label][img] for img in ordered_ids])
    for label in active_inputs
}

# In SHOW_DIFF mode: compute delta vs RGB for every non-RGB model
if SHOW_DIFF:
    rgb_arr = aligned_raw["RGB"]
    aligned = {
        label: arr - rgb_arr
        for label, arr in aligned_raw.items()
        if label != "RGB"          # RGB is the 0-reference; skip it
    }
    # Update active colours to match
    COLORS = {k: v for k, v in COLORS.items() if k in aligned}
else:
    aligned = aligned_raw

# ──────────────────────────────────────────────────────────────────────────────
# SPLIT INTO CHUNKS
# ──────────────────────────────────────────────────────────────────────────────

n = len(ordered_ids)
chunk_size = n // N_CHUNKS
chunk_slices = [
    slice(i * chunk_size, (i + 1) * chunk_size if i < N_CHUNKS - 1 else n)
    for i in range(N_CHUNKS)
]

if N_CHUNKS == 4:
    chunk_labels = _LABELS_4
elif N_CHUNKS == 3:
    chunk_labels = _LABELS_3
else:
    chunk_labels = [f"Chunk {i+1}" for i in range(N_CHUNKS)]

for label, sl in zip(chunk_labels, chunk_slices):
    lo = raw["RGB"][ordered_ids[sl.start]]
    hi = raw["RGB"][ordered_ids[sl.stop - 1]]
    print(f"  {label}: {sl.stop - sl.start} images  "
          f"(RGB IoU {lo:.3f} → {hi:.3f})")

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL X RANGE
# ──────────────────────────────────────────────────────────────────────────────

all_plot_vals = np.concatenate(list(aligned.values()))
if SHOW_DIFF:
    x_lo = np.percentile(all_plot_vals, 0.5) - 0.01
    x_hi = np.percentile(all_plot_vals, 99.9) + 0.02
    x_lo = min(x_lo, -0.01)   # always show the zero line
    x_hi = max(x_hi,  0.01)
else:
    x_lo = max(0.0, np.percentile(all_plot_vals, 0.5) - 0.01)
    x_hi = min(1.0, np.percentile(all_plot_vals, 99.9) + 0.01)

xs_kde = np.linspace(x_lo, x_hi, 400)

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE LAYOUT
# ──────────────────────────────────────────────────────────────────────────────

# Each panel is generous: 2.8 in wide, 4 in tall
panel_w = 2.8
fig_w = panel_w * N_CHUNKS + 0.6   # +0.6 for y-axis label room
fig_h = 4.8

fig, axes = plt.subplots(
    1, N_CHUNKS,
    figsize=(fig_w, fig_h),
    sharey=False,
)
# Leave room at top for title + legend, bottom for stats table
fig.subplots_adjust(
    left   = 0.07,
    right  = 0.96,
    top    = 0.78,
    bottom = 0.36,
    wspace = 0.45,
)

# ──────────────────────────────────────────────────────────────────────────────
# DRAW PANELS
# ──────────────────────────────────────────────────────────────────────────────

for col, (sl, panel_title) in enumerate(zip(chunk_slices, chunk_labels)):
    ax = axes[col]

    chunk_vals = {label: arr[sl] for label, arr in aligned.items()}
    chunk_n    = sl.stop - sl.start

    # Per-chunk x range: let each panel breathe a bit (still uses global KDE xs
    # for shape but clip display to per-chunk data range for tighter axes)
    local_all = np.concatenate(list(chunk_vals.values()))
    _lo = np.percentile(local_all, 0.1)
    _hi = np.percentile(local_all, 99.9)
    _pad = max((_hi - _lo) * 0.08, 0.01)   # 8% of local spread, min 0.01
    x_lo_local = _lo - _pad
    x_hi_local = _hi + _pad
    if SHOW_DIFF:
        x_lo_local = min(x_lo_local, -0.01)
        x_hi_local = max(x_hi_local,  0.01)

    # ── Zero reference line (SHOW_DIFF only) ─────────────────────────────────
    if SHOW_DIFF:
        ax.axvline(0, color="#888888", linewidth=0.9, linestyle="-",
                   alpha=0.7, zorder=1)

    # ── KDE curves ───────────────────────────────────────────────────────────
    for label, vals in chunk_vals.items():
        color = COLORS[label]
        arr   = np.asarray(vals)
        if len(arr) < 4:
            continue

        kde = gaussian_kde(arr, bw_method="silverman")
        ys  = kde(xs_kde)

        # Light fill under curve
        ax.fill_between(xs_kde, ys, alpha=_FILL_ALPHA, color=color)
        # Solid KDE line
        ax.plot(xs_kde, ys, color=color, linewidth=_LINE_WIDTH,
                label=label, zorder=3)

        # Dashed mean line
        mean_val = arr.mean()
        ax.axvline(mean_val, color=color, linewidth=1.0,
                   linestyle="--", alpha=0.9, zorder=4)

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_xlim(x_lo_local, x_hi_local)
    ax.set_ylim(bottom=0)

    xlabel = "ΔIoU vs RGB" if SHOW_DIFF else "IoU"
    ax.set_xlabel(xlabel, fontsize=9, labelpad=4)
    if col == 0:
        ax.set_ylabel("Density", fontsize=9, labelpad=4)
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)

    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4, prune="both"))
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    # Light horizontal grid only
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)

    # Panel title
    ax.set_title(panel_title, fontsize=9, fontweight="bold", pad=6)

    # Image-count badge (top-left inside panel)
    ax.text(0.03, 0.97, f"n\u202f=\u202f{chunk_n}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=7.5, color="#666666", style="italic")

    # ── Significance table ────────────────────────────────────────────────────
    labels_list = list(chunk_vals.keys())

    if SHOW_DIFF:
        # One-sample test: is ΔIoU significantly ≠ 0?
        rows = []
        for lbl in labels_list:
            p, stars = wilcoxon_vs_zero(chunk_vals[lbl])
            mu = np.mean(chunk_vals[lbl])
            rows.append((f"{lbl} ≠ 0", stars, p, mu))
    else:
        # Paired test between all model pairs
        rows = []
        for i in range(len(labels_list)):
            for j in range(i + 1, len(labels_list)):
                a_lbl, b_lbl = labels_list[i], labels_list[j]
                p, stars = wilcoxon_safe(chunk_vals[a_lbl], chunk_vals[b_lbl])
                mu = np.mean(chunk_vals[a_lbl]) - np.mean(chunk_vals[b_lbl])
                rows.append((f"{a_lbl} vs {b_lbl}", stars, p, mu))
                print(f"  [{panel_title}] {a_lbl} vs {b_lbl}: "
                      f"p={p:.4f} {stars}, Δμ={mu:+.4f}")

    # Build table text
    header      = f"{'Comparison':<22}  {'Sig':>3}   {'p':>6}   {'Δμ':>7}"
    separator   = "─" * len(header)
    table_lines = [header, separator]
    for comp, stars, p, mu in rows:
        table_lines.append(f"{comp:<22}  {stars:>3}   {p:>7.4f}   {mu:>+7.4f}")

    table_str = "\n".join(table_lines)

    ax.text(
        0.5, -0.26,
        table_str,
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=6.5,
        fontfamily="monospace",
        color="#1a1a1a",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#f8f8f8",
            edgecolor="#d0d0d0",
            linewidth=0.7,
        ),
    )

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE-LEVEL: LEGEND + TITLES
# ──────────────────────────────────────────────────────────────────────────────

# Build legend handles manually (one per model; avoids KDE/fill duplicates)
legend_handles = []
if SHOW_DIFF:
    for label in aligned:
        legend_handles.append(
            mlines.Line2D([], [], color=COLORS[label], linewidth=_LINE_WIDTH,
                          label=label)
        )
    # Add a reference element for the zero line
    legend_handles.append(
        mlines.Line2D([], [], color="#888888", linewidth=0.9, linestyle="-",
                      label="RGB baseline (0)")
    )
else:
    for label in aligned:
        legend_handles.append(
            mlines.Line2D([], [], color=COLORS[label], linewidth=_LINE_WIDTH,
                          label=label)
        )

# Mean-line legend entry (shared)
legend_handles.append(
    mlines.Line2D([], [], color="#555555", linewidth=1.0, linestyle="--",
                  label="Model mean")
)

fig.legend(
    handles        = legend_handles,
    loc            = "upper center",
    ncol           = len(legend_handles),
    fontsize       = 9,
    framealpha     = 0.95,
    edgecolor      = "#cccccc",
    bbox_to_anchor = (0.5, 0.97),
    handlelength   = 1.6,
    columnspacing  = 1.2,
)

# Main title
diff_suffix = " — ΔIoU vs RGB" if SHOW_DIFF else ""
fig.suptitle(
    TITLE + diff_suffix,
    fontsize   = 12,
    fontweight = "bold",
    y          = 1.03,
)

# Subtitle / caption
mode_note = (
    "ΔIoU = model IoU − RGB IoU per image.  Sig. test: one-sample Wilcoxon (vs 0)."
    if SHOW_DIFF else
    "Images sorted by RGB IoU (high → low) and split into equal-sized quartiles."
    "  Sig. test: paired Wilcoxon signed-rank."
)
sig_legend = "  Stars: *** p<0.001  ** p<0.01  * p<0.05  ns = not significant."

fig.text(
    0.5, 0.845,
    mode_note + sig_legend,
    ha      = "center",
    va      = "top",
    fontsize = 7,
    color   = "#555555",
    style   = "italic",
)

# ──────────────────────────────────────────────────────────────────────────────
# SAVE
# ──────────────────────────────────────────────────────────────────────────────

out_dir = Path(__file__).parent / "histogram_charts"
out_dir.mkdir(exist_ok=True)
safe_title = (TITLE + diff_suffix).replace(" ", "_").replace("/", "-").replace("—", "-")
out = out_dir / f"{safe_title}.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"\nSaved → {out}")
plt.show()
