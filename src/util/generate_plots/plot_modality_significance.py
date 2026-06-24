#!/usr/bin/env python3
"""
plot_modality_significance.py
=============================
Standard, paper-ready figures comparing input modalities (e.g. RGB / RGD / Depth / RGBD)
on a PAIRED, per-image IoU metric, with significance testing.

Produces two conventional figure types:
  * BAR  -- mean IoU per modality with 95% confidence-interval error bars and
            pairwise significance stars (the usual "X is significantly better" plot).
  * BOX  -- box-and-whisker of the per-image IoU distribution per modality with
            pairwise significance brackets.

Why paired tests: the same evaluation images are scored under every modality, so
observations are matched. Pairwise paired Wilcoxon signed-rank tests (robust to
the non-normal, bounded IoU differences) are Holm-corrected across all pairs.
Aggregate AP (one number per modality) has no distribution and cannot be tested.

Input format (one file per modality, matched by filename):
    {"per_image_iou": {"<img>": {"mean_iou": float, ...}, ...}}
    or:
    {"results": {"per_image_iou": {"<img>": {"mean_iou": float, ...}, ...}}}

Usage:
    # 1. Configure the 4 variables at the top of this script and run without arguments:
    python plot_modality_significance.py
    
    # 2. Or pass files directly:
    python plot_modality_significance.py --files iou_RGB.json iou_RGD.json iou_Depth.json iou_RGBD.json
    
    # 3. Or pass an input directory containing the IoU json files:
    python plot_modality_significance.py --input-dir models/ --kind both
"""
import argparse
import glob
import json
import os
import re
import sys
import itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# Centralized path resolution matching other scripts in this directory
_util_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _util_dir not in sys.path:
    sys.path.insert(0, _util_dir)
from paths import PROJECT_ROOT

# ==============================================================================
# MODALITY PATHS CONFIGURATION
# Point these variables to the paths of your respective IoU JSON files.
# E.g., PROJECT_ROOT / "models/rgb/RGB-2/latest/IoU_AP_Final/json_RGB-2/iou_results.json"
# If these variables are set, running the script without arguments will use them.
# ==============================================================================
RGBD_IOU_PATH = ""  # Path to RGBD or RBGD IoU json file
RGB_IOU_PATH = ""   # Path to RGB IoU json file
DEPTH_IOU_PATH = "" # Path to Depth IoU json file
RGD_IOU_PATH = ""   # Path to RGD IoU json file

# Palette
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]

# Modality mapping for the configured variables if we cannot auto-detect from the path
VARIABLE_MODALITIES = {
    RGBD_IOU_PATH: "RGBD",
    RGB_IOU_PATH: "RGB",
    DEPTH_IOU_PATH: "Depth",
    RGD_IOU_PATH: "RGD"
}

def set_pub_style():
    plt.rcParams.update({
        "savefig.dpi": 300, "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11.5,
        "axes.linewidth": 0.9, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    })

# ---------- data ----------
def modality_from_path(p):
    stem = os.path.splitext(os.path.basename(p))[0]
    stem_lower = stem.lower()
    if "rgbd" in stem_lower:
        return "RGBD"
    elif "rbgd" in stem_lower:
        return "RBGD"
    elif "rgd" in stem_lower:
        return "RGD"
    elif "rgb" in stem_lower:
        return "RGB"
    elif "depth" in stem_lower:
        return "Depth"
    
    m = re.search(r"(RGBD[\w-]*|RBGD[\w-]*|RGD[\w-]*|RGB[\w-]*|Depth[\w-]*|DEPTH[\w-]*|[A-Za-z0-9-]+)$", stem, re.IGNORECASE)
    if m:
        val = m.group(1)
        val_lower = val.lower()
        if "rgbd" in val_lower:
            return "RGBD"
        if "rbgd" in val_lower:
            return "RBGD"
        if "rgd" in val_lower:
            return "RGD"
        if "rgb" in val_lower:
            return "RGB"
        if "depth" in val_lower:
            return "Depth"
        return val
    return stem

def get_label_for_configured_path(path):
    detected = modality_from_path(path)
    # If the filename is generic (like "iou_results"), fallback to variable name mapping
    if detected.lower() in ("iou_results", "iou_results_"):
        resolved_path = str(Path(path).resolve())
        for k, v in VARIABLE_MODALITIES.items():
            if k and str(Path(k).resolve()) == resolved_path:
                if "rbgd" in resolved_path.lower():
                    return "RBGD"
                return v
        return "Unknown"
    return detected

def strip_aug(n): return re.sub(r"_aug_[hvr]+(?=\.)", "", n)

def load(path):
    if not os.path.exists(path):
        sys.exit(f"ERROR: File not found: {path}")
    with open(path) as f:
        blob = json.load(f)
    
    # Handle two JSON formats in the repo
    if "results" in blob and isinstance(blob["results"], dict):
        results_section = blob["results"]
    else:
        results_section = blob

    if "per_image_iou" not in results_section:
        sys.exit(f"ERROR: {path} lacks 'per_image_iou'. Aggregate scores cannot "
                 "be significance-tested (no distribution).")
    
    per_image = results_section["per_image_iou"]
    out = {}
    for fn, r in per_image.items():
        if isinstance(r, dict):
            if "mean_iou" in r:
                out[fn] = r["mean_iou"]
            elif "iou" in r:
                out[fn] = r["iou"]
            else:
                sys.exit(f"ERROR: Unexpected format in 'per_image_iou' of {path}: {r}")
        elif isinstance(r, (int, float)):
            out[fn] = float(r)
        else:
            sys.exit(f"ERROR: Unexpected format in 'per_image_iou' of {path}: {r}")
    return out

def build(files, collapse, order, use_variables=False):
    if use_variables:
        raw = {get_label_for_configured_path(p): load(p) for p in files}
    else:
        raw = {modality_from_path(p): load(p) for p in files}
        
    if collapse:
        out = {}
        for lab, d in raw.items():
            g = {}
            for fn, v in d.items(): g.setdefault(strip_aug(fn), []).append(v)
            out[lab] = {b: float(np.mean(vs)) for b, vs in g.items()}
        raw = out
        
    common = sorted(set.intersection(*(set(d) for d in raw.values())))
    if not common: sys.exit("ERROR: no shared image names across modalities.")
    arr = {lab: np.array([raw[lab][k] for k in common]) for lab in raw}
    
    # Default visual ordering: RGBD/RBGD, RGB, Depth, RGD
    DEFAULT_ORDER = ["RGBD", "RBGD", "RGB", "Depth", "RGD"]
    if order:
        labels = order
    else:
        known_labels = [l for l in DEFAULT_ORDER if l in arr]
        other_labels = [l for l in arr if l not in DEFAULT_ORDER]
        labels = known_labels + other_labels
        
    missing = [l for l in labels if l not in arr]
    if missing: sys.exit(f"ERROR: requested modalities not found: {missing}")
    return labels, arr, len(common)

# ---------- stats ----------
def stars(p): return "***" if p<1e-3 else "**" if p<1e-2 else "*" if p<5e-2 else "n.s."

def holm(ps):
    ps = np.asarray(ps, float); order = np.argsort(ps); m = len(ps)
    adj = np.empty(m); run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, (m-rank)*ps[idx]); adj[idx] = min(run, 1.0)
    return adj

def ci95(x):
    n = len(x); se = x.std(ddof=1)/np.sqrt(n)
    h = stats.t.ppf(0.975, n-1)*se
    return x.mean(), h

def pairwise(labels, arr, test):
    pairs = list(itertools.combinations(labels, 2))
    res = {}
    raw = []
    for a, b in pairs:
        d = arr[a]-arr[b]
        if test == "ttest":
            p = stats.ttest_rel(arr[a], arr[b]).pvalue
        else:
            p = 1.0 if np.allclose(d, 0) else stats.wilcoxon(arr[a], arr[b]).pvalue
        sd = d.std(ddof=1); dz = d.mean()/sd if sd>0 else 0.0
        res[(a, b)] = dict(p_raw=p, dz=dz, mean_diff=d.mean()); raw.append(p)
    for (pr, a) in zip(pairs, holm(raw)): res[pr]["p_adj"] = a
    return pairs, res

# ---------- brackets ----------
def bracket(ax, x1, x2, y, h, text, lw=1.0, fs=10):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color="black", lw=lw)
    ax.text((x1+x2)/2, y+h, text, ha="center", va="bottom", fontsize=fs)

# ---------- figures ----------
def fig_bar(labels, arr, pairs, res, n, metric, out):
    set_pub_style()
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    x = np.arange(len(labels))
    means = [arr[l].mean() for l in labels]
    errs = [ci95(arr[l])[1] for l in labels]
    colors = [OKABE_ITO[i % len(OKABE_ITO)] for i in range(len(labels))]
    ax.bar(x, means, width=0.62, color=colors, edgecolor="black", linewidth=0.8,
           yerr=errs, capsize=4, error_kw=dict(elinewidth=1.1, ecolor="black"))
    for xi, m in zip(x, means):
        ax.text(xi, 0.02, f"{m:.3f}", ha="center", va="bottom",
                fontsize=9.5, color="white", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel(f"Mean {metric}  (95% CI)"); ax.set_xlabel("Input modality")
    top = max(means)+max(errs)
    ax.set_ylim(0, max(1.0, top*1.05))
    
    # significance brackets, stacked
    y0 = top + 0.02; step = 0.05; level = 0
    for (a, b) in pairs:
        if res[(a, b)]["p_adj"] >= 0.05: continue
        x1, x2 = labels.index(a), labels.index(b)
        bracket(ax, x1, x2, y0+level*step, step*0.35, stars(res[(a, b)]["p_adj"]))
        level += 1
    ax.set_ylim(0, max(ax.get_ylim()[1], y0+level*step+0.04))
    ax.set_title(f"{metric} by modality", loc="left")
    fig.text(0.5, -0.02, f"Paired Wilcoxon, Holm-corrected, n = {n} matched "
             "images.  *** p<.001  ** p<.01  * p<.05", ha="center",
             fontsize=7.8, color="0.35")
    fig.savefig(f"{out}_bar.pdf", bbox_inches="tight")
    fig.savefig(f"{out}_bar.png", bbox_inches="tight")
    print(f"Wrote {out}_bar.pdf / .png")
    plt.close(fig)

def fig_box(labels, arr, pairs, res, n, metric, out):
    set_pub_style()
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    data = [arr[l] for l in labels]
    pos = np.arange(len(labels))+1
    colors = [OKABE_ITO[i % len(OKABE_ITO)] for i in range(len(labels))]
    means = np.array([d.mean() for d in data])
    best_idx = int(np.argmax(means))
    best_label = labels[best_idx]
    best_mean = means[best_idx]
    bp = ax.boxplot(data, positions=pos, widths=0.55, patch_artist=True,
                    showfliers=True,
                    flierprops=dict(marker="o", markersize=2.5,
                                    markerfacecolor="0.5", markeredgecolor="none", alpha=0.4),
                    medianprops=dict(color="black", lw=1.4),
                    whiskerprops=dict(color="0.3", lw=1.0),
                    capprops=dict(color="0.3", lw=1.0),
                    boxprops=dict(lw=0.9, edgecolor="black"))
    for i, (patch, c) in enumerate(zip(bp["boxes"], colors)):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
        if i == best_idx:
            patch.set_alpha(0.78)
            patch.set_linewidth(2.0)
    ax.scatter(pos, means, marker="D", s=34, color="black",
               edgecolors="white", linewidths=0.8, zorder=4)
    xtick_labels = [f"{lab}\nμ={m:.3f}" for lab, m in zip(labels, means)]
    ax.set_xticks(pos); ax.set_xticklabels(xtick_labels)
    for i, tick in enumerate(ax.get_xticklabels()):
        if i == best_idx:
            tick.set_fontweight("bold")
    ax.set_ylabel(f"Per-image {metric}"); ax.set_xlabel("Input modality")
    ax.grid(axis="y", color="0.88", linestyle="-", linewidth=0.8)
    ymax = max(np.percentile(d, 95) for d in data)
    ymin = min(np.percentile(d, 2) for d in data)
    span = max(ymax - ymin, 1e-6)
    y0 = ymax + span*0.07
    step = span*0.11
    level = 0
    for other in labels:
        if other == best_label:
            continue
        pair = (best_label, other) if (best_label, other) in res else (other, best_label)
        if res[pair]["p_adj"] >= 0.05:
            continue
        x1, x2 = pos[labels.index(best_label)], pos[labels.index(other)]
        bracket(ax, min(x1, x2), max(x1, x2), y0+level*step, step*0.30,
                stars(res[pair]["p_adj"]), fs=9.5)
        level += 1
    upper_from_brackets = y0 + (level * step) + span*0.08
    upper_from_note = ymax + span*0.30
    ax.set_ylim(ymin - span*0.06, max(upper_from_brackets, upper_from_note))
    ax.text(0.02, 0.985, f"Best mean {metric}: {best_label} (μ={best_mean:.3f})",
            transform=ax.transAxes, ha="left", va="top", fontsize=9.4, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.75", alpha=0.95))
    ax.set_title(f"Per-image {metric} distribution", loc="left")
    fig.text(0.5, -0.02, f"Mean shown as ◆. Best modality is highlighted. "
             f"Paired Wilcoxon (Holm), n = {n}.  *** p<.001  ** p<.01  * p<.05", ha="center",
             fontsize=7.8, color="0.35")
    fig.savefig(f"{out}_box.pdf", bbox_inches="tight")
    fig.savefig(f"{out}_box.png", bbox_inches="tight")
    print(f"Wrote {out}_box.pdf / .png")
    plt.close(fig)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    g = ap.add_mutually_exclusive_group(required=False)
    g.add_argument("--files", nargs="+")
    g.add_argument("--input-dir")
    ap.add_argument("--pattern", default="*iou_results*.json")
    ap.add_argument("--order", nargs="+", help="explicit modality order on the x-axis")
    ap.add_argument("--kind", choices=["bar", "box", "both"], default="both")
    ap.add_argument("--test", choices=["wilcoxon", "ttest"], default="wilcoxon")
    ap.add_argument("--collapse-aug", action="store_true",
                    help="average _aug_* copies into base images (independent units)")
    ap.add_argument("--metric-name", default="IoU")
    ap.add_argument("--out", default="fig_modality")
    a = ap.parse_args()

    files = a.files
    use_variables = False
    
    if not files and not a.input_dir:
        # Check if variables are configured
        configured_paths = [RGBD_IOU_PATH, RGB_IOU_PATH, DEPTH_IOU_PATH, RGD_IOU_PATH]
        valid_paths = [p for p in configured_paths if p]
        
        if len(valid_paths) == 4:
            missing_paths = [p for p in valid_paths if not os.path.exists(p)]
            if missing_paths:
                sys.exit(f"ERROR: Some configured paths do not exist:\n" + 
                         "\n".join(f"  - {p}" for p in missing_paths))
            files = valid_paths
            use_variables = True
        elif len(valid_paths) > 0:
            sys.exit(f"ERROR: Only {len(valid_paths)} of 4 paths are configured in script variables. "
                     f"Please configure all 4, or use --files/--input-dir.")
        else:
            # Try to auto-discover under models/
            models_dir = PROJECT_ROOT / "models"
            discovered = {}
            if models_dir.exists():
                for path in models_dir.glob("**/iou_results.json"):
                    path_str = str(path)
                    mod = modality_from_path(path_str)
                    if mod not in ["RGBD", "RBGD", "RGB", "Depth", "RGD"]:
                        continue
                    if mod not in discovered or "latest" in path_str:
                        discovered[mod] = path_str
            
            has_rgbd = "RGBD" in discovered or "RBGD" in discovered
            has_rgb = "RGB" in discovered
            has_depth = "Depth" in discovered
            has_rgd = "RGD" in discovered
            
            if has_rgbd and has_rgb and has_depth and has_rgd:
                files = []
                for mod in ["RGBD", "RBGD", "RGB", "Depth", "RGD"]:
                    if mod in discovered:
                        files.append(discovered[mod])
                print(f"Auto-discovered 4 IoU result files:")
                for f in files:
                    print(f"  - {f}")
            else:
                sys.exit(
                    "ERROR: No inputs provided. You must either:\n"
                    "  1. Pass --files <file1> <file2> <file3> <file4>\n"
                    "  2. Pass --input-dir <dir>\n"
                    "  3. Configure the 4 variables at the top of the script:\n"
                    "     RGBD_IOU_PATH, RGB_IOU_PATH, DEPTH_IOU_PATH, RGD_IOU_PATH\n\n"
                    "Auto-discovery under models/ failed. Found modalities: " + 
                    (", ".join(discovered.keys()) if discovered else "none")
                )
    
    elif a.input_dir:
        files = sorted(glob.glob(os.path.join(a.input_dir, a.pattern)))
        if not files:
            files = sorted(glob.glob(os.path.join(a.input_dir, "**", a.pattern), recursive=True))
            if not files:
                sys.exit(f"ERROR: No files matching '{a.pattern}' found in '{a.input_dir}'.")
                
    labels, arr, n = build(files, a.collapse_aug, a.order, use_variables)
    pairs, res = pairwise(labels, arr, a.test)

    print(f"\nn={n}  test={a.test}  collapse_aug={a.collapse_aug}")
    for l in labels:
        m, h = ci95(arr[l]); print(f"  {l:<10} mean={m:.4f}  95%CI=±{h:.4f}")
    print("  pairwise (Holm):")
    for (x, y) in pairs:
        r = res[(x, y)]
        print(f"    {x} vs {y}: Δ={r['mean_diff']:+.4f}  p={r['p_adj']:.2e} "
              f"{stars(r['p_adj'])}  dz={r['dz']:+.2f}")

    # Determine output folder
    default_out_dir = Path(__file__).parent / "significance_charts"
    out_path = Path(a.out)
    if not out_path.is_absolute() and len(out_path.parts) == 1:
        default_out_dir.mkdir(exist_ok=True)
        out_prefix = str(default_out_dir / out_path.name)
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_prefix = str(out_path)

    if a.kind in ("bar", "both"): fig_bar(labels, arr, pairs, res, n, a.metric_name, out_prefix)
    if a.kind in ("box", "both"): fig_box(labels, arr, pairs, res, n, a.metric_name, out_prefix)

if __name__ == "__main__":
    main()
