import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

TITLE      = "BW RGD Shadows (No Depth)"  # used for graph title and output filename
SHOW_DEPTH = False  # set to False to hide the DEPTH model
SHOW_DIFF  = True  # set to True to plot per-image IoU difference vs RGB
    
INPUTS = {
    "RGB":   Path("/projects/PUCHALLA/LLP2024/tumor-segmentation/testing_shadows_only/rgb/augmented/RGB-2/json_RGB-2/iou_results.json"),
    "DEPTH": Path("/projects/PUCHALLA/LLP2024/tumor-segmentation/testing_shadows_only/depth/baseline/DEPTH-5/json_DEPTH-5/iou_results.json"),
    "RGD":   Path("/projects/PUCHALLA/LLP2024/tumor-segmentation/testing_shadows_only/rgd/augmented/RGD-4/json_RGD-4/iou_results.json"),
}

COLORS = {"RGB": "#4E79A7", "DEPTH": "#F28E2B", "RGD": "#B07AA1"}


def load_iou(path: Path):
    with open(path) as f:
        data = json.load(f)
    # keyed by image_id -> mean_iou
    return {v["id"]: v["mean_iou"] for v in data["per_image_iou"].values()}


# Load all models as {image_id: iou}
data = {label: load_iou(path) for label, path in INPUTS.items() if label != "DEPTH" or SHOW_DEPTH}

# Determine x-order: sort image IDs by RGB IoU, highest → lowest
ordered_ids = sorted(data["RGB"], key=lambda img_id: data["RGB"][img_id], reverse=True)
x = range(1, len(ordered_ids) + 1)

ZORDERS = {"DEPTH": 2, "RGD": 3, "RGB": 4}
WINDOW = 10

fig, ax = plt.subplots(figsize=(16, 5))

rgb_ious = [data["RGB"][img_id] for img_id in ordered_ids]

all_ious = []
for label in data:
    raw_ious = [data[label][img_id] for img_id in ordered_ids]
    ious = [a - b for a, b in zip(raw_ious, rgb_ious)] if SHOW_DIFF else raw_ious
    all_ious.extend(ious)
    color = COLORS[label]
    z = ZORDERS[label]
    # Raw line: thin + semi-transparent
    ax.plot(x, ious, color=color, linewidth=0.8, alpha=0.25, zorder=z)
    # Smoothed trend
    smoothed = np.convolve(ious, np.ones(WINDOW) / WINDOW, mode="same")
    ax.plot(x, smoothed, label=label, color=color, linewidth=2.2, zorder=z + 0.5)
    # Mean line with value annotation
    mean_val = sum(ious) / len(ious)
    ax.axhline(mean_val, color=color, linestyle="--", linewidth=1.0, alpha=0.7, zorder=1)
    label_str = f"Δμ={mean_val:+.3f}" if SHOW_DIFF else f"μ={mean_val:.3f}"
    ax.text(len(x) + 1, mean_val, label_str, color=color, va="center", fontsize=8)

if SHOW_DIFF:
    ax.axhline(0, color="gray", linewidth=1.0, linestyle="-", alpha=0.5, zorder=0)

y_min = np.percentile(all_ious, 2)
y_max = max(all_ious)
pad = (y_max - y_min) * 0.05
ax.set_ylim(y_min - pad, y_max + pad)

ax.set_xlabel("Images (sorted by RGB IoU, high → low)", fontsize=11)
ax.set_ylabel("ΔIoU vs RGB" if SHOW_DIFF else "IoU", fontsize=11)
title_suffix = " — Difference vs RGB" if SHOW_DIFF else ""
ax.set_title(f"Per-Image IoU{title_suffix} — {TITLE}", fontsize=13, pad=10)
ax.legend(fontsize=10, framealpha=0.9)
step = max(1, len(ordered_ids) // 10)
ax.set_xticks([i for i in x if (i - 1) % step == 0])
ax.set_xticklabels([str(i) for i in x if (i - 1) % step == 0], fontsize=9)
ax.tick_params(axis="x", rotation=0)
ax.tick_params(axis="y", labelsize=9)
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.subplots_adjust(right=0.88)
out_dir = Path(__file__).parent / "line_graphs"
out_dir.mkdir(exist_ok=True)
safe_title = TITLE.replace(" ", "_").replace("/", "-")
out = out_dir / f"{safe_title}.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.show()
