import json
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parents[2] / "src/Detectron2/slurm_output/AUG_1000"

MODELS = {
    "RGB":   "RGB-1",
    "DEPTH": "DEPTH-1",
    "RGD":   "RGD-1",
}

COLORS = {"RGB": "#4E79A7", "DEPTH": "#F28E2B", "RGD": "#B07AA1"}


def load_iou(model_dir: str):
    path = BASE / model_dir / f"IoU_AP_Final/json_{model_dir}/iou_results.json"
    with open(path) as f:
        data = json.load(f)
    # keyed by image_id -> mean_iou
    return {v["id"]: v["mean_iou"] for v in data["per_image_iou"].values()}


# Load all models as {image_id: iou}
data = {label: load_iou(model_dir) for label, model_dir in MODELS.items()}

# Determine x-order: sort image IDs by RGB IoU, highest → lowest
ordered_ids = sorted(data["RGB"], key=lambda img_id: data["RGB"][img_id], reverse=True)
x = range(1, len(ordered_ids) + 1)

fig, ax = plt.subplots(figsize=(16, 5))

all_ious = []
for label in MODELS:
    ious = [data[label][img_id] for img_id in ordered_ids]
    all_ious.extend(ious)
    ax.plot(x, ious, label=label, color=COLORS[label], linewidth=1.8,
            marker="o", markersize=4, zorder=3)
    ax.axhline(sum(ious) / len(ious), color=COLORS[label], linestyle="--",
               linewidth=1.0, alpha=0.6)

y_min = min(all_ious)
y_max = max(all_ious)
pad = (y_max - y_min) * 0.15
ax.set_ylim(y_min - pad, y_max + pad)

ax.set_xlabel("Images (sorted by RGB IoU, high → low)", fontsize=11)
ax.set_ylabel("IoU", fontsize=11)
ax.set_title("Per-Image IoU — R-101 FPN Models (test set)", fontsize=13, pad=10)
ax.legend(fontsize=10, framealpha=0.9)
ax.set_xticks(list(x))
ax.set_xticklabels([str(img_id) for img_id in ordered_ids])
ax.tick_params(axis="x", labelsize=7, rotation=45)
ax.tick_params(axis="y", labelsize=9)
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

out = Path(__file__).parent / "iou_lineplot_101.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.show()
