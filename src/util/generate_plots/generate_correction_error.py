import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

TITLE = "Model Robustness: Baseline vs Augmented"  # used for graph title and output filename
METRIC = "mean_iou"  # "mean_iou" or "failure_rate"

INPUTS = {
    "RGB": {
        "baseline": Path("/projects/PUCHALLA/LLP2024/tumor-segmentation/src/util/failure_recreation_output/testing_output/rgb/baseline/RGB-2/json_RGB-2/iou_results.json"),
        "augmented": Path("/projects/PUCHALLA/LLP2024/tumor-segmentation/src/util/failure_recreation_output/testing_output/rgb/augmented/RGB-2/json_RGB-2/iou_results.json"),
    },
    "RGD": {
        "baseline": Path("/projects/PUCHALLA/LLP2024/tumor-segmentation/src/util/failure_recreation_output/testing_output/rgd/baseline/RGD-4/json_RGD-4/iou_results.json"),
        "augmented": Path("/projects/PUCHALLA/LLP2024/tumor-segmentation/src/util/failure_recreation_output/testing_output/rgd/augmented/RGD-4/json_RGD-4/iou_results.json"),
    }
}

COLORS = {"RGB": "#4E79A7", "RGD": "#B07AA1"}


def load_json_metrics(path: Path) -> dict:
    """Load iou_results.json and extract key metrics."""
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path) as f:
        data = json.load(f)

    # Handle two JSON formats:
    # Format 1: {"results": {"dataset_metrics": {...}, "per_image_iou": {...}}}
    # Format 2: {"dataset_metrics": {...}, "per_image_iou": {...}}

    if "results" in data and isinstance(data["results"], dict):
        # Format 1
        dataset_metrics = data["results"].get("dataset_metrics", {})
        per_image_iou = data["results"].get("per_image_iou", {})
    else:
        # Format 2
        dataset_metrics = data.get("dataset_metrics", {})
        per_image_iou = data.get("per_image_iou", {})

    mean_iou = dataset_metrics.get("mean_iou", 0.0)
    count_failed = dataset_metrics.get("count_failed", 0)
    total_count = len(per_image_iou)

    # Calculate failure rate (images below IoU threshold)
    failure_rate = count_failed / total_count if total_count > 0 else 0.0

    return {
        "mean_iou": mean_iou,
        "failure_rate": failure_rate,
        "count_failed": count_failed,
        "total_count": total_count,
    }


def calculate_correction_error(baseline_metrics: dict, augmented_metrics: dict, metric: str = "mean_iou") -> tuple:
    """
    Calculate correction error (CE) metric.

    CE = (1 - IoU_edge) / (1 - IoU_clean)

    Which simplifies to: error_corrupted / error_clean

    Returns tuple of (correction_error, baseline_error, augmented_error)
    Where CE > 1 means corruption degrades performance.
    """
    if metric == "mean_iou":
        # Error = 1 - mean_iou
        baseline_error = 1 - baseline_metrics["mean_iou"]
        augmented_error = 1 - augmented_metrics["mean_iou"]
    elif metric == "failure_rate":
        # Error = failure_rate
        baseline_error = baseline_metrics["failure_rate"]
        augmented_error = augmented_metrics["failure_rate"]
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Correction error: error_corrupted / error_clean
    if baseline_error == 0:
        # Edge case: no error on baseline, can't compute metric
        correction_error = 0.0
    else:
        correction_error = augmented_error / baseline_error

    return correction_error, baseline_error, augmented_error


# Load all metrics
data = {}
for model_name, paths in INPUTS.items():
    baseline_metrics = load_json_metrics(paths["baseline"])
    augmented_metrics = load_json_metrics(paths["augmented"])

    correction_error, baseline_error, augmented_error = calculate_correction_error(
        baseline_metrics, augmented_metrics, metric=METRIC
    )

    data[model_name] = {
        "baseline_error": baseline_error,
        "augmented_error": augmented_error,
        "correction_error": correction_error,
        "baseline_metrics": baseline_metrics,
        "augmented_metrics": augmented_metrics,
    }

    print(f"{model_name}:")
    print(f"  Baseline ({METRIC}): {baseline_metrics[METRIC]:.4f}")
    print(f"  Augmented ({METRIC}): {augmented_metrics[METRIC]:.4f}")
    print(f"  Baseline Error: {baseline_error:.4f}")
    print(f"  Augmented Error: {augmented_error:.4f}")
    print(f"  Correction Error (CE = error_aug / error_clean): {correction_error:.4f}")
    print()

# Create grouped bar chart
model_names = list(data.keys())
x_pos = np.arange(len(model_names))
width = 0.25  # width of each bar

fig, ax = plt.subplots(figsize=(10, 6))

baseline_errors = [data[m]["baseline_error"] for m in model_names]
augmented_errors = [data[m]["augmented_error"] for m in model_names]
correction_errors = [data[m]["correction_error"] for m in model_names]

# Plot grouped bars
bars1 = ax.bar(x_pos - width, baseline_errors, width, label="Baseline Error", alpha=0.8, color="#CCCCCC")
bars2 = ax.bar(x_pos, augmented_errors, width, label="Augmented Error", alpha=0.8, color="#888888")
bars3_colors = [COLORS[m] for m in model_names]
bars3 = ax.bar(x_pos + width, correction_errors, width, label="Correction Error", alpha=0.8, color=bars3_colors)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

# Styling
ax.set_xlabel("Model", fontsize=11)
ax.set_ylabel("Error Ratio", fontsize=11)
ax.set_title(f"{TITLE} — CE = (1 - IoU_edge) / (1 - IoU_clean)", fontsize=13, pad=10)
ax.set_xticks(x_pos)
ax.set_xticklabels(model_names)
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Add horizontal line at y=1 for reference (CE=1 means no degradation)
ax.axhline(1, color="black", linewidth=0.8, linestyle="--", alpha=0.5, label="No degradation (CE=1)")

fig.tight_layout()

# Save output
out_dir = Path(__file__).parent / "correction_error_charts"
out_dir.mkdir(exist_ok=True)
safe_title = TITLE.replace(" ", "_").replace("/", "-")
out = out_dir / f"{safe_title}_{METRIC}.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.show()
