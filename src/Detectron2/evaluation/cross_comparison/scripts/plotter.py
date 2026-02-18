import os
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict


# Constants
BASE_DATA_DIR = Path(
    "/projects/PUCHALLA/LLP2024/tumor-segmentation/data/processed_data"
)


class ComparisonPlotter:
    """Handles generation of comparison plots for failed images"""

    def __init__(self):
        """Initialize comparison plotter"""
        pass

    def plot_comparison(self, img_data: dict) -> plt.Figure:
        """
        Generate a comparison plot for a single image across all 3 models

        :param img_data: Image data dict with image_name, json_metrics, evaluation_results
        :type img_data: dict
        :return: Matplotlib figure or None if error
        :rtype: plt.Figure or None
        """
        try:
            image_name = img_data["image_name"]
            results = img_data["evaluation_results"]
            metrics = img_data["json_metrics"]

            # Create figure with 4 subplots (original + 3 model results)
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            fig.suptitle(f"Comparison: {image_name}", fontsize=16)

            # Plot original image
            original_path = os.path.join(
                BASE_DATA_DIR, "rgb", "test", "images", f"{image_name}"
            )
            if os.path.exists(original_path):
                original_img = cv2.imread(original_path)
                original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                axes[0].imshow(original_img)
                axes[0].set_title("Original Image")
                axes[0].axis("off")
            else:
                axes[0].text(
                    0.5,
                    0.5,
                    "Original\nNot Found",
                    ha="center",
                    va="center",
                    fontsize=12,
                )
                axes[0].axis("off")

            # Plot results from each model
            for idx, mode in enumerate(["rgb", "depth", "rgd"], start=1):
                if mode in results:
                    axes[idx].imshow(results[mode])
                    iou = metrics.get(mode, {}).get("mean_iou", 0.0)
                    axes[idx].set_title(f"{mode.upper()}\nIoU: {iou:.3f}")
                    axes[idx].axis("off")
                else:
                    axes[idx].text(
                        0.5, 0.5, f"{mode.upper()}\nN/A", ha="center", va="center"
                    )
                    axes[idx].axis("off")

            plt.tight_layout()
            return fig

        except Exception as e:
            print(f"  Error plotting {img_data.get('image_name', 'unknown')}: {e}")
            return None

    def generate_plots(self, failed_comparison: Dict, output_dir: str):
        """
        Generate and save comparison plots for failed images

        :param failed_comparison: Dictionary with categorized failed images
        :type failed_comparison: dict
        :param output_dir: Directory to save plots
        :type output_dir: str
        """
        plot_dir = os.path.join(output_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)

        # Plot failed_all_models
        if failed_comparison["failed_all_models"]:
            print(
                f"  Plotting {len(failed_comparison['failed_all_models'])} images that failed on all models..."
            )
            failed_all_dir = os.path.join(plot_dir, "failed_all_models")
            os.makedirs(failed_all_dir, exist_ok=True)

            for img_data in failed_comparison["failed_all_models"]:
                fig = self.plot_comparison(img_data)
                if fig:
                    output_path = os.path.join(
                        failed_all_dir, f"{img_data['image_name']}_comparison.png"
                    )
                    fig.savefig(output_path, bbox_inches="tight", dpi=150)
                    plt.close(fig)

        # Plot failed_two_models
        for category, images in failed_comparison["failed_two_models"].items():
            if images:
                print(f"  Plotting {len(images)} images that failed on {category}...")
                category_dir = os.path.join(plot_dir, f"failed_{category}")
                os.makedirs(category_dir, exist_ok=True)

                for img_data in images:
                    fig = self.plot_comparison(img_data)
                    if fig:
                        output_path = os.path.join(
                            category_dir, f"{img_data['image_name']}_comparison.png"
                        )
                        fig.savefig(output_path, bbox_inches="tight", dpi=150)
                        plt.close(fig)

        print(f"\nPlots saved to: {plot_dir}")

    def plot_results_legacy(self, results, image_name, image_id) -> plt.Figure:
        """
        Legacy method - now redirects to plot_comparison()
        Kept for backward compatibility

        :param results: Evaluation results dict
        :param image_name: Name of the image
        :param image_id: ID of the image
        :return: Matplotlib figure
        """
        # Create img_data dict in the format expected by plot_comparison
        img_data = {
            "image_name": image_name,
            "image_id": image_id,
            "evaluation_results": results,
            "json_metrics": {},  # Will be empty for this legacy method
        }

        return self.plot_comparison(img_data)
