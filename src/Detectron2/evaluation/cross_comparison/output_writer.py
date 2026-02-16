import os
import json
from typing import Dict


class OutputWriter:
    """Handles writing comparison results to JSON files"""

    def __init__(self, model_names: dict):
        """
        Initialize output writer

        :param model_names: Dictionary mapping mode to model name
        :type model_names: dict
        """
        self.model_names = model_names

    def save_results(
        self,
        failed_comparison: Dict,
        output_dir: str,
    ) -> Dict:
        """
        Saves comparison results in the format defined in model.json:
        - model_names: The 3 model names used
        - best_rgb: Images where RGB had the highest IoU
        - best_depth: Images where Depth had the highest IoU
        - best_rgd: Images where RGD had the highest IoU
        - failed_all: Images that failed on all models

        :param failed_comparison: Dictionary with categorized failed images
        :type failed_comparison: dict
        :param output_dir: Directory to save JSON file
        :type output_dir: str
        :return: Output data dictionary
        :rtype: dict
        """
        print("\nCreating JSON output from failed images data...")

        # Use filtered failed images data (already categorized by best performer)
        best_rgb = {}
        best_depth = {}
        best_rgd = {}
        failed_all = {}

        # Process best_rgb failures
        for img_data in failed_comparison.get("best_rgb", []):
            image_name = img_data["image_name"]
            metrics = img_data["json_metrics"]

            best_rgb[image_name] = {
                "IOU_rgb": round(metrics.get("rgb", {}).get("mean_iou", 0.0), 4),
                "IOU_depth": round(
                    metrics.get("depth", {}).get("mean_iou", 0.0), 4
                ),
                "IOU_rgd": round(metrics.get("rgd", {}).get("mean_iou", 0.0), 4),
            }

        # Process best_depth failures
        for img_data in failed_comparison.get("best_depth", []):
            image_name = img_data["image_name"]
            metrics = img_data["json_metrics"]

            best_depth[image_name] = {
                "IOU_rgb": round(metrics.get("rgb", {}).get("mean_iou", 0.0), 4),
                "IOU_depth": round(
                    metrics.get("depth", {}).get("mean_iou", 0.0), 4
                ),
                "IOU_rgd": round(metrics.get("rgd", {}).get("mean_iou", 0.0), 4),
            }

        # Process best_rgd failures
        for img_data in failed_comparison.get("best_rgd", []):
            image_name = img_data["image_name"]
            metrics = img_data["json_metrics"]

            best_rgd[image_name] = {
                "IOU_rgb": round(metrics.get("rgb", {}).get("mean_iou", 0.0), 4),
                "IOU_depth": round(
                    metrics.get("depth", {}).get("mean_iou", 0.0), 4
                ),
                "IOU_rgd": round(metrics.get("rgd", {}).get("mean_iou", 0.0), 4),
            }

        # Process failed_all
        for img_data in failed_comparison.get("failed_all_models", []):
            image_name = img_data["image_name"]
            metrics = img_data["json_metrics"]

            failed_all[image_name] = {
                "IOU_rgb": round(metrics.get("rgb", {}).get("mean_iou", 0.0), 4),
                "IOU_depth": round(
                    metrics.get("depth", {}).get("mean_iou", 0.0), 4
                ),
                "IOU_rgd": round(metrics.get("rgd", {}).get("mean_iou", 0.0), 4),
            }

        # Build output structure matching model.json
        output_data = {
            "model_names": self.model_names,
            "best_rgb": best_rgb,
            "best_depth": best_depth,
            "best_rgd": best_rgd,
            "failed_all": failed_all,
        }

        # Save JSON file
        json_output_path = os.path.join(output_dir, "comparison_results.json")
        with open(json_output_path, "w") as f:
            json.dump(output_data, f, indent=4)

        print(f"\nResults saved to: {json_output_path}")
        print(f"  Best RGB performers: {len(best_rgb)}")
        print(f"  Best Depth performers: {len(best_depth)}")
        print(f"  Best RGD performers: {len(best_rgd)}")
        print(f"  Failed on all models: {len(failed_all)}")

        return output_data
