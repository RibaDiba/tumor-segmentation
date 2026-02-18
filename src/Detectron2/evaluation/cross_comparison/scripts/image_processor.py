from typing import List, Dict, Callable
from .json_utils import JSONHandler


class ImageProcessor:
    """Processes and filters failed images"""

    def __init__(self, json_handler: JSONHandler, model_names: dict):
        """
        Initialize image processor

        :param json_handler: JSONHandler instance for loading data
        :type json_handler: JSONHandler
        :param model_names: Dictionary mapping mode to model name
        :type model_names: dict
        """
        self.json_handler = json_handler
        self.model_names = model_names
        self.results = []

    def process_failed_images(
        self, image_names: set, evaluate_callback: Callable
    ):
        """
        For each failed image, gather JSON metrics and re-run evaluation
        Stores results in self.results

        :param image_names: Set of image names to process
        :type image_names: set
        :param evaluate_callback: Function to evaluate an image (image_name, image_id) -> results
        :type evaluate_callback: Callable
        """
        for image_name in image_names:
            # Get metrics from JSON for each model
            json_metrics = {}
            image_id = None

            for mode, model_name in self.model_names.items():
                metrics = self.json_handler.get_image_metrics(model_name, image_name)

                if metrics:
                    json_metrics[mode] = metrics
                    if image_id is None:
                        image_id = metrics["id"]

            # Re-run evaluation to get visual outputs
            print(f"  Evaluating {image_name}...")
            evaluation_results = evaluate_callback(image_name, image_id)

            # Store in results for later filtering
            self.results.append(
                {
                    "image_name": image_name,
                    "image_id": image_id,
                    "json_metrics": json_metrics,
                    "evaluation_results": evaluation_results,
                }
            )

    def filter_best_performers(self) -> Dict[str, List[Dict]]:
        """
        Filter failed images by which model performed best (highest IoU)
        among the failures

        Categories:
        - best_rgb: Failed images where RGB had the highest IoU
        - best_depth: Failed images where Depth had the highest IoU
        - best_rgd: Failed images where RGD had the highest IoU
        - failed_all: All images that failed on all 3 models

        :return: Dictionary with categorized failed images
        :rtype: dict
        """
        best_rgb = []
        best_depth = []
        best_rgd = []
        failed_all = []

        for img_data in self.results:
            metrics = img_data["json_metrics"]

            # Get IoU scores for this failed image from all 3 models
            ious = {
                "rgb": metrics.get("rgb", {}).get("mean_iou", 0.0),
                "depth": metrics.get("depth", {}).get("mean_iou", 0.0),
                "rgd": metrics.get("rgd", {}).get("mean_iou", 0.0),
            }

            # Determine which model performed best
            best_model = max(ious, key=ious.get)

            # If the image failed on all 3 models (all have metrics)
            if all(mode in metrics for mode in ["rgb", "depth", "rgd"]):
                failed_all.append(img_data)

            # Categorize by best performer
            if best_model == "rgb":
                best_rgb.append(img_data)
            elif best_model == "depth":
                best_depth.append(img_data)
            elif best_model == "rgd":
                best_rgd.append(img_data)

        return {
            "best_rgb": best_rgb,
            "best_depth": best_depth,
            "best_rgd": best_rgd,
            "failed_all": failed_all,
        }

    def clear_results(self):
        """Clear stored results"""
        self.results = []
