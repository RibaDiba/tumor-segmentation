import os
import json
from typing import List, Dict


class JSONHandler:
    """Handles loading and extracting data from JSON IoU result files"""

    def __init__(
        self,
        json_root: str = "/projects/PUCHALLA/LLP2024/tumor-segmentation/src/Detectron2/slurm_output/IoU_fig/json",
    ):
        """
        Initialize JSON handler

        :param json_root: Root directory containing JSON IoU result files
        :type json_root: str
        """
        self.json_root = json_root
        self._json_cache = {}

    def load_json_file(self, model_name: str) -> dict:
        """
        Load JSON IoU results for a model

        :param model_name: Name of the model (e.g., "rgb-7030-4")
        :type model_name: str
        :return: Parsed JSON data
        :rtype: dict
        """
        if model_name in self._json_cache:
            return self._json_cache[model_name]

        json_path = os.path.join(self.json_root, f"{model_name}_json_results.json")

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        with open(json_path, "r") as f:
            data = json.load(f)

        self._json_cache[model_name] = data
        return data

    def get_failed_images(self, model_name: str) -> List[str]:
        """
        Reads model's JSON data and extracts images with IoU < 0.50

        :param model_name: Name of the model (e.g., "rgb-7030-4")
        :type model_name: str
        :return: List of image names that failed
        :rtype: List[str]
        """
        json_data = self.load_json_file(model_name)

        # Extract failed images from pre-computed section
        failed_dict = json_data["results"]["per_image_failed"]

        # Return just the image names (keys)
        return list(failed_dict.keys())

    def get_image_metrics(
        self, model_name: str, image_name: str
    ) -> Dict[str, any]:
        """
        Get metrics for a specific image from a model's results

        :param model_name: Name of the model
        :type model_name: str
        :param image_name: Name of the image
        :type image_name: str
        :return: Metrics dict with id, mean_iou, num_gt, num_pred
        :rtype: dict
        """
        json_data = self.load_json_file(model_name)
        failed_dict = json_data["results"]["per_image_failed"]

        if image_name in failed_dict:
            return failed_dict[image_name]
        return None
