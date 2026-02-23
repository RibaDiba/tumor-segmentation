import os
import json
import cv2
import torch
import numpy as np
from pathlib import Path
from pycocotools import mask as mask_util
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import Instances, Boxes

# Import our modular components
from .json_utils import JSONHandler
from .image_processor import ImageProcessor
from .plotter import ComparisonPlotter
from .output_writer import OutputWriter

"""
This class helps with taking an image and generating a graph
that can compare the image between all models
"""

# Constants
BASE_DATA_DIR = Path(
    "/projects/PUCHALLA/LLP2024/tumor-segmentation/data/processed_data"
)


class TumorEvaluator:
    """Main evaluator class that orchestrates the comparison workflow"""

    def __init__(
        self,
        model_names: dict,
        slurm_output_dir: str,
        model_type: str,
    ):
        """
        Initialize TumorEvaluator

        :param model_names: Dictionary mapping mode to model name (e.g., {"rgb": "rgb-7030-4", ...})
        :type model_names: dict
        :param slurm_output_dir: Root slurm output directory containing pre-saved inference outputs
        :type slurm_output_dir: str
        :param model_type: Model type subdirectory used in output path
        :type model_type: str
        """
        self.model_names = model_names
        self.model_type = model_type
        self.slurm_output_dir = slurm_output_dir
        self.failed_comparison = None
        self.outputs = {}

        # json_root is {slurm_output_dir}/{model_type}, JSONHandler appends /{model_name}/IoU_fig/json/
        json_root = os.path.join(slurm_output_dir, model_type)

        # Initialize modular components
        self.json_handler = JSONHandler(json_root)
        self.image_processor = ImageProcessor(self.json_handler, model_names)
        self.plotter = ComparisonPlotter()
        self.output_writer = OutputWriter(model_names)

        print("Loading pre-saved inference outputs...")
        self._get_outputs()
        print("Outputs loaded!")

    def _get_outputs(self):
        """
        Load pre-saved inference outputs for all models from JSON files.
        Outputs are indexed by image basename for fast lookup in _evaluate().
        """
        for mode, model_name in self.model_names.items():
            outputs_path = os.path.join(
                self.slurm_output_dir,
                self.model_type,
                model_name,
                "outputs",
                f"{model_name}_inference_outputs.json",
            )

            if not os.path.exists(outputs_path):
                raise FileNotFoundError(
                    f"Inference outputs not found for {mode} model '{model_name}': {outputs_path}"
                )

            with open(outputs_path, "r") as f:
                data = json.load(f)

            # Index test-set entries by image basename for O(1) lookup
            self.outputs[mode] = {
                os.path.basename(entry["file_name"]): entry
                for entry in data.get("test", [])
            }
            print(f"  {mode}: loaded {len(self.outputs[mode])} test predictions from {outputs_path}")

    def process_results(self):
        """
        Main function that:
        - Gets failed images from all 3 models
        - Finds overlapping failures
        - Re-runs evaluation on failed images
        - Filters results by best performer

        :return: Comparison data structure
        :rtype: dict
        """
        print("Extracting failed images from JSON files...")

        # Get failed images for each model
        failed_per_model = {}
        for mode, model_name in self.model_names.items():
            failed_per_model[mode] = set(
                self.json_handler.get_failed_images(model_name)
            )
            print(f"{mode}: {len(failed_per_model[mode])} failed images")

        # Find overlaps using set operations
        rgb_set = failed_per_model["rgb"]
        depth_set = failed_per_model["depth"]
        rgd_set = failed_per_model["rgd"]

        # Images that failed in all 3 models
        failed_all = rgb_set & depth_set & rgd_set

        # Images that failed in exactly 2 models
        failed_rgb_depth = (rgb_set & depth_set) - rgd_set
        failed_rgb_rgd = (rgb_set & rgd_set) - depth_set
        failed_depth_rgd = (depth_set & rgd_set) - rgb_set

        # Images that failed in only 1 model
        failed_rgb_only = rgb_set - depth_set - rgd_set
        failed_depth_only = depth_set - rgb_set - rgd_set
        failed_rgd_only = rgd_set - rgb_set - depth_set

        print(f"\nOverlap Analysis:")
        print(f"  Failed on all 3 models: {len(failed_all)}")
        print(
            f"  Failed on 2 models: {len(failed_rgb_depth) + len(failed_rgb_rgd) + len(failed_depth_rgd)}"
        )
        print(
            f"  Failed on 1 model: {len(failed_rgb_only) + len(failed_depth_only) + len(failed_rgd_only)}"
        )

        # Get all unique failed images (union of all sets)
        all_failed_images = rgb_set | depth_set | rgd_set
        print(f"\nProcessing {len(all_failed_images)} unique failed images...")

        # Clear results and process all failed images once
        self.image_processor.clear_results()
        self.image_processor.process_failed_images(
            all_failed_images, self._evaluate
        )

        # Filter results by best performer
        print("\nFiltering results by best performer...")
        filtered = self.image_processor.filter_best_performers()

        # Store categorized results
        self.failed_comparison = {
            "failed_all_models": filtered["failed_all"],
            "best_rgb": filtered["best_rgb"],
            "best_depth": filtered["best_depth"],
            "best_rgd": filtered["best_rgd"],
            "failed_two_models": {
                "rgb_depth": [
                    img
                    for img in self.image_processor.results
                    if img["image_name"] in failed_rgb_depth
                ],
                "rgb_rgd": [
                    img
                    for img in self.image_processor.results
                    if img["image_name"] in failed_rgb_rgd
                ],
                "depth_rgd": [
                    img
                    for img in self.image_processor.results
                    if img["image_name"] in failed_depth_rgd
                ],
            },
            "failed_one_model": {
                "rgb_only": [
                    img
                    for img in self.image_processor.results
                    if img["image_name"] in failed_rgb_only
                ],
                "depth_only": [
                    img
                    for img in self.image_processor.results
                    if img["image_name"] in failed_depth_only
                ],
                "rgd_only": [
                    img
                    for img in self.image_processor.results
                    if img["image_name"] in failed_rgd_only
                ],
            },
        }

        print("\nEvaluation complete! Use save_results() to output data.")
        return self.failed_comparison

    def save_results(
        self,
        plot_results: bool = False,
        output_dir: str = None,
    ):
        """
        Saves comparison results to JSON and optionally generates plots

        :param plot_results: Whether to generate and save comparison plots
        :type plot_results: bool
        :param output_dir: Directory to save JSON file (default: cross_comparison)
        :type output_dir: str
        :param plot_dir: Directory to save plots (default: evaluation_outputs)
        :type plot_dir: str
        :return: Output data dictionary
        :rtype: dict
        """
        if self.failed_comparison is None:
            raise ValueError(
                "No comparison data available. Run process_results() first."
            )

        # Set default directories
        if output_dir is None:
            output_dir = os.path.dirname(__file__)
        else:
            # Create output dir if not exist
            os.makedirs(output_dir, exist_ok=True)

        # Save JSON results using OutputWriter
        output_data = self.output_writer.save_results(
            self.failed_comparison, output_dir
        )

        # Generate plots if requested
        if plot_results:
            print("\nGenerating comparison plots...")
            self.plotter.generate_plots(self.failed_comparison, output_dir)

        return output_data

    def _evaluate(self, image_name: str, image_id: int):
        """
        Takes an image and visualizes predictions for all 3 models using
        pre-saved inference outputs (no model inference at runtime).

        :param image_name: Name of the image
        :type image_name: str
        :param image_id: ID of the image
        :type image_id: int
        :return: Dictionary with visualized results for each model
        :rtype: dict
        """
        results = {}

        for mode, mode_outputs in self.outputs.items():
            # Get the full path of the image
            image_path = os.path.join(
                BASE_DATA_DIR, mode, "test", "images", f"{image_name}"
            )

            if not os.path.exists(image_path):
                print(f"Error, could not find path: {image_path}")
                continue

            entry = mode_outputs.get(image_name)
            if entry is None:
                print(f"  Warning: no saved predictions for {image_name} in {mode} outputs")
                continue

            img = cv2.imread(image_path)
            h, w = img.shape[:2]

            # Reconstruct Instances from saved predictions
            instances = Instances((h, w))

            if len(entry["boxes"]) > 0:
                instances.pred_boxes = Boxes(torch.tensor(entry["boxes"], dtype=torch.float32))
                instances.scores = torch.tensor(entry["scores"], dtype=torch.float32)
                instances.pred_classes = torch.tensor(entry["classes"], dtype=torch.int64)

                # Decode RLE masks back to boolean tensors
                decoded_masks = []
                for rle in entry["masks"]:
                    rle_copy = {"size": rle["size"], "counts": rle["counts"].encode("utf-8")}
                    decoded_masks.append(mask_util.decode(rle_copy))
                instances.pred_masks = torch.from_numpy(
                    np.stack(decoded_masks)
                ).bool()
            else:
                instances.pred_boxes = Boxes(torch.empty((0, 4), dtype=torch.float32))
                instances.scores = torch.empty(0, dtype=torch.float32)
                instances.pred_classes = torch.empty(0, dtype=torch.int64)
                instances.pred_masks = torch.empty((0, h, w), dtype=torch.bool)

            v = Visualizer(img[:, :, ::-1])
            vis_output = v.draw_instance_predictions(instances)

            results[mode] = vis_output.get_image()

        return results
