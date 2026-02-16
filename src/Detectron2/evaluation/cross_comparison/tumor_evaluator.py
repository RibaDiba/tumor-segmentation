import os
import cv2
from pathlib import Path
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg

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
PROJECT_ROOT = Path(__file__).resolve().parents[5]
BASE_DATA_DIR = Path(
    "/projects/PUCHALLA/LLP2024/tumor-segmentation/data/processed_data"
)
CONFIG_PATH = PROJECT_ROOT / "configs" / "cfg.yaml"


class TumorEvaluator:
    """Main evaluator class that orchestrates the comparison workflow"""

    def __init__(
        self,
        model_names: dict,
        model_root_dir: str,
        json_root: str = "/projects/PUCHALLA/LLP2024/tumor-segmentation/src/Detectron2/slurm_output/IoU_fig/json",
    ):
        """
        Initialize TumorEvaluator

        :param model_names: Dictionary mapping mode to model name (e.g., {"rgb": "rgb-7030-4", ...})
        :type model_names: dict
        :param model_root_dir: Root directory containing model weights
        :type model_root_dir: str
        :param json_root: Root directory containing JSON IoU results
        :type json_root: str
        """
        self.model_names = model_names
        self.model_root_dir = model_root_dir
        self.failed_comparison = None
        self.predictors = {}
        self.configs = {}

        # Initialize modular components
        self.json_handler = JSONHandler(json_root)
        self.image_processor = ImageProcessor(self.json_handler, model_names)
        self.plotter = ComparisonPlotter()
        self.output_writer = OutputWriter(model_names)

        print("Loading models....")
        for mode, name in self.model_names.items():
            cfg = self._setup_cfgs(name)
            self.configs[mode] = cfg
            self.predictors[mode] = DefaultPredictor(cfg)

        print("Models loaded!")

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
        plot_dir: str = None,
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

        if plot_dir is None:
            plot_dir = os.path.join(
                os.path.dirname(__file__), "..", "evaluation_outputs"
            )

        os.makedirs(plot_dir, exist_ok=True)

        # Save JSON results using OutputWriter
        output_data = self.output_writer.save_results(
            self.failed_comparison, output_dir
        )

        # Generate plots if requested
        if plot_results:
            print("\nGenerating comparison plots...")
            self.plotter.generate_plots(self.failed_comparison, plot_dir)

        return output_data

    def _setup_cfgs(self, model_name: str):
        """
        Helper function that loads config for a model

        :param model_name: Name of the model
        :type model_name: str
        :return: Configuration object
        :rtype: CfgNode
        """
        cfg = get_cfg()
        cfg.merge_from_file(str(CONFIG_PATH))
        cfg.MODEL.WEIGHTS = os.path.join(
            self.model_root_dir, model_name, "model_final.pth"
        )
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

        return cfg

    def _evaluate(self, image_name: str, image_id: int):
        """
        Takes an image and segments it using all 3 loaded models

        :param image_name: Name of the image
        :type image_name: str
        :param image_id: ID of the image
        :type image_id: int
        :return: Dictionary with visualized results for each model
        :rtype: dict
        """
        results = {}

        for mode, predictor in self.predictors.items():
            # Get the full path of the image
            image_path = os.path.join(
                BASE_DATA_DIR, mode, "test", "images", f"{image_name}"
            )

            # Error if doesn't exist
            if not os.path.exists(image_path):
                print(f"Error, could not find path: {image_path}")
                continue

            # Read, inference, and visualize
            img = cv2.imread(image_path)
            outputs = predictor(img)

            v = Visualizer(
                img[:, :, ::-1],
                MetadataCatalog.get(self.configs[mode].DATASETS.TEST[0]),
            )
            vis_output = v.draw_instance_predictions(outputs["instances"].to("cpu"))

            results[mode] = vis_output.get_image()

        return results
