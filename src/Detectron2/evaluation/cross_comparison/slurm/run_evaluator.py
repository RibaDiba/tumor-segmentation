# this is to actually exec the evaluator in the job scheudler
import sys
import os
import argparse

# Add parent directory to path to import scripts package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import TumorEvaluator

def main(rgb_model, depth_model, rgd_model, model_type, slurm_output_dir, output_dir):

    # init the tumor eval
    evaluator = TumorEvaluator(
        model_names={
            "rgb": rgb_model,
            "depth": depth_model,
            "rgd": rgd_model
        },
        model_type=model_type,
        slurm_output_dir=slurm_output_dir,
    )

    comparison = evaluator.process_results()
    results = evaluator.save_results(
        plot_results=True,
        output_dir=output_dir
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="To run the evaluator for cross comparison"
    )

    parser.add_argument("--output", type=str, help="Output dir for failed image comparisons")
    parser.add_argument("--rgb-model", type=str, help="Model name for RGB")
    parser.add_argument("--depth-model", type=str, help="Model name for depth")
    parser.add_argument("--rgd-model", type=str, help="Model name for rgd")
    parser.add_argument("--model-type", type=str, help="Model type (subdirectory in slurm output, e.g. '7030_SPLIT')")
    parser.add_argument("--slurm-output-dir", type=str, help="Root slurm output directory containing pre-saved inference outputs")

    args = parser.parse_args()

    rgb_model_name = args.rgb_model
    depth_model_name = args.depth_model
    rgd_model_name = args.rgd_model
    output_dir = args.output
    model_type = args.model_type
    slurm_output_dir = args.slurm_output_dir

    main(rgb_model_name, depth_model_name, rgd_model_name, model_type, slurm_output_dir, output_dir)
