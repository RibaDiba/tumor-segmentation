from typing import Dict
import matplotlib.pyplot as plt
import json, os
import seaborn as sns
import pandas as pd


def get_bottom_count(data, count):
    pass


def create_graph(rgb_dict, depth_dict, rgd_dict, output_dir, name):
    keys = []
    values_rgb = []
    values_depth = []
    values_rgd = []

    rows = []

    os.makedirs(output_dir, exist_ok=True)

    for key_image in rgb_dict["results"]["per_image_iou"]:
        image_dict_rgb = rgb_dict["results"]["per_image_iou"][key_image]
        image_dict_depth = depth_dict["results"]["per_image_iou"][key_image]
        image_dict_rgd = rgd_dict["results"]["per_image_iou"][key_image]

        keys.append(image_dict_rgb["id"])
        values_rgb.append(image_dict_rgb["mean_iou"])
        values_depth.append(image_dict_depth["mean_iou"])
        values_rgd.append(image_dict_rgd["mean_iou"])

        rows.append(
            {
                "Image ID": key_image,
                "RGB": image_dict_rgb["mean_iou"],
                "Depth": image_dict_depth["mean_iou"],
                "RGD": image_dict_rgd["mean_iou"],
            }
        )

    plt.scatter(keys, values_rgb, label="RGB")
    plt.scatter(keys, values_depth, label="Depth")
    plt.scatter(keys, values_rgd, label="RGD")

    plt.xlabel("Image ID")
    plt.ylabel("IoU Score")

    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, name))

    # df = pd.DataFrame(rows)
    # df = df.set_index("Image ID")

    # # Make heatmap
    # plt.figure(figsize=(8, 10))
    # sns.heatmap(df, annot=False, cmap="viridis", cbar_kws={'label': 'IoU Score'}, vmin=0, vmax=1)

    # plt.title("IoU Scores per Image across Models")
    # plt.xlabel("Model")
    # plt.ylabel("Image ID")
    # plt.savefig("./example_plot")


def create_dict_json(file_path) -> Dict:
    with open(file_path, "r") as f:
        data = json.load(f)

    return data


def main():
    # load the json files
    rgb_json = "src/Detectron2/slurm_output/IoU_fig/json/rgb-5000-1_json_results.json"
    depth_json = (
        "src/Detectron2/slurm_output/IoU_fig/json/depth-5000-1_json_results.json"
    )
    rgd_json = "src/Detectron2/slurm_output/IoU_fig/json/rgd-5000-1_json_results.json"

    rgb_dict = create_dict_json(rgb_json)
    depth_dict = create_dict_json(depth_json)
    rgd_dict = create_dict_json(rgd_json)

    out = "./graphs"
    create_graph(
        rgb_dict, depth_dict, rgd_dict, output_dir=out, name="Scatter_Plot.png"
    )


if __name__ == "__main__":
    main()
