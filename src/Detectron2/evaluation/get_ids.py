import json

"""
code to get ids for poor performing images 
"""


def get_ids(model_dict, threshold):
    failed_ids = []
    for key in model_dict["results"]["per_image_iou"]:
        dict = model_dict["results"]["per_image_iou"][key]
        if dict["mean_iou"] <= threshold:
            failed_ids.append(dict["id"])

    return failed_ids


def create_dict_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    return data


def main():

    rgb_json = "src/Detectron2/slurm_output/IoU_fig/json/rgb-5000-1_json_results.json"
    depth_json = (
        "src/Detectron2/slurm_output/IoU_fig/json/depth-5000-1_json_results.json"
    )
    rgd_json = "src/Detectron2/slurm_output/IoU_fig/json/rgd-5000-1_json_results.json"

    rgb_dict = create_dict_json(rgb_json)
    depth_dict = create_dict_json(depth_json)
    rgd_dict = create_dict_json(rgd_json)

    print(get_ids(depth_dict, 0.5))


if __name__ == "__main__":
    main()
