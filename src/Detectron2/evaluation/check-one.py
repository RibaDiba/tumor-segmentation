from evaluate import *
from detectron2.engine import DefaultPredictor, DefaultTrainer
import cv2

"""
this file has a quick check for one file and generates an output 
not really meant to test all data at once 
"""


def eval_image(path_img: str, model_name: str, root_dir: str):

    img = cv2.imread(path_img)
    cfg = return_cfg(model_name=model_name, root_dir=root_dir)
    predictor = DefaultPredictor(cfg)
    outputs = predictor(img)  # get results

    # now visualize
    visualize(img, outputs=outputs, cfg=cfg)


eval_image(
    "../../../data/useable_data/032224 MCF7 EdPIT-EdPIT-1-49-08.jpg",
    "depth-7030-4",
    "../../../models/rgb-testing",
)


def display_original_and_prediction_with_annotations(
    val_dataset_dicts, predictor, val_metadata
):

    for d in random.sample(val_dataset_dicts, 1):  # Select number of images for display
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)

        # Create a visualizer object for the original image with annotations
        v_gt = Visualizer(im[:, :, ::-1], metadata=val_metadata, scale=0.5)
        out_gt = v_gt.draw_dataset_dict(d)

        # Create a visualizer object for the predicted image
        v_pred = Visualizer(
            im[:, :, ::-1],
            metadata=val_metadata,
            scale=0.5,
            instance_mode=ColorMode.IMAGE_BW,  # Remove the colors of unsegmented pixels
        )
        out_pred = v_pred.draw_instance_predictions(outputs["instances"].to("cpu"))

        # Set up subplots
        fig, ax = plt.subplots(1, 2, figsize=(7, 7))

        # Display the original image with annotations
        ax[0].imshow(out_gt.get_image()[:, :, ::-1])
        ax[0].set_title("Original Image with Annotations")

        # Display the predicted image
        ax[1].imshow(out_pred.get_image()[:, :, ::-1])
        ax[1].set_title("Predicted Image")

        for a in ax:
            a.axis("off")

        plt.show()


# Example usage
display_original_and_prediction_with_annotations(
    val_dataset_dicts, predictor, test_metadata
)
