import preprocess_images, importlib, monai, get_bounding_box, numpy as np
from statistics import mean
from transformers import SamModel, SamConfig, SamProcessor
import torch
from tqdm import tqdm
importlib.reload(preprocess_images)
importlib.reload(get_bounding_box)

from preprocess_images import preprocess_rgb, preprocess_grayscale, preprocess_rgbd
from get_bounding_box import get_bounding_box

def get_loss_rgb(model_path): 

    train_images, train_masks, val_images, val_masks, test_images, test_masks = preprocess_rgb('../data/useable_data', 70, 15, 15)

    # define our loss function
    seg_loss = monai.losses.MaskedDiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
    total_loss = []

    # init our model
    model_config = SamConfig.from_pretrained("facebook/sam-vit-base")
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    SAM_rgb = SamModel(config=model_config)
    SAM_rgb.load_state_dict(torch.load(model_path))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    SAM_rgb.to(device)

    for i in tqdm(range(len(test_images)), desc="Calculating Test Loss"):

        test_image = test_images(i)
        test_mask = test_masks(i)

        prompt = get_bounding_box(test_mask)

        inputs = processor(test_image, input_boxes=[[prompt]], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # get predicted mask
        SAM_rgb.eval()

        with torch.no_grad():
            outputs = SAM_rgb(**inputs, multimask_output=False)

        medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
        medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
        medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)

        loss = seg_loss(medsam_seg_prob, test_mask)
        total_loss.append(loss.item())

    mean_loss = mean(total_loss)

    return mean_loss

