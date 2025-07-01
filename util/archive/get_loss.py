import lib.preprocessing.preprocess_images as preprocess_images, importlib, monai, get_bounding_box, numpy as np
from statistics import mean
from transformers import SamModel, SamConfig, SamProcessor
import torch
from tqdm import tqdm
importlib.reload(preprocess_images)
importlib.reload(get_bounding_box)

from lib.preprocessing.preprocess_images import preprocess_rgb, preprocess_grayscale, preprocess_rgbd
from get_bounding_box import get_bounding_box

def get_loss_rgb(model_path, UseMedSAM=False): 

    train_images, train_masks, val_images, val_masks, test_images, test_masks = preprocess_rgb('../data/useable_data', 70, 15, 15)

    # define our loss function
    seg_loss = monai.losses.DiceFocalLoss(sigmoid=True, squared_pred=True, reduction='mean')
    total_loss = []

    # init our model
    if (UseMedSAM ==False): 
        model_config = SamConfig.from_pretrained("facebook/sam-vit-base")
        processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    else: 
        model_config = SamConfig.from_pretrained("wanglab/medsam-vit-base")
        processor = SamProcessor.from_pretrained("wanglab/medsam-vit-base")

    SAM_rgb = SamModel(config=model_config)
    SAM_rgb.load_state_dict(torch.load(model_path))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    SAM_rgb.to(device)

    for i in tqdm(range(len(test_images)), desc="Calculating Test Loss"):

        test_image = test_images[i]
        test_mask = test_masks[i]

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

        medsam_seg_prob = torch.from_numpy(medsam_seg_prob).float().to(device)
        test_mask = torch.from_numpy(test_mask).float().to(device)

        loss = seg_loss(medsam_seg_prob, test_mask)
        total_loss.append(loss.item())

    mean_loss = mean(total_loss)

    return mean_loss

def get_loss_grayscale(model_path, UseMedSAM=False): 

    train_images, train_masks, val_images, val_masks, test_images, test_masks = preprocess_grayscale('../data/useable_data', 70, 15, 15)

    # define our loss function
    seg_loss = monai.losses.DiceFocalLoss(sigmoid=True, squared_pred=True, reduction='mean')
    total_loss = []

    # init our model
    if (UseMedSAM ==False): 
        model_config = SamConfig.from_pretrained("facebook/sam-vit-base")
        processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    else: 
        model_config = SamConfig.from_pretrained("wanglab/medsam-vit-base")
        processor = SamProcessor.from_pretrained("wanglab/medsam-vit-base")

    SAM_grayscale = SamModel(config=model_config)
    SAM_grayscale.load_state_dict(torch.load(model_path))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    SAM_grayscale.to(device)

    for i in tqdm(range(len(test_images)), desc="Calculating Test Loss"):

        test_image = test_images[i]
        test_mask = test_masks[i]

        prompt = get_bounding_box(test_mask)

        inputs = processor(test_image, input_boxes=[[prompt]], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # get predicted mask
        SAM_grayscale.eval()

        with torch.no_grad():
            outputs = SAM_grayscale(**inputs, multimask_output=False)

        medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
        medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
        medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)

        medsam_seg_prob = torch.from_numpy(medsam_seg_prob).float().to(device)
        test_mask = torch.from_numpy(test_mask).float().to(device)

        loss = seg_loss(medsam_seg_prob, test_mask)
        total_loss.append(loss.item())

    mean_loss = mean(total_loss)

    return mean_loss

def get_loss_rgbd(model_path, UseMedSAM=False): 

    train_images, train_masks, val_images, val_masks, test_images, test_masks = preprocess_rgbd('../data/useable_data', 70, 15, 15)

    # define our loss function
    seg_loss = monai.losses.DiceFocalLoss(sigmoid=True, squared_pred=True, reduction='mean')
    total_loss = []

    # init our model
    if (UseMedSAM ==False): 
        model_config = SamConfig.from_pretrained("facebook/sam-vit-base")
        processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    else: 
        model_config = SamConfig.from_pretrained("wanglab/medsam-vit-base")
        processor = SamProcessor.from_pretrained("wanglab/medsam-vit-base")

    SAM_rgbd = SamModel(config=model_config)
    SAM_rgbd.load_state_dict(torch.load(model_path))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    SAM_rgbd.to(device)

    for i in tqdm(range(len(test_images)), desc="Calculating Test Loss"):

        test_image = test_images[i]
        test_mask = test_masks[i]

        prompt = get_bounding_box(test_mask)

        inputs = processor(test_image, input_boxes=[[prompt]], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # get predicted mask
        SAM_rgbd.eval()

        with torch.no_grad():
            outputs = SAM_rgbd(**inputs, multimask_output=False)

        medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
        medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
        medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)

        medsam_seg_prob = torch.from_numpy(medsam_seg_prob).float().to(device)
        test_mask = torch.from_numpy(test_mask).float().to(device)

        loss = seg_loss(medsam_seg_prob, test_mask)
        total_loss.append(loss.item())

    mean_loss = mean(total_loss)

    return mean_loss

