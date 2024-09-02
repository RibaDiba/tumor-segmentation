import cv2, os, random, importlib, numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt 
from PIL import Image
import preprocess_images

importlib.reload(preprocess_images)
from preprocess_images import read_images_to_array, preprocess_no_tumor

def add_no_tumor(train_images, train_masks, val_images, val_masks):
    no_tumor_images = read_images_to_array('../data/no_tumor')
    no_tumor_images = preprocess_no_tumor(no_tumor_images)

    for image in tqdm(no_tumor_images, desc="Adding images with no Tumor"):
        train_images.append(image)
        val_images.append(image)

        blank_image = np.ones((256, 256, 3), dtype=np.uint8) * 0
        train_masks.append(blank_image)
        val_masks.append(blank_image)

    print(f'New Images Added: {len(no_tumor_images)}')
    print()
    print(f'Total Training Images: {len(train_images)}')
    print(f'Total Validation Images: {len(val_images)}')

    return train_images, train_masks, val_images, val_masks

    