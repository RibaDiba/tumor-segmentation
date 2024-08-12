import os, importlib, cv2, numpy as np
import matplotlib.pyplot as plt 
import preprocess_images 

importlib.reload(preprocess_images)
from preprocess_images import preprocess_rgb, preprocess_grayscale, preprocess_rgbd

def save_images(image_array, folder_path, base_filename='image'):
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for idx, img in enumerate(image_array):
        filename = f"{base_filename}_{idx+1}.png"
        file_path = os.path.join(folder_path, filename)
        
        cv2.imwrite(file_path, img)

    print(f"Images have been saved to {folder_path}")

import os

def setup_rgb(folder_path, coco_json_dir, per_train, per_val, per_test):
    train_images, train_masks, val_images, val_masks, test_images, test_masks = preprocess_rgb(folder_path, per_train, per_val, per_test)

    save_images(train_images, os.path.join(coco_json_dir, 'rgb', 'train', 'images'), base_filename='train_image')
    save_images(train_masks, os.path.join(coco_json_dir, 'rgb', 'train', 'masks', 'Tumor'), base_filename='train_mask')
    save_images(val_images, os.path.join(coco_json_dir, 'rgb', 'val', 'images'), base_filename='val_image')
    save_images(val_masks, os.path.join(coco_json_dir, 'rgb', 'val', 'masks', 'Tumor'), base_filename='val_mask')
    save_images(test_images, os.path.join(coco_json_dir, 'rgb', 'test', 'images'), base_filename='test_image')
    save_images(test_masks, os.path.join(coco_json_dir, 'rgb', 'test', 'masks', 'Tumor'), base_filename='test_mask')

def setup_grayscale(folder_path, coco_json_dir, per_train, per_val, per_test):
    train_images, train_masks, val_images, val_masks, test_images, test_masks = preprocess_grayscale(folder_path, per_train, per_val, per_test)

    save_images(train_images, os.path.join(coco_json_dir, 'grayscale', 'train', 'images'), base_filename='train_image')
    save_images(train_masks, os.path.join(coco_json_dir, 'grayscale', 'train', 'masks', 'Tumor'), base_filename='train_mask')
    save_images(val_images, os.path.join(coco_json_dir, 'grayscale', 'val', 'images'), base_filename='val_image')
    save_images(val_masks, os.path.join(coco_json_dir, 'grayscale', 'val', 'masks', 'Tumor'), base_filename='val_mask')
    save_images(test_images, os.path.join(coco_json_dir, 'grayscale', 'test', 'images'), base_filename='test_image')
    save_images(test_masks, os.path.join(coco_json_dir, 'grayscale', 'test', 'masks', 'Tumor'), base_filename='test_mask')

def setup_rgbd(folder_path, coco_json_dir, per_train, per_val, per_test):
    train_images, train_masks, val_images, val_masks, test_images, test_masks = preprocess_rgbd(folder_path, per_train, per_val, per_test)

    save_images(train_images, os.path.join(coco_json_dir, 'rgbd', 'train', 'images'), base_filename='train_image')
    save_images(train_masks, os.path.join(coco_json_dir, 'rgbd', 'train', 'masks', 'Tumor'), base_filename='train_mask')
    save_images(val_images, os.path.join(coco_json_dir, 'rgbd', 'val', 'images'), base_filename='val_image')
    save_images(val_masks, os.path.join(coco_json_dir, 'rgbd', 'val', 'masks', 'Tumor'), base_filename='val_mask')
    save_images(test_images, os.path.join(coco_json_dir, 'rgbd', 'test', 'images'), base_filename='test_image')
    save_images(test_masks, os.path.join(coco_json_dir, 'rgbd', 'test', 'masks', 'Tumor'), base_filename='test_mask')



