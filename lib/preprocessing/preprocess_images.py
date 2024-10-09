import matplotlib.pyplot as plt 
import numpy as np 
import cv2, os, random, io, importlib
from scipy.interpolate import griddata
from tqdm import tqdm
from PIL import Image
from io import BytesIO

# import preprocessing functions 
import functions

# reload to save changes during dev (idk if this is required, but it is on notebook files)
importlib.reload(functions)

from functions import read_images_to_array, split_images, split_train_val_test, \
crop_raw_images, crop_masks, add_padding, zoom_at, create_binary_masks, crop_images, \
crop_images_offset, read_bin, read_contours_array, read_contours_array_depth, infuse_depth_into_blue_channel, \
read_all_bins, translate_images_np



def preprocess_rgb(folder_path, per_train, per_val, per_test):

    images = read_images_to_array(folder_path)

    masks, images = split_images(images)
   
    images = crop_raw_images(images)
    masks = crop_masks(masks)

    images, masks = add_padding(images, masks)

    masks = zoom_at(masks, 1.333, coord=None)
    masks = create_binary_masks(masks)

    images = crop_images(images)
    masks = crop_images_offset(masks, x_offset=-25)

    print(f'Number of Images: {len(images)}')
    print()

    train_images, train_masks, val_images, val_masks, test_images, test_masks = split_train_val_test(images, masks, per_train, per_val, per_test)

    print(f'Number of Train Images: {len(train_images)}')
    print(f'Number of Val Images: {len(val_images)}')
    print(f'Number of Test Images: {len(test_images)}')

    return train_images, train_masks, val_images, val_masks, test_images, test_masks

def preprocess_grayscale(folder_path, per_train, per_val, per_test):

    data_array = read_all_bins(folder_path)
    depth_maps = read_contours_array_depth(data_array)

    images = read_images_to_array(folder_path)
    masks, images = split_images(images)
   
    images = crop_images(images)
    

    depth_maps = crop_raw_images(depth_maps)
    depth_maps = crop_images(depth_maps)

    depth_maps, masks = add_padding(depth_maps, masks)

    masks = crop_masks(masks)
    masks = zoom_at(masks, 1.333, coord=None)
    masks = create_binary_masks(masks)
    masks = crop_images_offset(masks, x_offset=-25)

    print(f'Number of Images: {len(images)}')
    print()

    train_images, train_masks, val_images, val_masks, test_images, test_masks = split_train_val_test(depth_maps, masks, per_train, per_val, per_test)

    print(f'Number of Train Images: {len(train_images)}')
    print(f'Number of Val Images: {len(val_images)}')
    print(f'Number of Test Images: {len(test_images)}')

    return train_images, train_masks, val_images, val_masks, test_images, test_masks

"""
def preprocess_rgbd(folder_path, per_train, per_val, per_test):
    data_array = read_all_bins(folder_path)
    depth_maps = read_contours_array(data_array)

    images = read_images_to_array(folder_path)
    masks, images = split_images(images)

    # Resize depth maps to match image dimensions
    depth_maps_resized = [cv2.resize(depth_map, (image.shape[1], image.shape[0])) 
                          for depth_map, image in zip(depth_maps, images)]

    # Crop, pad, and process the depth maps
    depth_maps_resized = crop_raw_images(depth_maps_resized)
    depth_maps_resized = add_padding(depth_maps_resized, 0, 67)
    depth_maps_resized = crop_images(depth_maps_resized)

   
    masks = crop_masks(masks)
    masks = add_padding(masks, 31, 0)
    masks = zoom_at(masks, 1.156, coord=None)
    masks = create_binary_masks(masks)
    masks = crop_images(masks)

    print(f'Number of Images: {len(images)}')
    print()

  
    train_images, train_masks, val_images, val_masks, test_images, test_masks = split_train_val_test(
        depth_maps_resized, masks, per_train, per_val, per_test
    )

    # Print the number of images in each set
    print(f'Number of Train Images: {len(train_images)}')
    print(f'Number of Val Images: {len(val_images)}')
    print(f'Number of Test Images: {len(test_images)}')

    return train_images, train_masks, val_images, val_masks, test_images, test_masks
"""

# TODO: split into train, val, test

def preprocess_rgbd(folder_path, per_train, per_val, per_test):

    data_array = read_all_bins(folder_path)
    depth_maps = read_contours_array_depth(data_array)

    images = read_images_to_array(folder_path)
    masks, images = split_images(images)

    og_images = images

    depth_maps = crop_raw_images(depth_maps)
    images = crop_raw_images(images)

    images = infuse_depth_into_blue_channel(images, depth_maps)

    images, masks = add_padding(images, masks)

    images = crop_raw_images(images)
    images = crop_images(images)

    masks = crop_masks(masks)
    # TODO: check zoom
    masks = zoom_at(masks, 1.333, coord=None)
    masks = translate_images_np(masks, x_offset=25)
    masks = crop_images(masks)
    masks = create_binary_masks(masks)

    return images, og_images, masks, depth_maps

def preprocess_no_tumor(image_array):

    image_array = crop_raw_images(image_array)
    image_array = add_padding(image_array, 0, 67)
    image_array = crop_images(image_array) 

    return image_array

