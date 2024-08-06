import os, cv2, random, json, glob
import importlib
import format_images
importlib.reload(format_images)
from format_images import format_images

category_ids = {
    "Tumor": 0
}

MASK_EXT = 'png'
ORIGINAL_EXT = 'png'
image_id = 0
annotation_id = 0

def save_images(image_array, folder_path, base_filename='image'):
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for idx, img in enumerate(image_array):
        filename = f"{base_filename}_{idx+1}.png"
        file_path = os.path.join(folder_path, filename)
        
        cv2.imwrite(file_path, img)

    print(f"Images have been saved to {folder_path}")


train_images, train_masks, val_images, val_masks, test_images, test_masks = format_images()

save_images(train_images, './data/coco_json/train/images', base_filename='image')
save_images(train_masks, './data/coco_json/train/masks/Tumor', base_filename='image')
save_images(val_images, './data/coco_json/val/images', base_filename='image')
save_images(val_masks, './data/coco_json/val/masks/Tumor', base_filename='image')
save_images(test_images, './data/coco_json/test/images', base_filename='image')
save_images(test_masks, './data/coco_json/test/masks/Tumor', base_filename='image')
