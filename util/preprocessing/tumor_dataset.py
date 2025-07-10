import numpy as np, re, shutil
from typing import List, Tuple
from preprocess_images import *
from functions import *
from process_coco_json import *

class Dataset: 

    def __init__(self, data_path: str):
        self.data_path = data_path
        print("Init Dataset")

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

        self.rgb_train_mask_dir = os.path.join(project_root, "data/processed_data/rgb/train/masks/Tumor")
        self.rgb_val_mask_dir = os.path.join(project_root, "data/processed_data/rgb/val/masks/Tumor")
        self.rgb_test_mask_dir = os.path.join(project_root, "data/processed_data/rgb/test/masks/Tumor")

        self.rgb_train_dir = os.path.join(project_root, "data/processed_data/rgb/train/images/")
        self.rgb_val_dir = os.path.join(project_root, "data/processed_data/rgb/val/images/")
        self.rgb_test_dir = os.path.join(project_root, "data/processed_data/rgb/test/images/")

        self.depth_train_mask_dir = os.path.join(project_root, "data/processed_data/depth/train/masks/Tumor")
        self.depth_val_mask_dir = os.path.join(project_root, "data/processed_data/depth/val/masks/Tumor")
        self.depth_test_mask_dir = os.path.join(project_root, "data/processed_data/depth/test/masks/Tumor")

        self.depth_train_dir = os.path.join(project_root, "data/processed_data/depth/train/images/")
        self.depth_val_dir = os.path.join(project_root, "data/processed_data/depth/val/images/")
        self.depth_test_dir = os.path.join(project_root, "data/processed_data/depth/test/images/")

        self.rgd_train_mask_dir = os.path.join(project_root, "data/processed_data/rgd/train/masks/Tumor")
        self.rgd_val_mask_dir = os.path.join(project_root, "data/processed_data/rgd/val/masks/Tumor")
        self.rgd_test_mask_dir = os.path.join(project_root, "data/processed_data/rgd/test/masks/Tumor")

        self.rgd_train_dir = os.path.join(project_root, "data/processed_data/rgd/train/images/")
        self.rgd_val_dir = os.path.join(project_root, "data/processed_data/rgd/val/images/")
        self.rgd_test_dir = os.path.join(project_root, "data/processed_data/rgd/test/images/")
    
    """methods of this class 
    TODO: add image augentation functions 
    note: we will add image augmentation in the detectron2 training loader itself 
    """
   # image processing functions
    read_images_to_array = read_images_to_array
    crop_raw_images = crop_raw_images
    crop_masks = crop_masks
    add_padding = add_padding
    zoom_at = zoom_at
    create_binary_masks = create_binary_masks
    crop_images = crop_images
    crop_images_offset = crop_images_offset 
    translate_images = translate_images # util?
    read_neg_images = read_neg_images
    create_neg_masks = create_neg_masks
    infuse_depth_into_blue_channel = infuse_depth_into_blue_channel # TODO: still has to be worked on 
    read_contours_array_depth = read_contours_array_depth
    read_to_array_post = read_to_array_post
    read_folder_to_array = read_folder_to_array
    correct_binary_masks = correct_binary_masks

    # coco_json methods 
    images_annotations_info = images_annotations_info
    process_masks = process_masks

    # methods for data filtering 
    filter_subset_in_folder = filter_subset_in_folder
    remove_files_in_dir = remove_files_in_dir
    save_subset_array = save_subset_array
    subet_automation = subet_automation

    # preprocess function (by default preprocesses for rgb images)
    # might remove the settings feature
    def preprocess_images(self, rgb: bool=True, grayscale: bool= False, rgd: bool=False, add_negative: bool=False) -> None:
        self.masks, self.images_rgb, self.depth_info = self.read_images_to_array(self.data_path)
        self.og_masks = self.masks.copy()
        self.og_images = self.images_rgb.copy()

        # the option to process the negative images 
        if add_negative: 
            # Get the project root directory (2 levels up from current file)
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
            self.negative_images = self.read_neg_images(os.path.join(project_root, "data/raw_data/no_tumor"))
            self.negative_images = self.crop_raw_images(self.negative_images)
            self.negative_images = self.crop_images(self.negative_images) 

            self.negative_masks = self.create_neg_masks(len(self.negative_images))

            print(f'Processed {len(self.negative_images)} negative images')
            print("Note: this does not mean that these images have been added to training ")
        
        # preprocess rgb data 
        self.images_rgb = self.crop_raw_images(self.images_rgb)
        self.masks = self.crop_masks(self.masks)
        self.images_rgb, self.masks = self.add_padding(self.images_rgb, self.masks)
        self.masks = self.zoom_at(self.masks, 1.333, coord=None)
        self.masks = self.create_binary_masks(self.masks)
        self.images_rgb = self.crop_images(self.images_rgb)
        self.masks = self.crop_images_offset(self.masks, x_offset=-25)
        self.masks = self.correct_binary_masks(self.masks)

        # preprocess grayscale data 
        self.masks_clone_depth = self.og_masks
        self.masks_clone_depth = self.crop_masks(self.masks_clone_depth)
        self.images_depth_maps = self.read_contours_array_depth(self.depth_info)
        self.images_depth_maps = self.crop_raw_images(self.images_depth_maps)
        # copies because the masks should be the same accross all data 
        self.images_depth_maps, self.masks_clone_depth = self.add_padding(self.images_depth_maps, self.masks_clone_depth)
        self.images_depth_maps = self.crop_images(self.images_depth_maps)

        #RGD data - will not be using this until verified 
        self.masks_clone_rgd = self.og_masks
        self.masks_clone_rgd = self.crop_masks(self.masks_clone_rgd)
        self.images_rgd = self.read_contours_array_depth(self.depth_info)
        self.images_rgd = self.crop_raw_images(self.images_rgd)
        self.temp = self.og_images.copy()
        self.images_rgd = self.infuse_depth_into_blue_channel(self.temp, self.images_rgd)
        self.images_rgd, self.masks_clone_rgd = self.add_padding(self.images_rgd, self.masks_clone_rgd)
        self.images_rgd = self.crop_images(self.images_rgd)

        print("Preprocessing done!")
        print(f'Number of RGB Images: {len(self.images_rgb)}')
        print(f'Number of Depth Map Images: {len(self.images_depth_maps)}')
        print(f'Number of RGD images: {len(self.images_rgd)}')

    # assigns each set to train, val, and test groups 
    def split_train_val_test(self, per_train: float, per_val: float, per_test: float) -> None: 
        if (per_test + per_train + per_val > 100): 
            print("Error: percentages must add up to 100")
            return
        
        # this is universal
        train_num = int(len(self.images_rgb) * (per_train / 100))
        val_num = int(len(self.images_rgb) * (per_val / 100))
        test_num = len(self.images_rgb) - train_num - val_num  

        indices = np.arange(len(self.images_rgb))

        train_indices = indices[:train_num]
        val_indices = indices[train_num:train_num + val_num]
        test_indices = indices[train_num + val_num:]

        # rgb data 
        self.train_images_rgb = [self.images_rgb[i] for i in train_indices]
        self.train_masks = [self.masks[i] for i in train_indices]
        self.val_images_rgb = [self.images_rgb[i] for i in val_indices]
        self.val_masks = [self.masks[i] for i in val_indices]
        self.test_images_rgb = [self.images_rgb[i] for i in test_indices]
        self.test_masks = [self.masks[i] for i in test_indices]

        # depth info data 
        self.train_images_depth = [self.images_depth_maps[i] for i in train_indices]
        self.val_images_depth = [self.images_depth_maps[i] for i in val_indices]
        self.test_images_depth = [self.images_depth_maps[i] for i in test_indices]

        # rgd
        self.train_images_rgd = [self.images_rgd[i] for i in train_indices]
        self.val_images_rgd = [self.images_rgd[i] for i in val_indices]
        self.test_images_rgd = [self.images_rgd[i] for i in test_indices]

        print("")
        print(f'Number of training sets: {len(self.train_images_rgb)}')
        print(f'Number of validation sets: {len(self.val_images_rgb)}')
        print(f'Number of testing sets: {len(self.test_images_rgb)}')
        
    # sanity check 
    def check(self):
        f, ax = plt.subplots(1, 2)
        ax[0].imshow(self.images[0])
        ax[0].set_title("Image")
        ax[1].imshow(self.masks[0])
        ax[1].set_title("Mask")
        plt.show()

    # this will load data from the cashe 
    # returns train_images for each type (rgb, depth, rgd) and shared masks
    def load_data(self, rgb: bool=False, depth: bool=False, rgd: bool=False,) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        if rgb: 
            rgb_root = os.path.join(project_root, "data/processed_data/rgb")
            return self.read_to_array_post(rgb_root)
        if depth: 
            depth_root = os.path.join(project_root, "data/processed_data/depth")
            return self.read_to_array_post(depth_root)
        if rgd: 
            rgd_root = os.path.join(project_root, "data/processed_data/rgd")
            return self.read_to_array_post(rgd_root)
    
    # this will save our data into directories for each image type, with shared masks
    def cashe_data(self) -> None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        
        # Create RGB image and mask directories
        train_img_dir_rgb = os.path.join(project_root, "data/processed_data/rgb/train/images")
        val_img_dir_rgb = os.path.join(project_root, "data/processed_data/rgb/val/images")
        test_img_dir_rgb = os.path.join(project_root, "data/processed_data/rgb/test/images")
        
        train_mask_dir_rgb = os.path.join(project_root, "data/processed_data/rgb/train/masks/Tumor")
        val_mask_dir_rgb = os.path.join(project_root, "data/processed_data/rgb/val/masks/Tumor")
        test_mask_dir_rgb = os.path.join(project_root, "data/processed_data/rgb/test/masks/Tumor")
        
        # Create Depth image and mask directories
        train_img_dir_depth = os.path.join(project_root, "data/processed_data/depth/train/images")
        val_img_dir_depth = os.path.join(project_root, "data/processed_data/depth/val/images")
        test_img_dir_depth = os.path.join(project_root, "data/processed_data/depth/test/images")
        
        train_mask_dir_depth = os.path.join(project_root, "data/processed_data/depth/train/masks/Tumor")
        val_mask_dir_depth = os.path.join(project_root, "data/processed_data/depth/val/masks/Tumor")
        test_mask_dir_depth = os.path.join(project_root, "data/processed_data/depth/test/masks/Tumor")
        
        # Create RGD image and mask directories
        train_img_dir_rgd = os.path.join(project_root, "data/processed_data/rgd/train/images")
        val_img_dir_rgd = os.path.join(project_root, "data/processed_data/rgd/val/images")
        test_img_dir_rgd = os.path.join(project_root, "data/processed_data/rgd/test/images")
        
        train_mask_dir_rgd = os.path.join(project_root, "data/processed_data/rgd/train/masks/Tumor")
        val_mask_dir_rgd = os.path.join(project_root, "data/processed_data/rgd/val/masks/Tumor")
        test_mask_dir_rgd = os.path.join(project_root, "data/processed_data/rgd/test/masks/Tumor")
        
        # Create all directories
        os.makedirs(train_img_dir_rgb, exist_ok=True)
        os.makedirs(val_img_dir_rgb, exist_ok=True)
        os.makedirs(test_img_dir_rgb, exist_ok=True)
        os.makedirs(train_mask_dir_rgb, exist_ok=True)
        os.makedirs(val_mask_dir_rgb, exist_ok=True)
        os.makedirs(test_mask_dir_rgb, exist_ok=True)
        
        os.makedirs(train_img_dir_depth, exist_ok=True)
        os.makedirs(val_img_dir_depth, exist_ok=True)
        os.makedirs(test_img_dir_depth, exist_ok=True)
        os.makedirs(train_mask_dir_depth, exist_ok=True)
        os.makedirs(val_mask_dir_depth, exist_ok=True)
        os.makedirs(test_mask_dir_depth, exist_ok=True)
        
        os.makedirs(train_img_dir_rgd, exist_ok=True)
        os.makedirs(val_img_dir_rgd, exist_ok=True)
        os.makedirs(test_img_dir_rgd, exist_ok=True)
        os.makedirs(train_mask_dir_rgd, exist_ok=True)
        os.makedirs(val_mask_dir_rgd, exist_ok=True)
        os.makedirs(test_mask_dir_rgd, exist_ok=True)
        
        # Save RGB images to cache directories
        for i, img in enumerate(self.train_images_rgb):
            cv2.imwrite(os.path.join(train_img_dir_rgb, f"train_{i}.jpg"), img)
        
        for i, img in enumerate(self.val_images_rgb):
            cv2.imwrite(os.path.join(val_img_dir_rgb, f"val_{i}.jpg"), img)
        
        for i, img in enumerate(self.test_images_rgb):
            cv2.imwrite(os.path.join(test_img_dir_rgb, f"test_{i}.jpg"), img)
        
        # Save Depth images to cache directories (if they exist)
        if hasattr(self, 'train_images_depth') and self.train_images_depth:
            for i, img in enumerate(self.train_images_depth):
                cv2.imwrite(os.path.join(train_img_dir_depth, f"train_{i}.jpg"), img)
            
            for i, img in enumerate(self.val_images_depth):
                cv2.imwrite(os.path.join(val_img_dir_depth, f"val_{i}.jpg"), img)
            
            for i, img in enumerate(self.test_images_depth):
                cv2.imwrite(os.path.join(test_img_dir_depth, f"test_{i}.jpg"), img)
        
        # Save RGD images to cache directories (if they exist)
        if hasattr(self, 'train_images_rgd') and self.train_images_rgd:
            for i, img in enumerate(self.train_images_rgd):
                cv2.imwrite(os.path.join(train_img_dir_rgd, f"train_{i}.jpg"), img)
            
            for i, img in enumerate(self.val_images_rgd):
                cv2.imwrite(os.path.join(val_img_dir_rgd, f"val_{i}.jpg"), img)
            
            for i, img in enumerate(self.test_images_rgd):
                cv2.imwrite(os.path.join(test_img_dir_rgd, f"test_{i}.jpg"), img)
        
        # Save masks to all cache directories (same masks for all image types)
        # RGB masks
        for i, mask in enumerate(self.train_masks):
            cv2.imwrite(os.path.join(train_mask_dir_rgb, f"train_{i}.png"), mask)
        
        for i, mask in enumerate(self.val_masks):
            cv2.imwrite(os.path.join(val_mask_dir_rgb, f"val_{i}.png"), mask)
        
        for i, mask in enumerate(self.test_masks):
            cv2.imwrite(os.path.join(test_mask_dir_rgb, f"test_{i}.png"), mask)
        
        # Depth masks (same as RGB masks)
        for i, mask in enumerate(self.train_masks):
            cv2.imwrite(os.path.join(train_mask_dir_depth, f"train_{i}.png"), mask)
        
        for i, mask in enumerate(self.val_masks):
            cv2.imwrite(os.path.join(val_mask_dir_depth, f"val_{i}.png"), mask)
        
        for i, mask in enumerate(self.test_masks):
            cv2.imwrite(os.path.join(test_mask_dir_depth, f"test_{i}.png"), mask)
        
        # RGD masks (same as RGB masks)
        for i, mask in enumerate(self.train_masks):
            cv2.imwrite(os.path.join(train_mask_dir_rgd, f"train_{i}.png"), mask)
        
        for i, mask in enumerate(self.val_masks):
            cv2.imwrite(os.path.join(val_mask_dir_rgd, f"val_{i}.png"), mask)
        
        for i, mask in enumerate(self.test_masks):
            cv2.imwrite(os.path.join(test_mask_dir_rgd, f"test_{i}.png"), mask)
        
        print("Data cached successfully.")

    # this uses the binary mask to generate the json file  
    def convert_binary_to_coco(self) -> None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

        #rgb
        train_mask_dir = os.path.join(project_root, "data/processed_data/rgb/train/masks")
        val_mask_dir = os.path.join(project_root, "data/processed_data/rgb/val/masks")
        test_mask_dir = os.path.join(project_root, "data/processed_data/rgb/test/masks")

        train_json_dir = os.path.join(project_root, "data/processed_data/rgb/train/images/train.json")
        val_json_dir = os.path.join(project_root, "data/processed_data/rgb/val/images/val.json")
        test_json_dir = os.path.join(project_root, "data/processed_data/rgb/test/images/test.json")

        self.process_masks(mask_path=train_mask_dir, dest_json=train_json_dir)
        self.process_masks(mask_path=val_mask_dir, dest_json=val_json_dir)
        self.process_masks(mask_path=test_mask_dir, dest_json=test_json_dir)

        #depth
        train_mask_dir = os.path.join(project_root, "data/processed_data/depth/train/masks")
        val_mask_dir = os.path.join(project_root, "data/processed_data/depth/val/masks")
        test_mask_dir = os.path.join(project_root, "data/processed_data/depth/test/masks")

        train_json_dir = os.path.join(project_root, "data/processed_data/depth/train/images/train.json")
        val_json_dir = os.path.join(project_root, "data/processed_data/depth/val/images/val.json")
        test_json_dir = os.path.join(project_root, "data/processed_data/depth/test/images/test.json")

        self.process_masks(mask_path=train_mask_dir, dest_json=train_json_dir)
        self.process_masks(mask_path=val_mask_dir, dest_json=val_json_dir)
        self.process_masks(mask_path=test_mask_dir, dest_json=test_json_dir)

        #rgd
        train_mask_dir = os.path.join(project_root, "data/processed_data/rgd/train/masks")
        val_mask_dir = os.path.join(project_root, "data/processed_data/rgd/val/masks")
        test_mask_dir = os.path.join(project_root, "data/processed_data/rgd/test/masks")

        train_json_dir = os.path.join(project_root, "data/processed_data/rgd/train/images/train.json")
        val_json_dir = os.path.join(project_root, "data/processed_data/rgd/val/images/val.json")
        test_json_dir = os.path.join(project_root, "data/processed_data/rgd/test/images/test.json")

        self.process_masks(mask_path=train_mask_dir, dest_json=train_json_dir)
        self.process_masks(mask_path=val_mask_dir, dest_json=val_json_dir)
        self.process_masks(mask_path=test_mask_dir, dest_json=test_json_dir)

    """
    this function takes in an array of ints that correspond to the file name and creates a new directory of filtered images 
    this new filtered image directory will only contain images that do not have the number in the int array
    """
    def create_subset(self, train_arr: List[int], val_arr, test_arr) -> None:
        dir_arr_train = [
            self.rgb_train_dir,
            self.rgb_train_mask_dir,
            self.depth_train_dir,
            self.depth_train_mask_dir,
            self.rgd_train_dir,
            self.rgd_train_mask_dir,
        ]
        dir_arr_val = [
            self.rgb_val_dir,
            self.rgb_val_mask_dir,
            self.depth_val_dir,
            self.depth_val_mask_dir,
            self.rgd_val_dir,
            self.rgd_val_mask_dir,
        ]
        dir_arr_test = [
            self.rgb_test_dir,
            self.rgb_test_mask_dir,
            self.depth_test_dir,
            self.depth_test_mask_dir,
            self.rgd_test_dir,
            self.rgd_test_mask_dir,
        ]

        self.subet_automation(dir_array=dir_arr_train, arr=train_arr)
        self.subet_automation(dir_array=dir_arr_val, arr=val_arr)
        self.subet_automation(dir_array=dir_arr_test, arr=test_arr)
                

