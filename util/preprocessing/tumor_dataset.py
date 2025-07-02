import numpy as np
from typing import List, Tuple
from preprocess_images import *
from functions import *
from process_coco_json import *

class Dataset: 

    def __init__(self, data_path: str):
        self.data_path = data_path
        print("Init Dataset")
    
    # methods of this class 
    # TODO: add image augentation functions 
    # note: we will add image augmentation in the detectron2 training loader itself 
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

    # coco_json methods 
    images_annotations_info = images_annotations_info
    process_masks = process_masks

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
        
        # RGB image directories
        train_img_dir_rgb = os.path.join(project_root, "data/processed_data/rgb/train/images")
        val_img_dir_rgb = os.path.join(project_root, "data/processed_data/rgb/val/images")
        test_img_dir_rgb = os.path.join(project_root, "data/processed_data/rgb/test/images")
        
        # Depth image directories
        train_img_dir_depth = os.path.join(project_root, "data/processed_data/depth/train/images")
        val_img_dir_depth = os.path.join(project_root, "data/processed_data/depth/val/images")
        test_img_dir_depth = os.path.join(project_root, "data/processed_data/depth/test/images")
        
        # RGD image directories
        train_img_dir_rgd = os.path.join(project_root, "data/processed_data/rgd/train/images")
        val_img_dir_rgd = os.path.join(project_root, "data/processed_data/rgd/val/images")
        test_img_dir_rgd = os.path.join(project_root, "data/processed_data/rgd/test/images")
        
        # Mask directories (use RGB masks by default, but could be switched to any type)
        train_mask_dir = os.path.join(project_root, "data/processed_data/rgb/train/masks/Tumor")
        val_mask_dir = os.path.join(project_root, "data/processed_data/rgb/val/masks/Tumor")
        test_mask_dir = os.path.join(project_root, "data/processed_data/rgb/test/masks/Tumor")

        # Check if cached data exists for RGB (use this as our baseline check)
        if (os.path.exists(train_img_dir_rgb) and os.path.exists(val_img_dir_rgb) and os.path.exists(test_img_dir_rgb) and
            os.path.exists(train_mask_dir) and os.path.exists(val_mask_dir) and os.path.exists(test_mask_dir)):
        
            # Sort file lists to ensure corresponding images and masks are aligned
            # RGB Images
            train_img_files_rgb = sorted(os.listdir(train_img_dir_rgb))
            val_img_files_rgb = sorted(os.listdir(val_img_dir_rgb))
            test_img_files_rgb = sorted(os.listdir(test_img_dir_rgb))
            
            # Depth Images (if they exist)
            train_img_files_depth = sorted(os.listdir(train_img_dir_depth)) if os.path.exists(train_img_dir_depth) else []
            val_img_files_depth = sorted(os.listdir(val_img_dir_depth)) if os.path.exists(val_img_dir_depth) else []
            test_img_files_depth = sorted(os.listdir(test_img_dir_depth)) if os.path.exists(test_img_dir_depth) else []
            
            # RGD Images (if they exist)
            train_img_files_rgd = sorted(os.listdir(train_img_dir_rgd)) if os.path.exists(train_img_dir_rgd) else []
            val_img_files_rgd = sorted(os.listdir(val_img_dir_rgd)) if os.path.exists(val_img_dir_rgd) else []
            test_img_files_rgd = sorted(os.listdir(test_img_dir_rgd)) if os.path.exists(test_img_dir_rgd) else []
            
            # Masks (shared across image types)
            train_mask_files = sorted(os.listdir(train_mask_dir))
            val_mask_files = sorted(os.listdir(val_mask_dir))
            test_mask_files = sorted(os.listdir(test_mask_dir))
            
            # Load RGB images
            self.train_images_rgb = [cv2.imread(os.path.join(train_img_dir_rgb, f)) for f in train_img_files_rgb]
            self.val_images_rgb = [cv2.imread(os.path.join(val_img_dir_rgb, f)) for f in val_img_files_rgb]
            self.test_images_rgb = [cv2.imread(os.path.join(test_img_dir_rgb, f)) for f in test_img_files_rgb]
            
            # Load Depth images (if they exist)
            self.train_images_depth = [cv2.imread(os.path.join(train_img_dir_depth, f)) for f in train_img_files_depth] if train_img_files_depth else []
            self.val_images_depth = [cv2.imread(os.path.join(val_img_dir_depth, f)) for f in val_img_files_depth] if val_img_files_depth else []
            self.test_images_depth = [cv2.imread(os.path.join(test_img_dir_depth, f)) for f in test_img_files_depth] if test_img_files_depth else []
            
            # Load RGD images (if they exist)
            self.train_images_rgd = [cv2.imread(os.path.join(train_img_dir_rgd, f)) for f in train_img_files_rgd] if train_img_files_rgd else []
            self.val_images_rgd = [cv2.imread(os.path.join(val_img_dir_rgd, f)) for f in val_img_files_rgd] if val_img_files_rgd else []
            self.test_images_rgd = [cv2.imread(os.path.join(test_img_dir_rgd, f)) for f in test_img_files_rgd] if test_img_files_rgd else []
            
            # Load masks (shared across image types)
            self.train_masks = [cv2.imread(os.path.join(train_mask_dir, f), cv2.IMREAD_GRAYSCALE) for f in train_mask_files]
            self.val_masks = [cv2.imread(os.path.join(val_mask_dir, f), cv2.IMREAD_GRAYSCALE) for f in val_mask_files]
            self.test_masks = [cv2.imread(os.path.join(test_mask_dir, f), cv2.IMREAD_GRAYSCALE) for f in test_mask_files]
            
            print("Data loaded from cache.")
            print(f'RGB training images: {len(self.train_images_rgb)}, validation: {len(self.val_images_rgb)}, testing: {len(self.test_images_rgb)}')
            if self.train_images_depth:
                print(f'Depth training images: {len(self.train_images_depth)}, validation: {len(self.val_images_depth)}, testing: {len(self.test_images_depth)}')
            if self.train_images_rgd:
                print(f'RGD training images: {len(self.train_images_rgd)}, validation: {len(self.val_images_rgd)}, testing: {len(self.test_images_rgd)}')
            print(f'Training masks: {len(self.train_masks)}, validation: {len(self.val_masks)}, testing: {len(self.test_masks)}')
        else:
            # js for convience 
            print("cache not found, processing images")
            self.preprocess_images(rgb=True)
            self.split_train_val_test(80, 10, 10)
            print("Data processed from source.")

        if rgb: 
            return self.train_images_rgb, self.train_masks, self.val_images_rgb, self.val_masks, self.test_images_rgb, self.test_masks
        elif depth: 
            return self.train_images_depth, self.train_masks, self.val_images_depth, self.val_masks, self.test_images_depth, self.test_masks
        elif rgd: 
            return self.train_images_rgd, self.train_masks, self.val_images_rgd, self.val_masks, self.test_images_rgd, self.test_masks
        else: 
            print("Select a datatype to return")
    
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

        #rgb
        train_mask_dir = os.path.join(project_root, "data/processed_data/depth/train/masks")
        val_mask_dir = os.path.join(project_root, "data/processed_data/depth/val/masks")
        test_mask_dir = os.path.join(project_root, "data/processed_data/depth/test/masks")

        train_json_dir = os.path.join(project_root, "data/processed_data/depth/train/images/train.json")
        val_json_dir = os.path.join(project_root, "data/processed_data/depth/val/images/val.json")
        test_json_dir = os.path.join(project_root, "data/processed_data/depth/test/images/test.json")

        self.process_masks(mask_path=train_mask_dir, dest_json=train_json_dir)
        self.process_masks(mask_path=val_mask_dir, dest_json=val_json_dir)
        self.process_masks(mask_path=test_mask_dir, dest_json=test_json_dir)

        #rgb
        train_mask_dir = os.path.join(project_root, "data/processed_data/rgd/train/masks")
        val_mask_dir = os.path.join(project_root, "data/processed_data/rgd/val/masks")
        test_mask_dir = os.path.join(project_root, "data/processed_data/rgd/test/masks")

        train_json_dir = os.path.join(project_root, "data/processed_data/rgd/train/images/train.json")
        val_json_dir = os.path.join(project_root, "data/processed_data/rgd/val/images/val.json")
        test_json_dir = os.path.join(project_root, "data/processed_data/rgd/test/images/test.json")

        self.process_masks(mask_path=train_mask_dir, dest_json=train_json_dir)
        self.process_masks(mask_path=val_mask_dir, dest_json=val_json_dir)
        self.process_masks(mask_path=test_mask_dir, dest_json=test_json_dir)


# test - this will be removed     
# Get the project root directory (2 levels up from current file)
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
# d = Dataset(os.path.join(project_root, "data/raw_data/useable_data"))
# d.preprocess_images(rgb=True)
# d.split_train_val_test(80, 10, 10)

# d.cashe_data()
# train_images, train_masks, val_images, val_masks, test_images, test_masks = d.load_data()

