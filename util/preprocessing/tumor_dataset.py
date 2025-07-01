import numpy as np
from typing import List, Tuple
from preprocess_images import *
from functions import *
from process_coco_json import *

class Dataset: 

    def __init__(self, data_path: str):
        self.data_path = data_path
    
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
    crop_images_offset = crop_images_offset # why is this here 
    translate_images = translate_images # util?
    read_neg_images = read_neg_images
    create_neg_masks = create_neg_masks

    # coco_json methods 
    images_annotations_info = images_annotations_info
    process_masks = process_masks

    # preprocess function (by default preprocesses for rgb images)
    # might remove the settings feature
    def preprocess_images(self, rgb: bool=True, grayscale: bool= False, rgd: bool=False, add_negative: bool=False) -> None:
        self.masks, self.images, self.depth_info = self.read_images_to_array(self.data_path)
        
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
        self.images = self.crop_raw_images(self.images)
        self.masks = self.crop_masks(self.masks)
        self.images, self.masks = add_padding(self.images, self.masks)
        self.masks = self.zoom_at(self.masks, 1.333, coord=None)
        self.masks = self.create_binary_masks(self.masks)
        self.images = self.crop_images(self.images)
        self.masks = self.crop_images_offset(self.masks, x_offset=-25)

        print("Preprocessing done!")
        print(f'Number of RGB Images: {len(self.images)}')

    # assigns each set to train, val, and test groups 
    def split_train_val_test(self, per_train: float, per_val: float, per_test: float) -> None: 
        if (per_test + per_train + per_val > 100): 
            print("Error: percentages must add up to 100")
            return
        
        train_num = int(len(self.images) * (per_train / 100))
        val_num = int(len(self.images) * (per_val / 100))
        test_num = len(self.images) - train_num - val_num  

        indices = np.arange(len(self.images))

        train_indices = indices[:train_num]
        val_indices = indices[train_num:train_num + val_num]
        test_indices = indices[train_num + val_num:]

        self.train_images = [self.images[i] for i in train_indices]
        self.train_masks = [self.masks[i] for i in train_indices]
        self.val_images = [self.images[i] for i in val_indices]
        self.val_masks = [self.masks[i] for i in val_indices]
        self.test_images = [self.images[i] for i in test_indices]
        self.test_masks = [self.masks[i] for i in test_indices]

        print("")
        print(f'Number of training sets: {len(self.train_images)}')
        print(f'Number of validation sets: {len(self.val_images)}')
        print(f'Number of testing sets: {len(self.test_images)}')

    # add coco conversion here 
    def convert_binary_to_coco(self):
        return self 
        
    # sanity check 
    def check(self):
        f, ax = plt.subplots(1, 2)
        ax[0].imshow(self.images[0])
        ax[0].set_title("Image")
        ax[1].imshow(self.masks[0])
        ax[1].set_title("Mask")
        plt.show()

    # this will load data from the cashe 
    # returns train_images, train_masks, val_images, val_masks, test_images, test_masks 
    def load_data(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        
        train_img_dir = os.path.join(project_root, "data/processed_data/train/images")
        val_img_dir = os.path.join(project_root, "data/processed_data/val/images")
        test_img_dir = os.path.join(project_root, "data/processed_data/test/images")
        
        train_mask_dir = os.path.join(project_root, "data/processed_data/train/masks")
        val_mask_dir = os.path.join(project_root, "data/processed_data/val/masks")
        test_mask_dir = os.path.join(project_root, "data/processed_data/test/masks")

        if (os.path.exists(train_img_dir) and os.path.exists(val_img_dir) and os.path.exists(test_img_dir) and
            os.path.exists(train_mask_dir) and os.path.exists(val_mask_dir) and os.path.exists(test_mask_dir)):
        
            self.train_images = [cv2.imread(os.path.join(train_img_dir, f)) for f in os.listdir(train_img_dir)]
            self.val_images = [cv2.imread(os.path.join(val_img_dir, f)) for f in os.listdir(val_img_dir)]
            self.test_images = [cv2.imread(os.path.join(test_img_dir, f)) for f in os.listdir(test_img_dir)]
            
            self.train_masks = [cv2.imread(os.path.join(train_mask_dir, f), cv2.IMREAD_GRAYSCALE) for f in os.listdir(train_mask_dir)]
            self.val_masks = [cv2.imread(os.path.join(val_mask_dir, f), cv2.IMREAD_GRAYSCALE) for f in os.listdir(val_mask_dir)]
            self.test_masks = [cv2.imread(os.path.join(test_mask_dir, f), cv2.IMREAD_GRAYSCALE) for f in os.listdir(test_mask_dir)]
            
            print("Data loaded from cache.")
        else:
            # js for convience 
            print("cashe not found, processing images")
            self.preprocess_images(rgb=True)
            self.split_train_val_test(80, 10, 10)
            print("Data processed from source.")

        return self.train_images, self.train_masks, self.val_images, self.val_masks, self.test_images, self.test_masks
    
    # this will save our data into a directory to be used for all models 
    def cashe_data(self) -> None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        
        train_img_dir = os.path.join(project_root, "data/processed_data/train/images")
        val_img_dir = os.path.join(project_root, "data/processed_data/val/images")
        test_img_dir = os.path.join(project_root, "data/processed_data/test/images")
        
        # Note: as per the nature of the conversion code, the masks have to be seperated into classes 
        train_mask_dir = os.path.join(project_root, "data/processed_data/train/masks/Tumor")
        val_mask_dir = os.path.join(project_root, "data/processed_data/val/masks/Tumor")
        test_mask_dir = os.path.join(project_root, "data/processed_data/test/masks/Tumor")

        os.makedirs(train_img_dir, exist_ok=True)
        os.makedirs(val_img_dir, exist_ok=True)
        os.makedirs(test_img_dir, exist_ok=True)
        os.makedirs(train_mask_dir, exist_ok=True)
        os.makedirs(val_mask_dir, exist_ok=True)
        os.makedirs(test_mask_dir, exist_ok=True)

        # Save images to cache directories
        for i, img in enumerate(self.train_images):
            cv2.imwrite(os.path.join(train_img_dir, f"train_{i}.jpg"), img)

        for i, img in enumerate(self.val_images):
            cv2.imwrite(os.path.join(val_img_dir, f"val_{i}.jpg"), img)

        for i, img in enumerate(self.test_images):
            cv2.imwrite(os.path.join(test_img_dir, f"test_{i}.jpg"), img)
        
        # Save masks to cache directories
        for i, mask in enumerate(self.train_masks):
            cv2.imwrite(os.path.join(train_mask_dir, f"train_{i}.png"), mask)

        for i, mask in enumerate(self.val_masks):
            cv2.imwrite(os.path.join(val_mask_dir, f"val_{i}.png"), mask)

        for i, mask in enumerate(self.test_masks):
            cv2.imwrite(os.path.join(test_mask_dir, f"test_{i}.png"), mask)

        print("Data cached successfully.")
        
    def convert_binary_to_coco(self) -> None:
        # rgb only for now
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

        train_mask_dir = os.path.join(project_root, "data/processed_data/train/masks")
        val_mask_dir = os.path.join(project_root, "data/processed_data/val/masks")
        test_mask_dir = os.path.join(project_root, "data/processed_data/test/masks")

        train_json_dir = os.path.join(project_root, "data/processed_data/train/images/train.json")
        val_json_dir = os.path.join(project_root, "data/processed_data/val/images/val.json")
        test_json_dir = os.path.join(project_root, "data/processed_data/test/images/test.json")

        process_masks(train_mask_dir, train_json_dir)
        process_masks(val_mask_dir, val_json_dir)
        process_masks(test_mask_dir, test_json_dir)


# test - this will be removed     
# Get the project root directory (2 levels up from current file)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
d = Dataset(os.path.join(project_root, "data/raw_data/useable_data"))
d.preprocess_images(rgb=True)
d.split_train_val_test(80, 10, 10)

d.cashe_data()
train_images, train_masks, val_images, val_masks, test_images, test_masks = d.load_data()

