# imports 
import matplotlib.pyplot as plt 
import numpy as np 
import cv2, os, random, io, re, shutil
from scipy.interpolate import griddata
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from typing import Tuple, List 


def read_images_to_array(self, folder_path: str, read_bins: bool= True) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[str]]:

    """
    This assumes a clean dataset. First reads the directory and then returns sorted 
    images into 3 categories -> 

    Base images 
    Segmented images from the scanner
    Binary file information for depth images 

    Parameters 
    ----------
    folder_path : str
        this is the folder path that will contain all the processed data 
    read_bins : boolean 
        if we want to get the depth information for our models, then this 
        should be set to true. The reason this is a boolean because reading 
        the point cloud information takes a lot of computational overhead 
    
    Returns
    -------
    segmented_images : List[np.ndarray]
        this is an array of images of the mice that have been segmented with
        the scanner 
    base_images : List[np.ndarray]
        this is an array of raw images of the tumor with no scanner involved 
    depth_info : List[np.ndarray] 
        this currently has the point cloud information from the binary file 
    valid_filenames : List[str]
        list of strings that preserve the filenames of all the tumors, which 
        was mainly helpful for the tumor flagging webapp
    """

    segmented_images = []
    base_images = []
    depth_info = []
    # take the filenames to save later 
    valid_filenames = []

    filenames = sorted(os.listdir(folder_path))
    for filename in tqdm(filenames, desc="Reading files"):
        full_path = os.path.join(folder_path, filename)

        if filename.endswith(".jpg") and not filename.endswith("_texture.jpg"):
            img = cv2.imread(full_path)
            if img is not None:
                segmented_images.append(img)
                valid_filenames.append(filename)

        elif filename.endswith("_texture.jpg"):
            img = cv2.imread(full_path)
            if img is not None:
                base_images.append(img)

        if read_bins: 
            if filename.endswith(".bin"):
                try:
                    file_path = os.path.join(folder_path, filename)
                    x, y, z = read_bin(file_path)
                    depth_info.append((x, y, z, filename)) 
                except Exception as e:
                    print(f"Failed to read binary file {filename}: {e}")

    return segmented_images, base_images, depth_info, valid_filenames

def read_neg_images(self, folder_path: str) -> List[np.ndarray]:

    """
    this reads a path that specifically reads images that have "no tumor"
    we are not sure that these images don't have tumors, but we made 
    that assumption when filtering our data 

    Parameters
    ----------
    folder_path : str
        path where all the negative images are stored, this was done
        because the negative images were stored seperate 
    
    Returns 
    -------
    neg_images : List[np.ndarray]
        an array of images that do not contain a tumor

    """

    filenames = sorted(os.listdir)
    neg_images = []
    for filename in tqdm(filenames, desc="Reading negative images"):
        full_path = os.path.join(folder_path, filename)
        img = cv2.imread(full_path)
        neg_images.append(img)

    return neg_images

def create_neg_masks(self, length: float) -> List[np.ndarray]:

    """
    this function creates a mask for the negative images, whcih 
    is just an array of 0s (since the image does not have a tumor)

    Parameters
    ----------
    length : float 
        the amount of masks we need for negative images 
    
    Returns
    -------
        negative_masks : List[np.ndarray]
            Essentially an array of 0s that represent an empty image
    
    Note
    ----
    We don't currently use this feature, so the size of the mask is 
    inaccurate here and must be added as a parameter if this function
    is used
    """

    negative_masks = []
    for i in range(length):
        negative_mask = np.ones((495, 492), dtype=np.uint8) * 0
        negative_masks.append(negative_mask)

    return negative_masks 

def split_train_val_test(images: List[np.ndarray], masks: List[np.ndarray], per_train: int, per_val: int, per_test: int):

    """
    takes images and masks and splits them into training, validation, and tests
    
    Parameters
    ----------
    images : List[np.ndarray]
        at this point, this should be the annotated images for the tumor 
    masks : List[np.ndarray] 
        the masks that correspond to each image 
    per_train, per_val, per_test : int 
        these are the percentages that we want to split our data amoung 

    Returns
    -------
    train_images, val_images, test_images : List[np.ndarray] 
        image arrays that correspond to train, val, test
    train_masks, val_masks, test_masks
        mask arrays that correspond to train, val, test

    """

    # returns error if they don't add up 
    assert (per_train + per_val + per_test) == 100, "The percentages must sum up to 100."
    
    train_num = int(len(images) * (per_train / 100))
    val_num = int(len(images) * (per_val / 100))
    test_num = len(images) - train_num - val_num  
    
    indices = np.arange(len(images))
    
    train_indices = indices[:train_num]
    val_indices = indices[train_num:train_num + val_num]
    test_indices = indices[train_num + val_num:]
    
    train_images = [images[i] for i in train_indices]
    train_masks = [masks[i] for i in train_indices]
    val_images = [images[i] for i in val_indices]
    val_masks = [masks[i] for i in val_indices]
    test_images = [images[i] for i in test_indices]
    test_masks = [masks[i] for i in test_indices]
    
    return train_images, train_masks, val_images, val_masks, test_images, test_masks

def crop_raw_images(self, image_array: List[np.ndarray]): 

    """
    crops the raw images by doing a circle crop. These images are part of the set that are just
    pictures of the tumor 

    Parameters
    ----------
    image_array : List[np.ndarray]
        list of images from the raw picture dataset (unsegmented)
    
    Return 
    ------
    cropped_images : List[np.ndarray]
        returns the images after they have been cropped 
    """
    
    cropped_images = [] 
    
    for i in range(len(image_array)): 
        
        image = image_array[i]
        
        mask = np.zeros(image.shape, dtype=np.uint8)
        mask = cv2.circle(mask, (320, 240), 180, (255,255,255), -1)

        res = cv2.bitwise_and(image, mask)
        res[mask==0] = 255
        
        cropped_images.append(res)

    return cropped_images

def crop_masks(self, image_array: List[np.ndarray]):

    """
    before any mask processing is done, the images with the 
    red segmented region have a circle crop 

    Parameters
    ----------
    image_array : List[np.ndarray]
        these are the "masks" before any processing has been done 
    
    Return 
    ------
    cropped_images : List[np.ndarray]
        returns the images after they have been cropped 
    """

    cropped_images = []

    for i in range(len(image_array)): 
        image = image_array[i]
        
        mask = np.zeros(image.shape, dtype=np.uint8)
        mask = cv2.circle(mask, (288, 307), 200, (255,255,255), -1)

        res = cv2.bitwise_and(image, mask)
        res[mask==0] = 255
        
        cropped_images.append(res)

    return cropped_images

def add_padding(self, image_array: List[np.ndarray], mask_array: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:

    """
    the goal for our preprocessing code is to make sure that the masks and raw image 
    line up exactly on top of each other, so adding padding to both images will make 
    them the same size.

    because the size of the image really matters here, we can only support specific 
    size images.

    See Also 
    --------
    Our pytests for supported sizes shows what image sizes we support 

    Parameters
    ----------
    image_array : List[np.ndarray] 
        this is the array of images without the segmentation 
    mask_array : List[np.ndarray]
        at this point we have not chroma keyed them, but these are the images for the 
        ground truth 
    
    Return 
    ------
    padded_images, padded_masks : List[np.ndarray]
        both the image and mask array are changed
    """

    padded_images = []
    padded_masks = []

    for i in range(len(image_array)):

        image = image_array[i]
        mask = mask_array[i]  # Assuming mask_array is a list of masks

        # Check mask dimensions using len()
        if len(mask[0]) == 492:  # Check the number of columns (width)
            # MC_data
            padded_image = cv2.copyMakeBorder(
                image,
                7,
                7,
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(255,255,255)
            )

            padded_mask = cv2.copyMakeBorder(
                mask,
                0,
                0,
                74,
                74,
                cv2.BORDER_CONSTANT,
                value=(255,255,255)
            )

            padded_images.append(padded_image)
            padded_masks.append(padded_mask)

        elif len(mask[0]) == 577:  # Check the number of columns (width)
            # invotive data
            padded_image = cv2.copyMakeBorder(
                image,
                67,
                67,
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(255,255,255)
            )

            padded_mask = cv2.copyMakeBorder(
                mask,
                0,
                0,
                31,
                31,
                cv2.BORDER_CONSTANT,
                value=(255,255,255)
            )

            padded_images.append(padded_image)
            padded_masks.append(padded_mask)

        else: 
            print(f"Error: Mask dimensions {len(mask)} not recognized")

    return padded_images, padded_masks

def zoom_at(self, image_array: List[np.ndarray], zoom: float, coord: float=None) -> List[np.ndarray]:
    
    """
    takes the image and "zooms" in at a specific coords. We use this to make sure that
    the "circles" of the two images are the same size. This is key in making sure that 
    the tumors lay on top of each other 

    Parameters
    ----------
    image_array : List[np.ndarray]
        list of images that we want to zoom 
    zoom : float 
        this is the factor by which we are applying the zoom 
    coord : float 
        this is where we are applying the zoom, by default it is in the center
    
    Return 
    ------
    zoomed_array : List[np.ndarray]
        image array after the zooming process is done 

    """

    zoomed_array = []
    
    for img in image_array: 
        
        h, w, _ = [ zoom * i for i in img.shape ]

        if coord is None: cx, cy = w/2, h/2
        else: cx, cy = [ zoom*c for c in coord ]

        img = cv2.resize( img, (0, 0), fx=zoom, fy=zoom)
        img = img[ int(round(cy - h/zoom * .5)) : int(round(cy + h/zoom * .5)),
                   int(round(cx - w/zoom * .5)) : int(round(cx + w/zoom * .5)),
                   : ]
        zoomed_array.append(img)
    
    return zoomed_array

def create_binary_masks(self, image_array: List[np.ndarray]) -> List[np.ndarray]:

    """
    creates binary masks of the dataset to be used to create the COCO JSON annotations.
    We did a simple chroma key of the red region of the pre segmented images.

    Parameters
    ----------
    image_array : List[np.ndarray] 
        array of masks 
    
    Return 
    ------
    binary_masks : List[np.ndarray] 
        these are binary masks created 
    
    """

    binary_masks = []
    
    for image in image_array:
        if image.ndim == 2:
            image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] != 3:
            raise ValueError("Input image must have 3 channels (BGR format).")
        else:
            image_color = image
        
        hsv = cv2.cvtColor(image_color, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 150, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 150, 50])
        upper_red2 = np.array([180, 255, 255])

        # Create masks for the red color
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

        # Combine the two masks
        mask = cv2.bitwise_or(mask1, mask2)
        
        binary_masks.append(mask)
        
    return binary_masks

def crop_images(self, image_array: List[np.ndarray]) -> List[np.ndarray]: 

    """
    crops images to a standard 256 by 256 in the center for training, this 
    is done for both the images and the masks 

    Parameters 
    ----------
    image_array : List[np.ndarray] 
        image array to crop 

    Return 
    ------
    cropped_images : List[np.ndarray] 
        cropped set of images from the param array 
    """
    
    cropped_images = []
    
    for i in range(len(image_array)): 
        
        image = image_array[i]
        
        image_height, image_width = image.shape[:2]
        
        # Bounding box dimensions
        box_width, box_height = 256, 256

        x_top_left = (image_width - box_width) // 2
        y_top_left = (image_height - box_height) // 2
        x_bottom_right = x_top_left + box_width
        y_bottom_right = y_top_left + box_height
        
        cropped_image = image[y_top_left:y_bottom_right, x_top_left:x_bottom_right]
        cropped_images.append(cropped_image)
                              
    return cropped_images

def crop_images_offset(self, image_array: List[np.ndarray], x_offset: float=0, y_offset: float=0) -> List[np.ndarray]:
    
    """
    this is a utility function that is used in the case that some formats of images 
    could need to be offset in order to line up. Takes in x and y values to determine 
    how much to offset 

    It also applies a crop to the image at the same time. Unclear why this is a 
    seperate method.

    Parameters 
    ------
    image_array : List[np.ndarray] 
        image array to offset 
    x_offset : float 
        x value to offset the image horizontally 
    y_offset : float 
        y value to offset the image vertically

    Returns
    -------
    cropped_images : List[np.ndarray] 
        List of images after they have been cropped and offset 
    """
    
    cropped_images = []
    
    for image in image_array:
        image_height, image_width = image.shape[:2]
        
        # Bounding box dimensions
        box_width, box_height = 256, 256
        
        # Ensure the image is large enough to crop
        if image_width < box_width or image_height < box_height:
            print(f"Skipping image with dimensions {image_width}x{image_height}, too small for cropping.")
            continue
        
        # Calculate the top-left corner with offsets
        x_top_left = (image_width - box_width) // 2 + x_offset
        y_top_left = (image_height - box_height) // 2 + y_offset
        
        # Ensure the crop doesn't go out of bounds
        x_top_left = max(0, min(x_top_left, image_width - box_width))
        y_top_left = max(0, min(y_top_left, image_height - box_height))

        # Calculate bottom-right coordinates
        x_bottom_right = x_top_left + box_width
        y_bottom_right = y_top_left + box_height
        
        # Crop the image
        cropped_image = image[y_top_left:y_bottom_right, x_top_left:x_bottom_right]
        cropped_images.append(cropped_image)
                              
    return cropped_images

def translate_images(self, images: List[np.ndarray], x_offset: float, y_offset: float=0):
    
    """
    this is a utility function that is used in the case that some formats of images 
    could need to be offset in order to line up. Takes in x and y values to determine 
    how much to offset 

    Parameters 
    ------
    image_array : List[np.ndarray] 
        image array to offset 
    x_offset : float 
        x value to offset the image horizontally 
    y_offset : float 
        y value to offset the image vertically
    """
    
    translated_images = [] 

    for img_np in images:
        height, width, channels = img_np.shape

        translated_img_np = np.ones((height, width, channels), dtype=np.uint8) * 255  

        x_start = max(0, x_offset)
        x_end = min(width, width + x_offset)
        y_start = max(0, y_offset)
        y_end = min(height, height + y_offset)

        src_x_start = max(0, -x_offset)
        src_x_end = width - max(0, x_offset)
        src_y_start = max(0, -y_offset)
        src_y_end = height - max(0, y_offset)

        translated_img_np[y_start:y_end, x_start:x_end] = img_np[src_y_start:src_y_end, src_x_start:src_x_end]

        translated_images.append(translated_img_np)

    return translated_images

def read_bin(file_path): 

    """
    opens a binary file in a specific path and extracts the point cloud 
    information from that. Returns lists of x, y, and z values 

    Parameters
    ----------
    file_path : str 
        file path to binary file 

    Returns 
    -------
    grid_x, grid_y, grid_z
        lists of floats that represent the point cloud of the tumor 
    """

    with open(file_path, 'rb') as fid:
        data = np.fromfile(fid, dtype='>f8')
    
    points = data.reshape(-1, 3)
    
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    grid_x, grid_y = np.meshgrid(
        np.linspace(min(x), max(x), 256),
        np.linspace(min(y), max(y), 256)
    )
    
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='linear')

    return grid_x, grid_y, grid_z

def read_contours_array_depth(self, data_array):
     
    """
    takes the point cloud information from the data_array and generates 
    contour plots that represent the point cloud. 

    After generating contour plots in the "Grays" color, we take a picture 
    of them from a topview angle and use that as our "depth" image 

    Parameters
    ----------
    data_array : Tuple
        point cloud information to be turned into contour plots 
    
    Returns 
    -------
    image_array : List[np.ndarray]
        image array of contour plots 
    """

    image_array = []

    for data in tqdm(data_array, desc="Saving Contour Plots"):
        x, y, z, original_filename = data
        base_file_name = os.path.splitext(original_filename)[0]  
        file_name = f"{base_file_name}.png"

        plt.contourf(x, y, z, levels=100, cmap="Grays")
        plt.gca().set_aspect('equal')
        plt.axis("off")

        buf = io.BytesIO()
        plt.savefig(buf, format='jpg')
        buf.seek(0)

        image = Image.open(buf)
        image = np.array(image)
        image_array.append(image)

        buf.close()
        plt.close()

    return image_array

def infuse_depth_into_blue_channel(
    self,
    image_array: List[np.ndarray], 
    depth_array: List[np.ndarray]
) -> List[np.ndarray]:
    """
    Infuses depth map information into the blue channel of RGB images
    by fully replacing the blue channel with the normalized depth map.
    
    Parameters:
        image_array (List[np.ndarray]): List of RGB images (each shape: H x W x 3)
        depth_array (List[np.ndarray]): List of grayscale or BGR depth maps (each shape: H x W or H x W x 3)

    Returns:
        List[np.ndarray]: List of RGB images with depth fully infused into the blue channel
    """
    if len(image_array) != len(depth_array):
        raise ValueError("image_array and depth_array must have the same length")

    image_array_infused = []

    for i in tqdm(range(len(image_array)), desc="Infusing Images"):
        image = image_array[i]
        depth_map = depth_array[i]

        # Validate image and depth map
        if image is None or depth_map is None:
            raise ValueError(f"Missing image or depth map at index {i}")

        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(f"Image at index {i} is not 3-channel (RGB)")

        # Resize depth map to match image dimensions
        depth_map_resized = cv2.resize(depth_map, (image.shape[1], image.shape[0]))

        # Convert depth map to grayscale if it's BGR
        if len(depth_map_resized.shape) == 3 and depth_map_resized.shape[2] == 3:
            depth_map_resized = cv2.cvtColor(depth_map_resized, cv2.COLOR_BGR2GRAY)

        # Normalize and invert depth map to 0-255 (uint8)
        depth_map_normalized = 255 - cv2.normalize(
            depth_map_resized, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

        # Split image channels
        _, g, r = cv2.split(image)

        # Replace the blue channel with the depth map
        infused_blue = depth_map_normalized

        # Merge back the channels
        infused_image = cv2.merge((infused_blue, g, r))

        image_array_infused.append(infused_image)

    return image_array_infused

# this is to correct the chroma key error with the previous function 
def correct_binary_masks(self, mask_array: List[np.ndarray]) -> List[np.ndarray]:
    
    """
    Makes sure that the binary masks do not have holes, which can be caused by 
    defects while chroma keying the red region of the images. If there are holes 
    in the masks, then the coco formatting pipeline will create multiple annotations 
    for the same image.

    Parameters:
        mask_array (List[np.ndarray]): List of all masks

    Returns:
        List[np.ndarray]: fixed images after corrected by this function
    """

    fixed_images = []
    for i, img in enumerate(mask_array):
        # they are saved as 3 channel color images 
        binary = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2GRAY)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        binary = binary.astype(np.uint8)

        filled = np.zeros_like(binary)
        cv2.fillPoly(filled, contours, 255)

        fixed_images.append(filled)
    
    return fixed_images


def read_folder_to_array(self, folder_path: str) -> List[np.ndarray]:
    """
    This function reads a folder and returns an array of images that is 
    in the folder. This is a helper function that is used to load the data 
    that has been stored and processed in the /processed_data directory

    Parameters:
        folder_path (str): directory location of where you want to pull 
        from 
    
    Returns: 
        image_array: array of images from that directory, ignores other 
        files 
    """
    image_array = []

    exts = [".jpg", ".png"]
    files = [
        f for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
        and os.path.splitext(f)[1].lower() in exts
    ]

    # sort the files in the same way 
    files.sort()

    for f in files: 
        full_path = os.path.join(folder_path, f)
        img = cv2.imread(full_path)
        if img is None: 
            print("error with file read")
            continue
        image_array.append(img)
    
    return image_array

def read_to_array_post(self, root_path: str) -> List[np.ndarray]:

    """
    This function reads all of the subdirectories in the processed_data 
    directory and returns their images. 

    Parameters: 
        root_path (str): processed_data directory
    Returns: 
        returns the data for each split 
    """

    train_images = self.read_folder_to_array(folder_path=os.path.join(root_path, "train/images"))
    train_masks = self.read_folder_to_array(folder_path=os.path.join(root_path, "train/masks/Tumor"))

    val_images = self.read_folder_to_array(folder_path=os.path.join(root_path, "val/images"))
    val_masks = self.read_folder_to_array(folder_path=os.path.join(root_path, "val/masks/Tumor"))

    test_images = self.read_folder_to_array(folder_path=os.path.join(root_path, "test/images"))
    test_masks = self.read_folder_to_array(folder_path=os.path.join(root_path, "test/masks/Tumor"))

    return train_images, train_masks, val_images, val_masks, test_images, test_masks

# this is made for individual directories
def filter_subset_in_folder(self, arr: List[int], folder_path):
        exts = [".jpg", ".png"]
        files = [
            f for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f))
            and os.path.splitext(f)[1].lower() in exts
        ]

        image_array = []

        for f in files:
            file_num = int(re.search(r"\d+", f).group())
            if file_num in arr:
                print(f'Image {file_num} is removed')
            else: 
                img = cv2.imread(os.path.join(folder_path, f))
                image_array.append(img)

        return image_array

def remove_files_in_dir(self, folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)  # remove file or symbolic link
            elif os.path.isdir(file_path):
                # this is bassically if for some reason there is a folder
                continue
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

# saves the new image array 
def save_subset_array(self, folder_path: str, image_array: List[np.ndarray], type: str):
    for i, img in enumerate(image_array):
        # could use original names here but max
        cv2.imwrite(os.path.join(folder_path, f"{type}_{i}.jpg"), img)

def subet_automation(self, dir_array, arr): 
        for dir in dir_array: 
            # First, get the filtered array of images to keep
            image_array = self.filter_subset_in_folder(arr=arr, folder_path=dir)
            
            # Create a unique backup directory for each original directory
            dir_name = os.path.basename(dir)
            backup_dir = os.path.join(os.path.dirname(dir), f"temp_backup_{dir_name}")
            os.makedirs(backup_dir, exist_ok=True)
            
            # Save the filtered images to the backup directory
            img_type = "image" if "images" in dir else "mask"
            self.save_subset_array(folder_path=backup_dir, image_array=image_array, type=img_type)
            
            # Now that we have a backup, it's safe to remove all files from the original directory
            self.remove_files_in_dir(folder_path=dir)
            
            # Move the files from the backup to the original directory
            for file in os.listdir(backup_dir):
                src = os.path.join(backup_dir, file)
                dst = os.path.join(dir, file)
                shutil.move(src, dst)
            
            # Remove the backup directory for this directory
            shutil.rmtree(backup_dir)

