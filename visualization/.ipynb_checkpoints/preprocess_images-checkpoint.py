import matplotlib.pyplot as plt 
import numpy as np 
import cv2, os, random, io
from scipy.interpolate import griddata
from tqdm import tqdm
from PIL import Image

# preprocessing image functions

def read_images_to_array(folder_path):

  image_array = []
  # Get a sorted list of filenames
  filenames = sorted(os.listdir(folder_path))
  for filename in filenames:
    if filename.endswith(".jpg") or filename.endswith(".png"):
      img_path = os.path.join(folder_path, filename)
      img = cv2.imread(img_path)

      if img is not None:
        image_array.append(img)

  return image_array

def read_bin_files_to_array(folder_path):
    bin_files = []
    filenames = sorted(os.listdir(folder_path))
    for filename in filenames:
        if filename.endswith('.bin'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'rb') as file:
                data = np.fromfile(file, dtype=np.float32)
                bin_files.append(data)

    return bin_files

def split_images(image_array): 

    red_region_images = []
    raw_images = [] 

    for image in image_array:
        if image[25,100].sum() == 255*3 :
            red_region_images.append(image)
        else: 
            raw_images.append(image) 
            
    return red_region_images, raw_images

def split_train_val_test(images, masks):

    train_images = []
    train_masks = []
    val_images = []
    val_masks = []
    test_images = []
    test_masks = []

    for i in range(len(images)): 

    # these numbers are made specifically for this dataset 
        
        if i < 20: 
            train_images.append(images[i])
            train_masks.append(masks[i])
        elif i < 25:
            val_images.append(images[i])
            val_masks.append(masks[i])
        else: 
            test_images.append(images[i])
            test_masks.append(masks[i])

    return train_images, train_masks, val_images, val_masks, test_images, test_masks

def crop_raw_images(image_array): 
    
    cropped_images = [] 
    
    for i in range(len(image_array)): 
        
        image = image_array[i]
        
        mask = np.zeros(image.shape, dtype=np.uint8)
        mask = cv2.circle(mask, (320, 240), 180, (255,255,255), -1)

        res = cv2.bitwise_and(image, mask)
        res[mask==0] = 255
        
        cropped_images.append(res)

    return cropped_images

def crop_masks(image_array):
    cropped_images = []

    for i in range(len(image_array)): 
        image = image_array[i]
        
        mask = np.zeros(image.shape, dtype=np.uint8)
        mask = cv2.circle(mask, (288, 307), 200, (255,255,255), -1)

        res = cv2.bitwise_and(image, mask)
        res[mask==0] = 255
        
        cropped_images.append(res)

    return cropped_images

def add_padding(image_array, amt_x, amt_y): 
    
    padded_images = []
    
    for image in image_array: 

        padded_image = cv2.copyMakeBorder(
            image,
            amt_y,
            amt_y,
            amt_x,
            amt_x,
            cv2.BORDER_CONSTANT,
            value=(255,255,255)
        )
        
        padded_images.append(padded_image)
        
    return padded_images

def zoom_at(image_array, zoom, coord=None):
    
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

def create_binary_masks(image_array):
    binary_masks = []
    
    for image in image_array:
        # Ensure image is in BGR format (convert if necessary)
        if image.ndim == 2:
            # Convert grayscale to BGR color (assuming gray image)
            image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] != 3:
            raise ValueError("Input image must have 3 channels (BGR format).")
        else:
            image_color = image
        
        # Convert BGR to HSV
        hsv = cv2.cvtColor(image_color, cv2.COLOR_BGR2HSV)

        # Define lower and upper bounds for red color in HSV
        lower_red = np.array([0, 150, 115])
        upper_red = np.array([255, 255, 255])

        # Create mask using inRange function
        mask = cv2.inRange(hsv, lower_red, upper_red)

        # Apply bitwise AND operation using color image
        res = cv2.bitwise_and(image_color, image_color, mask=mask)
        
        binary_masks.append(mask)
        
    return binary_masks

def crop_images(image_array): 
    
    cropped_images = []
    
    for i in range(len(image_array) -1): 
        
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

# reading bin files

def read_bin(file_path): 
    with open(file_path, 'rb') as fid:
        data = np.fromfile(fid, dtype='>f8')
    
    points = data.reshape(-1, 3)

    #points[:, 0] -= np.median(points[:, 0])
    #points[:, 1] -= np.median(points[:, 1])
    #points[:, 2] -= np.median(points[:, 2])
    
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    grid_x, grid_y = np.meshgrid(
        np.linspace(min(x), max(x), 256),
        np.linspace(min(y), max(y), 256)
    )
    
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='linear')

    return grid_x, grid_y, grid_z

def read_all_bins(folder_path):

     data_array = []
     filenames = sorted(os.listdir(folder_path))
     
     for filename in tqdm(filenames, desc="Reading Bin Files"):
          if filename.endswith(".bin"):
               file_path = os.path.join(folder_path, filename)
               x, y, z = read_bin(file_path)
               data_array.append((x, y, z, filename)) 
    
     return data_array

def read_contours_array(data_array):
    
     image_array = []
    
     for data in tqdm(data_array, desc="Reading Contour Plots"):
          x, y, z, filename = data

          plt.contourf(x,y,z, levels=100, cmap="grey")
          plt.gca().set_aspect('equal')

          x, y, z, filename = data

          plt.contourf(x, y, z, levels=100, cmap="grey")
          plt.gca().set_aspect('equal')

          # Save the plot to a buffer
          buf = io.BytesIO()
          plt.savefig(buf, format='png')
          buf.seek(0)

          # Convert the buffer to an image
          image = Image.open(buf)
          image = np.array(image)
          image_array.append(image)

          buf.close()
          plt.close()

     return image_array      


def infuse_depth_into_blue_channel(image_array, depth_array):
    image_array_infused = []

    for i in tqdm(range(len(image_array)), desc="Infusing Images"):
        image = image_array[i]
        depth_map = depth_array[i]

        # Resize the depth map to match the image dimensions
        depth_map_resized = cv2.resize(depth_map, (image.shape[1], image.shape[0]))

        # Ensure depth map is single channel (grayscale)
        if len(depth_map_resized.shape) == 3:
            depth_map_resized = cv2.cvtColor(depth_map_resized, cv2.COLOR_BGR2GRAY)

        # Split the image into RGB channels
        b, g, r = cv2.split(image)

        # Normalize depth map to match the blue channel (0-255) and convert to uint8
        depth_map_normalized = cv2.normalize(depth_map_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Infuse the depth map into the blue channel
        if b.shape == depth_map_normalized.shape:
            infused_blue = cv2.addWeighted(b, 0.5, depth_map_normalized, 0.5, 0)
        else:
            raise ValueError("Dimension mismatch between the blue channel and the depth map.")

        # Merge the channels back
        infused_image = cv2.merge((infused_blue, g, r))

        image_array_infused.append(infused_image)

    return image_array_infused

         

# split images before infusing the raw ones 

def prepreprocess():
    
     folder_path = './data/images'
     images = read_images_to_array(folder_path)
     masks, raw = split_images(images)

     return masks, raw




def preprocess_images():
    
     bin_path = './data/bin_files'
     data_array = read_all_bins(bin_path)

     masks, raw = prepreprocess()
     og = raw

     depth_maps = read_contours_array(data_array)
     image_array = infuse_depth_into_blue_channel(raw, depth_maps)

     train_images, train_masks, val_images, val_masks, test_images, test_masks = split_train_val_test(image_array, masks)

     train_images = crop_raw_images(train_images)
     train_images = add_padding(train_images, 0, 67)
     train_masks = crop_masks(train_masks)
     train_masks = add_padding(train_masks, 31, 0)
     train_masks = zoom_at(train_masks, 1.156, coord=None)
     train_masks = create_binary_masks(train_masks)

     #train_images = crop_images(train_images)
     #rain_masks = crop_images(train_masks)

     val_images = crop_raw_images(val_images)
     val_images = add_padding(val_images, 0, 67)
     val_masks = crop_masks(val_masks)
     val_masks = add_padding(val_masks, 31, 0)
     val_masks = zoom_at(val_masks, 1.156, coord=None)
     val_masks = create_binary_masks(val_masks)
     
     #val_images = crop_images(val_images)
     #val_masks = crop_images(val_masks)

     test_images = crop_raw_images(test_images)
     test_images = add_padding(test_images, 0, 67)
     test_masks = crop_masks(test_masks)
     test_masks = add_padding(test_masks, 31, 0)
     test_masks = zoom_at(test_masks, 1.156, coord=None)
     test_masks = create_binary_masks(test_masks)

     # test_images = crop_images(test_images)
     # test_masks = crop_images(test_masks)
     

     return  train_images, train_masks, val_images, val_masks, test_images, test_masks, og





