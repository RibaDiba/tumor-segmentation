import os, numpy as np, matplotlib.pyplot as plt, cv2, io, importlib, preprocess_images
from tqdm import tqdm
from scipy.interpolate import griddata
from PIL import Image
importlib.reload(preprocess_images)

from preprocess_images import preprocess_rgb_single

# this is to have the depth images (both grayscale and rgbd) ready because reading all the bin files every instance takes time 

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

def save_contour_grayscale(data_array, folder_path, base_name):

    os.makedirs(folder_path, exist_ok=True)
    
    for i, data in enumerate(tqdm(data_array, desc="Reading and Saving Contour Plots")):
        x, y, z, filename = data

        plt.contourf(x, y, z, levels=100, cmap="Grays")
        plt.gca().set_aspect('equal')
        plt.axis('off')

        file_path = os.path.join(folder_path, f'{base_name}-{i}.png')
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0)

        plt.close()

def save_contour_rgbd(data_array, image_array, folder_path, base_name):

    os.makedirs(folder_path, exist_ok=True)
    depth_array = []

    for data in tqdm(data_array, desc="Reading and Saving Contour Plots"):

        x, y, z, filename = data

        plt.contourf(x,y,z, levels=100, cmap="Grays")
        plt.gca().set_aspect('equal')
        plt.axis('off')
        x, y, z, filename = data

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        image = Image.open(buf)
        image = np.array(image)
        depth_array.append(image)

        buf.close()
        plt.close()
    
    infuse_depth_into_blue_channel(image_array, depth_array, folder_path, base_name)


def infuse_depth_into_blue_channel(image_array, depth_array, folder_path, base_name):
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

        file_path = os.path.join(folder_path, f'{base_name}-{i}.png')
        cv2.imwrite(file_path, infused_image)


def main():

    data_array = read_all_bins('./data/useable_data')

    save_contour_grayscale(data_array, folder_path='./data/contour_plots/grayscale', base_name='grayscale')

    # get natural images for rgbd 
    images, masks = preprocess_rgb_single('./data/useable_data')
    save_contour_rgbd(data_array=data_array, folder_path='./data/contour_plots/rgbd', image_array=images, base_name='rgbd')

# exec the void functions 
if __name__ == "__main__":
    main()

