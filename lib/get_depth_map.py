# imports 
import matplotlib.pyplot as plt 
import numpy as np 
import cv2, os, random
from scipy.interpolate import griddata
from tqdm import tqdm

def read_bin(file_path): 
    with open(file_path, 'rb') as fid:
        data = np.fromfile(fid, dtype='>f8')
    
    points = data.reshape(-1, 3)

    points[:, 0] -= np.median(points[:, 0])
    points[:, 1] -= np.median(points[:, 1])
    points[:, 2] -= np.median(points[:, 2])
    
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

def save_contours(data_array, folder_path):
    os.makedirs(folder_path, exist_ok=True)

    for data in tqdm(data_array, desc="Saving Contour Plots"):
          x, y, z, original_filename = data
          base_file_name = os.path.splitext(original_filename)[0]  
          file_name = f"{base_file_name}.png"
          path = os.path.join(folder_path, file_name)

          plt.contourf(x, y, z, levels=100, cmap="Grays")
          plt.gca().set_aspect('equal')
          plt.savefig(path)
          plt.close()


# test to see if read_all_bins works
data_array = read_all_bins('./data/bin_files')
save_contours(data_array, './data/depth_images')