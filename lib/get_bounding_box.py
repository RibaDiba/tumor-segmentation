import cv2, numpy as np, random

def get_bounding_box(ground_truth_map):
  
    if np.all(ground_truth_map == 0):
        # If all zeros, create a random bounding box
        H, W = ground_truth_map.shape
        x_min = np.random.randint(0, W)
        x_max = np.random.randint(x_min + 1, W + 1)  # Ensure x_max > x_min
        y_min = np.random.randint(0, H)
        y_max = np.random.randint(y_min + 1, H + 1)  # Ensure y_max > y_min
        
        bbox = [x_min, y_min, x_max, y_max]
        return bbox
    else: 
        y_indices, x_indices = np.where(ground_truth_map > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        H, W = ground_truth_map.shape
        x_min = max(0, x_min - np.random.randint(0, 50))
        x_max = min(W, x_max + np.random.randint(0, 50))
        y_min = max(0, y_min - np.random.randint(0, 50))
        y_max = min(H, y_max + np.random.randint(0, 50))
        bbox = [x_min, y_min, x_max, y_max]

        return bbox


# TODO: finish this function 
def get_bounding_box_circumscribed(image_mask, x, y, radius):
    
    top_left = (x - radius, y - radius)
    bottom_right = (x + radius, y + radius)
    w = 2 * radius
    h = 2 * radius 

    bbox = [x, y, x+w, y+h]

    return bbox
    