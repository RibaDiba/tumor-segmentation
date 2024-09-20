import cv2, numpy as np

def get_bounding_box(image_mask):
    
    if np.all(image_mask == 0):
        # If all zeros, create a random bounding box
        H, W = image_mask.shape
        x_min = np.random.randint(0, W)
        x_max = np.random.randint(x_min + 1, W + 1)  # Ensure x_max > x_min
        y_min = np.random.randint(0, H)
        y_max = np.random.randint(y_min + 1, H + 1)  # Ensure y_max > y_min
        
        bbox = [x_min, y_min, x_max, y_max]
    else: 
        if len(image_mask.shape) == 2 or image_mask.shape[2] == 1:
            gray = image_mask
        else:
            gray = cv2.cvtColor(image_mask, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return (0, 0, 0, 0)
        
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        bbox = [x, y, x+w, y+h]
    
    return bbox

# this is to create a looser bounding box 
def get_bounding_box_new(image_mask): 
    if np.all(image_mask == 0):
        # If all zeros, create a random bounding box
        H, W = image_mask.shape
        x_min = np.random.randint(0, W)
        x_max = np.random.randint(x_min + 1, W + 1)  # Ensure x_max > x_min
        y_min = np.random.randint(0, H)
        y_max = np.random.randint(y_min + 1, H + 1)  # Ensure y_max > y_min
        
        bbox = [x_min, y_min, x_max, y_max]
    else: 
        mask_height, mask_width = mask.shape

        while True:
            box_width = random.randint(20, mask_width // 4)
            box_height = random.randint(20, mask_height // 4)

            min_x = max(0, center_x - box_width)
            min_y = max(0, center_y - box_height)
            max_x = min(mask_width, center_x + box_width)
            max_y = min(mask_height, center_y + box_height)

            bbox = (min_x, min_y, max_x, max_y)

            mask_region = mask[min_y:max_y, min_x:max_x]

            if np.count_nonzero(mask_region) == 0:
                return bbox
    return bbox

def find_mask_center(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        raise ValueError("No contours found in the mask")

    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    
    if M["m00"] == 0:
        raise ValueError("Empty contour area")

    center_x = int(M["m10"] / M["m00"])
    center_y = int(M["m01"] / M["m00"])
    
    return center_x, center_y

# TODO: finish this function 
def get_bounding_box_circumscribed(image_mask, x, y, radius):
    
    top_left = (x - radius, y - radius)
    bottom_right = (x + radius, y + radius)
    w = 2 * radius
    h = 2 * radius 

    bbox = [x, y, x+w, y+h]

    return bbox
    