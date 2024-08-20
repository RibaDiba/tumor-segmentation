## Overview

This project is organized into several key directories, each containing specific components related to the development and deployment of machine learning models. Below is a detailed overview of the directory structure and their respective purposes.

### Directory Structure

- **`lib/`**: 
  - This directory contains all the files related to data preprocessing and utility functions.
  - The purpose of consolidating these files in the `lib` directory is to centralize the preprocessing logic, making it easier to manage and import across different parts of the project.
  - Contents: 
    - ``preprocess_images.py``: this file contains the functions to get our data ready for SAM and MedSAM, it includes 3 different functions that allow the user to get 3 types of training data (rgb, grayscale, and rgbd). Here's how this function works: 
    ```py
    train_images, train_masks, val_images, val_masks, test_images, test_masks = preprocess_rgb('path/to/data', percent_train, percent_val, percent_test)
    ```
    - ``setup_coco_json.py``: this file creates the directories for the detectron2 model, it does not create the coco_json file, but rather just sets up the directories (this has already been run inside this repo and does not need to be done unless data has been updated)
    - ``process_coco_json.py``: this file creates the coco_json file 
    - ``read_save_bin_files.py``: this file can manually save contour plots created by our preprocessing functions for viewing purposes
    - ``get_bounding_box.py``: this file contains the function that automatically produces a bounding box based on a binary mask

- **`models/`**:
  - This directory is dedicated to storing the model files used in the project.
  - Due to GitHub's file size restrictions (100 MB), you may need to upload these models manually to the cluster

### Getting Started

1. **Clone the Repository**:
   - Start by cloning the repository to your local machine or cluster environment.

   ```bash
   git clone https://github.com/your-repository.git
   ```

