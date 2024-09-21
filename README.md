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
     - ``get_loss.py``: this file contains functions to get the mean loss based on a model, works for all types of models as well as medsam models, still has to be updated for detectron2. 

- **`models/`**:
     - This directory is dedicated to storing the model files used in the project.
     - Due to GitHub's file size restrictions (100 MB), you may need to upload these models manually to the cluster or local envoirment

- **`data/`**: 
     - This directory hosts all our data, both useable and non-usable data goes here 
     - Contents:
          - ``get_useable_data.py``: This file is run in order to sort through data and find sets of images that contain:
               - name.jpg
               - name_texture.jpg
               - name.bin 
          - ``/coco_json``: This is where data is stored in a specfic format for our detectron2 model. This directory is cerated and populated by ``lib/setup_coco_json.py`` and ``lib/process_coco_json.py``
          - ``/useable_data``: This directory contains all the data that we can use in training. This file is populated by ``get_useable_data.py``
          - any other directory inside of this one contains the raw datasets themsevles 


- **`SAM/`**:
     - This directory contains files to train and visualize a model with SAM
     - Code in this directory access data from ``./data/useable_data`` and preprocessing code from ``./lib``. The ``preprocess_images.py`` file is used here to get all training, validation, and test data 
     - Contents: 
          - ``tumor_segmentation.ipynb``: this is a notebook file that contains the training/validation loop in order to train a new model. Make sure to set the output dir to the correct location with your specified name. Ex: 
          ```py
          folder_path_model = '../models/your_model_name.pth'
          ```
          - ``visualizing.ipynb``: this is a notebook file that allows you to test a specifc model. Remember to ajust the model path to ensure that you have the correct model loaded
          - ``calculate_loss.ipynb``: this notebook file is being built to get an easy way to display loss
          - ``test_bounding_box.ipynb``: this notebook file was created to code the bounding box function

- **`Detectron2/`**:
     - This diectory contains files to train a model with Detectron2 
     - Code in this directory accesses data from ``./data/coco_json``. Coco_json is a specifc file format used to train object detection and segmentation models. A coco_json file contains annotation information that we can use to train. ``lib/process_coco_json.py`` contains code that uses binary masks to create a coco_json file
     - Contents: 
          - ``tumor_segmentation.ipynb``: this is a notebook file that contains the training loop to train a new model. Make sure you specify the file location correctly 
          - **Detectron2 cannot be installed on the cluster envoirment, make sure you train on a local env with a gpu**

### Getting Started

1. **Clone the Repository**:
   - Start by cloning the repository to your local machine or cluster environment.

   ```bash
   git clone https://github.com/your-repository.git
   ```

   - Install dependancies and libraries 

   ```bash 
     pip install -r requirements.txt
   ```

