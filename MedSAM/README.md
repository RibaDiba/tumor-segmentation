## Fine Tuning with MedSAM 

### Use 

In this branch, there is one central notebook.  "tumor_segmentation.ipynb' has what you need to train the dataset. The other files in this folder correspond to different functions to load into the notebook to preprocess the code. 

To  preprocess and train the dataRun 'get_bounding_box.py", and 'preprocess_images.py". Then run get_depth_map.py to get depth maps. Now the data is preprocessed and you can use it to train in "tumor_segmentation.ipynb"
