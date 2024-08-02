## Fine Tuning with SAM 

### Use 

In this branch, there are two notebooks.  "tumor_segmentation.ipynb' has what you need to train the dataset. "Visualizing.ipynb" has what you need to visualize the results after training the data. The other files in this folder correspond to different functions to load into the notebook to preprocess the code. 

To  preprocess and train the dataRun 'get_bounding_box.py", and 'preprocess_images.py". Then run get_depth_map.py to get depth maps. Now the data is preprocessed and you can use it to train in "tumor_segmentation.ipynb"

After you are done training it, visualize the results using "visualizing.ipynb"

To directly visualize the model, just use the model checkpoint found in google drive and run the notebook.  

### Where do our models go?

Well since github only allows a maximum of 100mb of data, they are on our shared google drive.
