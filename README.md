## Overview 

Each directory has its respective code for the type of model. The lib directory contains all files that we used for preprocessing, the purpose of this was to have everything in one place that we can import from. The models folder is where all the models go, they would most likely need to be uploaded if you're working on the cluster (github doesn't allow for files that are over 100 MB).

An additional note: bin files are currently processed in the notebook itself, which would take around 40 min, we still have to write a function that would automatically save the images.