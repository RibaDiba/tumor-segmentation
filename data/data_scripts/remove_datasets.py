import os, shutil

"""
this is just a simple script to remove a certain types of datasets 
could be done with a command line
"""

def remove_dataset(unique_string: str, new_path: str): 
    """
    remove a certain type of image/data from useable_data and place it into a new path
    """
    os.makedirs(new_path, exist_ok=True) # make sure new path exists 
    useable_data_path = "../huggingface-repo/useable_data" # default path 

    for filename in os.listdir(useable_data_path): 
        basename, ext = os.path.splitext(filename)

        if basename.startswith(unique_string): 
            path = os.path.join(useable_data_path, filename)
            new_path_file = os.path.join(new_path, filename)
            shutil.copy(path, new_path_file)
            os.remove(path)

    print(f"Removed files that start with {unique_string} from useable_data and placed into {new_path}")

def main(): 
    remove_dataset(unique_string="L3.3 Parental 2-Control-", 
                   new_path="../huggingface-repo/removed-data/invotive")

if __name__ == "__main__": 
    main()    