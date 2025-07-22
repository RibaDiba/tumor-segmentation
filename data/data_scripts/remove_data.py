import os
from typing import List

"""
another util function for managing data 
just removes data with filenames
"""

# this is only for the one i'm filtering right now
r_filenames = [
    "032224 MCF7 EdPIT-Control-1-17-01",
    "032224 MCF7 EdPIT-Control-1-24-24",
    "032224 MCF7 EdPIT-iv only-1-10-09",
    "032224 MCF7 EdPIT-iv only-1-13-05"
]

def remove_data(dir, r_filenames: List[str]):
    removed_names = []
    if not os.path.exists(dir):
        print("cant find dir")
        return
    
    for filename in os.listdir(dir):
        name, ext = os.path.splitext(filename)

        # make sure we run os.remove once 
        if ext == ".jpg" and not name.endswith("_texture"):
            if name in r_filenames: 
                os.remove(os.path.join(dir, filename))
                os.remove(os.path.join(dir, f"{name}_texture.jpg"))
                os.remove(os.path.join(dir, f"{name}.bin"))

                removed_names.append(filename)

    print("removed the following filenames")
    print(removed_names)

remove_data("data/raw_data/useable_data", r_filenames)