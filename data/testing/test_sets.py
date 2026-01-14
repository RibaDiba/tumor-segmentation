import pytest, os, cv2
from pathlib import Path

"""
this test is to just show how many sets there are 
prints the number of sets found and number of basenames that are invalid 

essentially, we are creating a dictionary whose keys are unique "base names" 
the unique basenames point to a set that contains each found type (bin, jpg, _texture) --> this is filled during a first passthrough 
the second passthrough through the dict checks to see if each key points to a set that has all 3 file types
"""

# Get project root directory
project_root = Path(__file__).parent.parent.parent


@pytest.mark.parametrize(
    "dir",
    [
        (project_root / "data/huggingface-repo/useable_data"),
    ],
)
def test_sets(dir):
    # Convert to string if it's a Path object for compatibility with os functions
    dir_str = str(dir)
    assert os.path.isdir(dir_str), "directory does not exist"

    # creating a dictionary here
    file_sets = {}

    # now we're going to get all files and sort them
    for file_name in os.listdir(dir_str):
        base_name, ext = os.path.splitext(file_name)

        if base_name.endswith("_texture"):
            actual_base = base_name[:-8]  # remove "_texture"
            file_type = "texture"
        elif ext.lower() == ".bin":
            actual_base = base_name
            file_type = "bin"
        elif ext.lower() == ".jpg":
            actual_base = base_name
            file_type = "jpg"
        else:
            continue

        # create the base set if it doesn't exist
        if actual_base not in file_sets:
            file_sets[actual_base] = set()

        file_sets[actual_base].add(file_type)

    complete_sets = 0
    incomplete_sets = 0
    missing_files_report = []

    required_types = {"jpg", "texture", "bin"}

    for base_name, found_types in file_sets.items():
        missing_types = required_types - found_types

        if not missing_types:
            complete_sets += 1
        else:
            incomplete_sets += 1
            missing_files = []
            for missing_type in missing_types:
                if missing_type == "texture":
                    missing_files.append(f"{base_name}_texture.jpg")
                elif missing_type == "bin":
                    missing_files.append(f"{base_name}.bin")
                elif missing_type == "jpg":
                    missing_files.append(f"{base_name}.jpg")

            missing_files_report.append(
                {
                    "base_name": base_name,
                    "found_types": found_types,
                    "missing_files": missing_files,
                }
            )

    # code below is for logging purposes
    print(
        f"""File Set Validation Report for {dir}
                =====================================
                Total file sets found: {len(file_sets)}
                Complete sets (all 3 files): {complete_sets}
                Incomplete sets: {incomplete_sets}
                """
    )

    # i wrote this only if the data has few bad ones, not really helpful if there are large amounts of data that are bad
    # if incomplete_sets > 0:
    #     print("Detailed breakdown of incomplete sets:")
    #     print("-" * 40)
    #     for item in missing_files_report:
    #         print(f"Base name: {item['base_name']}")
    #         print(f"  Found: {', '.join(sorted(item['found_types']))}")
    #         print(f"  Missing: {', '.join(item['missing_files'])}")
    #         print()

    assert (
        incomplete_sets == 0
    ), f"Found {incomplete_sets} incomplete file sets. See report above for details."
