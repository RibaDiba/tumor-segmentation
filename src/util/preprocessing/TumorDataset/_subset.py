from typing import List


class SubsetMixin:

    """
    this function takes in an array of ints that correspond to the file name and creates a new directory of filtered images
    this new filtered image directory will only contain images that do not have the number in the int array
    """

    def create_subset(self, train_arr: List[int], val_arr, test_arr) -> None:
        dir_arr_train = [
            self.rgb_train_dir,
            self.rgb_train_mask_dir,
            self.depth_train_dir,
            self.depth_train_mask_dir,
            self.rgd_train_dir,
            self.rgd_train_mask_dir,
        ]
        dir_arr_val = [
            self.rgb_val_dir,
            self.rgb_val_mask_dir,
            self.depth_val_dir,
            self.depth_val_mask_dir,
            self.rgd_val_dir,
            self.rgd_val_mask_dir,
        ]
        dir_arr_test = [
            self.rgb_test_dir,
            self.rgb_test_mask_dir,
            self.depth_test_dir,
            self.depth_test_mask_dir,
            self.rgd_test_dir,
            self.rgd_test_mask_dir,
        ]

        self.subset_automation(dir_array=dir_arr_train, arr=train_arr)
        self.subset_automation(dir_array=dir_arr_val, arr=val_arr)
        self.subset_automation(dir_array=dir_arr_test, arr=test_arr)
