import numpy as np


class SplittingMixin:

    # assigns each set to train, val, and test groups
    def split_train_val_test(
        self, per_train: float, per_val: float, per_test: float
    ) -> None:
        if per_test + per_train + per_val > 100:
            print("Error: percentages must add up to 100")
            return

        # this is universal
        train_num = int(len(self.images_rgb) * (per_train / 100))
        val_num = int(len(self.images_rgb) * (per_val / 100))
        test_num = len(self.images_rgb) - train_num - val_num

        indices = np.arange(len(self.images_rgb))

        train_indices = indices[:train_num]
        val_indices = indices[train_num : train_num + val_num]
        test_indices = indices[train_num + val_num :]

        # split the filenames to each training set
        self.train_filenames = [self.filenames[i] for i in train_indices]
        self.val_filenames = [self.filenames[i] for i in val_indices]
        self.test_filenames = [self.filenames[i] for i in test_indices]

        # rgb data
        self.train_images_rgb = [self.images_rgb[i] for i in train_indices]
        self.train_masks = [self.masks[i] for i in train_indices]
        self.val_images_rgb = [self.images_rgb[i] for i in val_indices]
        self.val_masks = [self.masks[i] for i in val_indices]
        self.test_images_rgb = [self.images_rgb[i] for i in test_indices]
        self.test_masks = [self.masks[i] for i in test_indices]

        # depth info data
        self.train_images_depth = [self.images_depth_maps[i] for i in train_indices]
        self.val_images_depth = [self.images_depth_maps[i] for i in val_indices]
        self.test_images_depth = [self.images_depth_maps[i] for i in test_indices]

        # rgd
        self.train_images_rgd = [self.images_rgd[i] for i in train_indices]
        self.val_images_rgd = [self.images_rgd[i] for i in val_indices]
        self.test_images_rgd = [self.images_rgd[i] for i in test_indices]

        # rgbd (rgb clone + contour-render depth + rawgrid depth); only present
        # when read_bins=True. Split all three in lockstep with the same indices.
        if hasattr(self, "images_rgbd_grid"):
            self.train_images_rgbd_rgb = [self.images_rgbd_rgb[i] for i in train_indices]
            self.val_images_rgbd_rgb = [self.images_rgbd_rgb[i] for i in val_indices]
            self.test_images_rgbd_rgb = [self.images_rgbd_rgb[i] for i in test_indices]

            self.train_images_rgbd_contor = [self.images_rgbd_contor[i] for i in train_indices]
            self.val_images_rgbd_contor = [self.images_rgbd_contor[i] for i in val_indices]
            self.test_images_rgbd_contor = [self.images_rgbd_contor[i] for i in test_indices]

            self.train_images_rgbd_grid = [self.images_rgbd_grid[i] for i in train_indices]
            self.val_images_rgbd_grid = [self.images_rgbd_grid[i] for i in val_indices]
            self.test_images_rgbd_grid = [self.images_rgbd_grid[i] for i in test_indices]

        print("")
        print(f"Number of training sets: {len(self.train_images_rgb)}")
        print(f"Number of validation sets: {len(self.val_images_rgb)}")
        print(f"Number of testing sets: {len(self.test_images_rgb)}")
