import os

from ..augmentations import AugmentationClass


class PreprocessingMixin:

    # preprocess function (by default preprocesses for rgb images)
    # might remove the settings feature
    def preprocess_images(
        self, add_negative: bool = False, read_bins: bool = True
    ) -> None:
        self.masks, self.images_rgb, self.depth_info, self.filenames = (
            self.read_images_to_array(self.data_path, read_bins=read_bins)
        )
        self.og_masks = self.masks.copy()
        self.og_images = self.images_rgb.copy()

        # the option to process the negative images
        if add_negative:
            # Get the project root directory (4 levels up from current file)
            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../../../..")
            )
            self.negative_images = self.read_neg_images(
                os.path.join(project_root, "data/raw_data/no_tumor")
            )
            self.negative_images = self.crop_raw_images(self.negative_images)
            self.negative_images = self.crop_images(self.negative_images)

            self.negative_masks = self.create_neg_masks(len(self.negative_images))

            print(f"Processed {len(self.negative_images)} negative images")
            print(
                "Note: this does not mean that these images have been added to training "
            )

        # preprocess rgb data
        self.images_rgb = self.crop_raw_images(self.images_rgb)
        self.masks = self.crop_masks(self.masks)
        self.images_rgb, self.masks = self.add_padding(self.images_rgb, self.masks)
        self.masks = self.zoom_at(self.masks, 1.333, coord=None)
        self.images_rgb = self.crop_images(self.images_rgb)
        self.masks = self.crop_images_offset(self.masks, x_offset=-25)
        self.masks = self.create_binary_masks(self.masks)
        self.masks = self.correct_binary_masks(self.masks)

        if read_bins:
            # preprocess grayscale data
            self.masks_clone_depth = self.og_masks
            self.masks_clone_depth = self.crop_masks(self.masks_clone_depth)
            self.images_depth_maps = self.read_contours_array_depth(self.depth_info)
            self.images_depth_maps = self.crop_raw_images(self.images_depth_maps)
            # copies because the masks should be the same across all data
            self.images_depth_maps, self.masks_clone_depth = self.add_padding(
                self.images_depth_maps, self.masks_clone_depth
            )
            self.images_depth_maps = self.crop_images(self.images_depth_maps)

            self.masks_clone_rgd = self.og_masks
            self.masks_clone_rgd = self.crop_masks(self.masks_clone_rgd)
            self.images_rgd = self.read_contours_array_depth(self.depth_info)
            self.images_rgd = self.crop_raw_images(self.images_rgd)
            self.temp = self.og_images.copy()
            self.images_rgd = self.infuse_depth_into_blue_channel(
                self.temp, self.images_rgd
            )
            self.images_rgd, self.masks_clone_rgd = self.add_padding(
                self.images_rgd, self.masks_clone_rgd
            )
            self.images_rgd = self.crop_images(self.images_rgd)

            # now preprocess rgbd data - clone the rgb data then get the .npy files 
            self.images_rgbd_rgb = self.images_rgb
            self.images_rgbd_contor = self.images_depth_maps # clone from depth maps 
            self.images_rgbd_grid = self.read_contours_with_grid(self.depth_info) 


        print("Preprocessing done!")
        print(f"Number of RGB Images: {len(self.images_rgb)}")
        if read_bins:
            print(f"Number of Depth Map Images: {len(self.images_depth_maps)}")
            print(f"Number of RGD images: {len(self.images_rgd)}")

    def preprocess_augs(
        self,
        rotate_degrees: float = 15.0,
    ) -> None:
        """
        Separate preprocessing pipeline for use with custom augmentations.
        Each modality's images are processed independently; masks are shared
        (all modalities have the same annotation). augment_images() loops until
        the combined dataset reaches target_size, then shuffles all modalities
        with a single permutation to keep them in sync.

        THIS NEEDS TO BE UPDATED 
        """

        # read information
        self.masks, self.images, self.depth_info, self.filenames = (
            self.read_images_to_array(self.data_path, read_bins=True)
        )

        # local mask copies for per-modality spatial processing (all start identical)
        mask_rgb   = self.masks.copy()
        mask_depth = self.masks.copy()
        mask_rgd   = self.masks.copy()

        # rgb image pipeline
        self.images_rgb = self.images.copy()
        self.images_rgb = self.crop_raw_images(self.images_rgb)
        mask_rgb = self.crop_masks(mask_rgb)
        self.images_rgb, mask_rgb = self.add_padding(self.images_rgb, mask_rgb)
        mask_rgb = self.zoom_at(mask_rgb, 1.333, coord=None)
        self.images_rgb = self.crop_images(self.images_rgb)
        mask_rgb = self.crop_images_offset(mask_rgb, x_offset=-25)
        mask_rgb = self.create_binary_masks(mask_rgb)
        mask_rgb = self.correct_binary_masks(mask_rgb)

        # depth image pipeline
        self.images_depth_maps = self.read_contours_array_depth(self.depth_info)
        self.images_depth_maps = self.crop_raw_images(self.images_depth_maps)
        mask_depth = self.crop_masks(mask_depth)
        self.images_depth_maps, mask_depth = self.add_padding(self.images_depth_maps, mask_depth)
        mask_depth = self.zoom_at(mask_depth, 1.333, coord=None)
        self.images_depth_maps = self.crop_images(self.images_depth_maps)
        mask_depth = self.crop_images_offset(mask_depth, x_offset=-25)
        mask_depth = self.create_binary_masks(mask_depth)
        mask_depth = self.correct_binary_masks(mask_depth)

        # rgd image pipeline
        self.images_rgd = self.read_contours_array_depth(self.depth_info)
        self.images_rgd = self.crop_raw_images(self.images_rgd)
        temp = self.crop_raw_images(self.images.copy())
        self.images_rgd = self.infuse_depth_into_blue_channel(temp, self.images_rgd)
        mask_rgd = self.crop_masks(mask_rgd)
        self.images_rgd, mask_rgd = self.add_padding(self.images_rgd, mask_rgd)
        mask_rgd = self.zoom_at(mask_rgd, 1.333, coord=None)
        self.images_rgd = self.crop_images(self.images_rgd)
        mask_rgd = self.crop_images_offset(mask_rgd, x_offset=-25)
        mask_rgd = self.create_binary_masks(mask_rgd)
        mask_rgd = self.correct_binary_masks(mask_rgd)

        # set canonical shared mask (all three are identical; rgb is canonical)
        self.masks = mask_rgb

        # now preprocess rgbd data - clone the rgb data then get the .npy files 
        self.images_rgbd_rgb = self.images_rgb
        self.images_rgbd_contor = self.images_depth_maps # clone from depth maps 
        self.images_rgbd_grid = self.read_contours_with_grid(self.depth_info) 

        # now we can pass the images into the augmentation class
        self.augmentations = AugmentationClass(
            self,
            rotate_degrees=rotate_degrees,
            test_only=False,
        )

        self.augmentations.augment_images()
        (
            self.images_rgb,
            self.masks,
            self.images_depth_maps,
            self.masks_clone_depth,
            self.images_rgd,
            self.masks_clone_rgd,
            self.images_rgbd_rgb,
            self.images_rgbd_contor,
            self.images_rgbd_grid,
            self.masks_clone_rgbd,
            self.filenames,
        ) = self.augmentations.return_augmentations()

        print("Preprocessing done!")
        print(f"Number of RGB Images: {len(self.images_rgb)}")
        print(f"Number of Depth Map Images: {len(self.images_depth_maps)}")
        print(f"Number of RGD images: {len(self.images_rgd)}")
        print(f"Number of RGBD image sets: {len(self.images_rgbd_grid)}")
