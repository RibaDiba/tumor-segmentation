import os
from detectron2.data.datasets import register_coco_instances


class CocoMixin:

    # this uses the binary mask to generate the json file
    def convert_binary_to_coco(self) -> None:
        # rgb
        train_mask_dir = os.path.join(self.processed_root, "rgb/train/masks")
        val_mask_dir = os.path.join(self.processed_root, "rgb/val/masks")
        test_mask_dir = os.path.join(self.processed_root, "rgb/test/masks")

        train_json_dir = os.path.join(
            self.processed_root, "rgb/train/images/train.json"
        )
        val_json_dir = os.path.join(self.processed_root, "rgb/val/images/val.json")
        test_json_dir = os.path.join(self.processed_root, "rgb/test/images/test.json")

        self.process_masks(mask_path=train_mask_dir, dest_json=train_json_dir)
        self.process_masks(mask_path=val_mask_dir, dest_json=val_json_dir)
        self.process_masks(mask_path=test_mask_dir, dest_json=test_json_dir)

        # depth
        train_mask_dir = os.path.join(self.processed_root, "depth/train/masks")
        val_mask_dir = os.path.join(self.processed_root, "depth/val/masks")
        test_mask_dir = os.path.join(self.processed_root, "depth/test/masks")

        train_json_dir = os.path.join(
            self.processed_root, "depth/train/images/train.json"
        )
        val_json_dir = os.path.join(self.processed_root, "depth/val/images/val.json")
        test_json_dir = os.path.join(self.processed_root, "depth/test/images/test.json")

        self.process_masks(mask_path=train_mask_dir, dest_json=train_json_dir)
        self.process_masks(mask_path=val_mask_dir, dest_json=val_json_dir)
        self.process_masks(mask_path=test_mask_dir, dest_json=test_json_dir)

        # rgd
        train_mask_dir = os.path.join(self.processed_root, "rgd/train/masks")
        val_mask_dir = os.path.join(self.processed_root, "rgd/val/masks")
        test_mask_dir = os.path.join(self.processed_root, "rgd/test/masks")

        train_json_dir = os.path.join(
            self.processed_root, "rgd/train/images/train.json"
        )
        val_json_dir = os.path.join(self.processed_root, "rgd/val/images/val.json")
        test_json_dir = os.path.join(self.processed_root, "rgd/test/images/test.json")

        self.process_masks(mask_path=train_mask_dir, dest_json=train_json_dir)
        self.process_masks(mask_path=val_mask_dir, dest_json=val_json_dir)
        self.process_masks(mask_path=test_mask_dir, dest_json=test_json_dir)

        # rgbd_early (only present if cached). file_name in the generated json is
        # <stem>.jpg, matching the cached RGB image; the depth .npy sits alongside
        # for the 4-channel mapper.
        for variant in ("rgbd_early",):
            for split in ("train", "val", "test"):
                mask_dir = os.path.join(self.processed_root, variant, split, "masks")
                if not os.path.isdir(mask_dir):
                    continue
                json_dir = os.path.join(
                    self.processed_root, variant, split, "images", f"{split}.json"
                )
                self.process_masks(mask_path=mask_dir, dest_json=json_dir)

    """
    this function is js for convience to be used everywhere
    also because some versions dont have the correct paths
    """

    def register_instances(
        self,
        rgb: bool = False,
        depth: bool = False,
        rgd: bool = False,
        rgbd_early: bool = False,
    ) -> None:
        if rgb:
            register_coco_instances(
                "my_dataset_train",
                {},
                os.path.join(self.rgb_train_dir, "train.json"),
                self.rgb_train_dir,
            )
            register_coco_instances(
                "my_dataset_val",
                {},
                os.path.join(self.rgb_val_dir, "val.json"),
                self.rgb_val_dir,
            )
            register_coco_instances(
                "my_dataset_test",
                {},
                os.path.join(self.rgb_test_dir, "test.json"),
                self.rgb_test_dir,
            )
        elif depth:
            register_coco_instances(
                "my_dataset_train",
                {},
                os.path.join(self.depth_train_dir, "train.json"),
                self.depth_train_dir,
            )
            register_coco_instances(
                "my_dataset_val",
                {},
                os.path.join(self.depth_val_dir, "val.json"),
                self.depth_val_dir,
            )
            register_coco_instances(
                "my_dataset_test",
                {},
                os.path.join(self.depth_test_dir, "test.json"),
                self.depth_test_dir,
            )
        elif rgd:
            register_coco_instances(
                "my_dataset_train",
                {},
                os.path.join(self.rgd_train_dir, "train.json"),
                self.rgd_train_dir,
            )
            register_coco_instances(
                "my_dataset_val",
                {},
                os.path.join(self.rgd_val_dir, "val.json"),
                self.rgd_val_dir,
            )
            register_coco_instances(
                "my_dataset_test",
                {},
                os.path.join(self.rgd_test_dir, "test.json"),
                self.rgd_test_dir,
            )
        elif rgbd_early:
            register_coco_instances(
                "my_dataset_train",
                {},
                os.path.join(self.rgbd_early_train_dir, "train.json"),
                self.rgbd_early_train_dir,
            )
            register_coco_instances(
                "my_dataset_val",
                {},
                os.path.join(self.rgbd_early_val_dir, "val.json"),
                self.rgbd_early_val_dir,
            )
            register_coco_instances(
                "my_dataset_test",
                {},
                os.path.join(self.rgbd_early_test_dir, "test.json"),
                self.rgbd_early_test_dir,
            )
