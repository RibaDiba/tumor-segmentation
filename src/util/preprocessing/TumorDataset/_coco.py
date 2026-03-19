import os
from detectron2.data.datasets import register_coco_instances


class CocoMixin:

    # this uses the binary mask to generate the json file
    def convert_binary_to_coco(self) -> None:
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../..")
        )

        # rgb
        train_mask_dir = os.path.join(
            project_root, "data/processed_data/rgb/train/masks"
        )
        val_mask_dir = os.path.join(project_root, "data/processed_data/rgb/val/masks")
        test_mask_dir = os.path.join(project_root, "data/processed_data/rgb/test/masks")

        train_json_dir = os.path.join(
            project_root, "data/processed_data/rgb/train/images/train.json"
        )
        val_json_dir = os.path.join(
            project_root, "data/processed_data/rgb/val/images/val.json"
        )
        test_json_dir = os.path.join(
            project_root, "data/processed_data/rgb/test/images/test.json"
        )

        self.process_masks(mask_path=train_mask_dir, dest_json=train_json_dir)
        self.process_masks(mask_path=val_mask_dir, dest_json=val_json_dir)
        self.process_masks(mask_path=test_mask_dir, dest_json=test_json_dir)

        # depth
        train_mask_dir = os.path.join(
            project_root, "data/processed_data/depth/train/masks"
        )
        val_mask_dir = os.path.join(project_root, "data/processed_data/depth/val/masks")
        test_mask_dir = os.path.join(
            project_root, "data/processed_data/depth/test/masks"
        )

        train_json_dir = os.path.join(
            project_root, "data/processed_data/depth/train/images/train.json"
        )
        val_json_dir = os.path.join(
            project_root, "data/processed_data/depth/val/images/val.json"
        )
        test_json_dir = os.path.join(
            project_root, "data/processed_data/depth/test/images/test.json"
        )

        self.process_masks(mask_path=train_mask_dir, dest_json=train_json_dir)
        self.process_masks(mask_path=val_mask_dir, dest_json=val_json_dir)
        self.process_masks(mask_path=test_mask_dir, dest_json=test_json_dir)

        # rgd
        train_mask_dir = os.path.join(
            project_root, "data/processed_data/rgd/train/masks"
        )
        val_mask_dir = os.path.join(project_root, "data/processed_data/rgd/val/masks")
        test_mask_dir = os.path.join(project_root, "data/processed_data/rgd/test/masks")

        train_json_dir = os.path.join(
            project_root, "data/processed_data/rgd/train/images/train.json"
        )
        val_json_dir = os.path.join(
            project_root, "data/processed_data/rgd/val/images/val.json"
        )
        test_json_dir = os.path.join(
            project_root, "data/processed_data/rgd/test/images/test.json"
        )

        self.process_masks(mask_path=train_mask_dir, dest_json=train_json_dir)
        self.process_masks(mask_path=val_mask_dir, dest_json=val_json_dir)
        self.process_masks(mask_path=test_mask_dir, dest_json=test_json_dir)

    """
    this function is js for convience to be used everywhere
    also because some versions dont have the correct paths
    """

    def register_instances(
        self, rgb: bool = False, depth: bool = False, rgd: bool = False
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
