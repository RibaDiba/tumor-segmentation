import os, cv2, json, glob

# note: this will have to be changed if we add a non-tumor class
category_ids = {
    "Tumor": 0
}

MASK_EXT = 'png'
ORIGINAL_EXT = 'png'
image_id = 0
annotation_id = 0

def process_masks(mask_path, dest_json):
    global image_id, annotation_id
    image_id = 0
    annotation_id = 0

    coco_format = {
        "info": {},
        "licenses": [],
        "images": [],
        "categories": [{"id": value, "name": key, "supercategory": key} for key, value in category_ids.items()],
        "annotations": [],
    }

    coco_format["images"], coco_format["annotations"], annotation_cnt = images_annotations_info(mask_path)

    with open(dest_json, "w") as outfile:
        json.dump(coco_format, outfile, sort_keys=True, indent=4)

    print("Created %d annotations for images in folder: %s" % (annotation_cnt, mask_path))

def images_annotations_info(maskpath):

    global image_id, annotation_id
    annotations = []
    images = []

    for category in category_ids.keys():
        for mask_image in glob.glob(os.path.join(maskpath, category, f'*.{MASK_EXT}')):
            original_file_name = f'{os.path.basename(mask_image).split(".")[0]}.{ORIGINAL_EXT}'
            mask_image_open = cv2.imread(mask_image)
            
            height, width, _ = mask_image_open.shape

            if original_file_name not in map(lambda img: img['file_name'], images):
                image = {
                    "id": image_id + 1,
                    "width": 640,
                    "height": 614,
                    "file_name": original_file_name,
                }
                images.append(image)
                image_id += 1
            else:
                image = [element for element in images if element['file_name'] == original_file_name][0]

            gray = cv2.cvtColor(mask_image_open, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

            for contour in contours:
                bbox = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                segmentation = contour.flatten().tolist()

                annotation = {
                    "iscrowd": 0,
                    "id": annotation_id,
                    "image_id": image['id'],
                    "category_id": category_ids[category],
                    "bbox": bbox,
                    "area": area,
                    "segmentation": [segmentation],
                }

                if area > 0:
                    annotations.append(annotation)
                    annotation_id += 1

    return images, annotations, annotation_id

def get_coco_rgb(coco_json_path_rgb):

    train_mask_path = os.path.join(coco_json_path_rgb, "train/masks")
    train_json_path = os.path.join(coco_json_path_rgb, "train/images/train.json")

    val_mask_path = os.path.join(coco_json_path_rgb, "val/maks")
    val_json_path = os.path.join(coco_json_path_rgb, "val/images/val.json")

    test_mask_path = os.path.join(coco_json_path_rgb, 'test/masks')
    test_json_path = os.path.join(coco_json_path_rgb, 'test/images/test.json')

    process_masks(train_mask_path, train_json_path)
    process_masks(val_mask_path, val_json_path)
    process_masks(test_mask_path, test_json_path)


    print('Done creating COCO JSON annotations for all files')

def get_coco_grayscale(coco_json_path_grayscale):

    train_mask_path = os.path.join(coco_json_path_grayscale, "train/masks")
    train_json_path = os.path.join(coco_json_path_grayscale, "train/images/train.json")

    val_mask_path = os.path.join(coco_json_path_grayscale, "val/maks")
    val_json_path = os.path.join(coco_json_path_grayscale, "val/images/val.json")

    test_mask_path = os.path.join(coco_json_path_grayscale, 'test/masks')
    test_json_path = os.path.join(coco_json_path_grayscale, 'test/images/test.json')

    process_masks(train_mask_path, train_json_path)
    process_masks(val_mask_path, val_json_path)
    process_masks(test_mask_path, test_json_path)

    print('Done creating COCO JSON annotations for all files')

def get_coco_rgbd(coco_json_path_rgbd):

    train_mask_path = os.path.join(coco_json_path_rgbd, "train/masks")
    train_json_path = os.path.join(coco_json_path_rgbd, "train/images/train.json")

    val_mask_path = os.path.join(coco_json_path_rgbd, "val/maks")
    val_json_path = os.path.join(coco_json_path_rgbd, "val/images/val.json")

    test_mask_path = os.path.join(coco_json_path_rgbd, 'test/masks')
    test_json_path = os.path.join(coco_json_path_rgbd, 'test/images/test.json')

    process_masks(train_mask_path, train_json_path)
    process_masks(val_mask_path, val_json_path)
    process_masks(test_mask_path, test_json_path)

    print('Done creating COCO JSON annotations for all files')

