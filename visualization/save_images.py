import os, random, cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

def read_images_to_array(folder_path):

  image_array = []
  # Get a sorted list of filenames
  filenames = sorted(os.listdir(folder_path))
  for filename in filenames:
    if filename.endswith(".jpg") or filename.endswith(".png"):
      img_path = os.path.join(folder_path, filename)
      img = cv2.imread(img_path)

      if img is not None:
        image_array.append(img)

  return image_array

def save_images(folder_path1, folder_path2, folder_path3, title1, title2, title3, folder_save_path):
  image_array1 = read_images_to_array(folder_path1)
  image_array2 = read_images_to_array(folder_path2)
  image_array3 = read_images_to_array(folder_path3)

  for i in tqdm(range(len(image_array1)), desc="Saving Images"):

     image1 = image_array1[i]
     image2 = image_array2[i]
     image3 = image_array3[i]

     fig, ax = plt.subplots(nrows=3, figsiz=[7,7])

     ax[0].imshow(image1)
     ax[0].set_title(title1)

     ax[1].imshow(image2)
     ax[1].set_title(title2)

     ax[2].imshow(image3)
     ax[2].set_title(title3)

     for a in ax:
       a.axis("off")

     fig.savefig(os.path.join(folder_save_path, f"image_{i}.png"))
     fig.close()




