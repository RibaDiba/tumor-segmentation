import os, random, cv2
import tqdm
import matplotlib.pyplot as plt

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


def save_images(image_array1, image_array2, folder_path, filename='image'):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for i in range(len(image_array1)):
        img1 = image_array1[i]
        img2 = image_array2[i]

        # Create vertical stack of subplots
        fig, ax = plt.subplots(nrows=2) 

        ax[0].imshow(img1)
        ax[0].set_title("RGB", fontsize=20)  

        ax[1].imshow(img2, cmap='gray')
        ax[1].set_title("Greyscale", fontsize=20)  

        for a in ax:
            a.axis('off')

        fig.tight_layout()  
        fig.savefig(os.path.join(folder_path, f"{filename}_{i}.png"), bbox_inches='tight')  
        plt.close(fig)  


folder_path_rgb = "./images_rgb"
folder_path_greyscale = './images_greyscale'

rgb = read_images_to_array(folder_path_rgb)
greyscale = read_images_to_array(folder_path_greyscale)

save_images(rgb, greyscale, './images_final', filename="image")