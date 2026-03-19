import matplotlib.pyplot as plt


class UtilsMixin:

    # sanity check
    def check(self):
        f, ax = plt.subplots(1, 2)
        ax[0].imshow(self.images[0])
        ax[0].set_title("Image")
        ax[1].imshow(self.masks[0])
        ax[1].set_title("Mask")
        plt.show()
