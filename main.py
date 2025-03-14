from infer import loadBestModel, splitAndProcessImg, reconstruct_image
from matplotlib import pyplot as plt
import cv2
import skimage.filters as filters
import numpy as np


def UniformLighting(
    image, debug=True
):  # See : https://stackoverflow.com/questions/63612617/how-do-you-remove-the-unnecessary-blackness-from-binary-images-mainly-caused-by
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (95, 95), 0)
    division = cv2.divide(gray, smooth, scale=255)
    sharp = filters.unsharp_mask(division, radius=1.5, amount=1.5, preserve_range=False)
    sharp = (255 * sharp).clip(0, 255).astype(np.uint8)
    thresh = cv2.threshold(sharp, 0, 255, cv2.THRESH_OTSU)[1]
    # if debug:
    #    showImage([(gray, 'Gray'), (smooth, 'Smooth'), (sharp, 'Sharp'), (thresh, 'Threshold')])
    return thresh


if __name__ == "__main__":
    network = loadBestModel()
    network.eval()
    network.to("cuda")

    img = cv2.imread("a.jpg")
    imga = UniformLighting(img)
    img_splitted = splitAndProcessImg(imga)
    reconstructed_image = reconstruct_image(img_splitted, img.shape[0], img.shape[1])
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(img)

    plt.subplot(1, 3, 2)
    plt.title("Preprocessed Image")
    plt.imshow(imga, cmap="gray")

    plt.subplot(1, 3, 3)
    plt.title("Reconstructed Image")
    plt.imshow(reconstructed_image, cmap="gray")
    plt.show()
