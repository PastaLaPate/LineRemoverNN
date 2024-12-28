import cv2
import matplotlib.pyplot as plt
import numpy


def thresholdImage(img: numpy.ndarray):
    #plt.imshow(img.transpose([1, 2, 0]), cmap='gray')
    #plt.show()
    return cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)[1]


if __name__ == "__main__":
    img = cv2.imread("./test.png", cv2.IMREAD_GRAYSCALE)
    thresholded = thresholdImage(img)
    plt.subplot(1, 2, 1)
    plt.title("In")
    plt.imshow(img, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("Out")
    plt.imshow(thresholded, cmap="gray")
    plt.show()
