import cv2
import matplotlib.pyplot as plt
import numpy

a = cv2.imread("test.png", cv2.IMREAD_GRAYSCALE)
plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(a, cmap="gray")
blur = cv2.GaussianBlur(a, (3, 3), 0)
_, a = cv2.threshold(blur, 0, 5, cv2.THRESH_BINARY_INV)
b = cv2.morphologyEx(a, cv2.MORPH_CLOSE, numpy.ones((2, 2), numpy.uint8), iterations=2)
plt.subplot(1, 3, 2)
plt.title("Thresholded")
plt.imshow(a, cmap="gray")
plt.subplot(1, 3, 3)
plt.title("Dilated")
plt.imshow(b, cmap="gray")
plt.show()
