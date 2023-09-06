import cv2
import numpy as np
from pprint import pprint


def pxdog(img, size, p, sigma, eps, phi, k=1.6):
    eps /= 255
    g1 = cv2.GaussianBlur(img, (size, size), sigma)
    g2 = cv2.GaussianBlur(img, (size, size), sigma*k)
    d = (1 + p) * g1 - p * g2
    d /= d.max()
    e = 1 + np.tanh(phi*(d-eps))
    e[e >= 1] = 1
    return e * 255


bg_image = cv2.imread("sample/poko.png")
image = bg_image.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = np.array(image, dtype=np.float64)
height, width = image.shape[:2]
image = pxdog(image, 17, 40, 1.4, 0, 15)
_, image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
a = np.array(image, np.uint8)
image = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
image = cv2.bitwise_not(image)
result = cv2.subtract(bg_image, image)

pprint(image)
cv2.imwrite("sample/test3.png", result)
