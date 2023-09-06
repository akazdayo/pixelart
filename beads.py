import cv2
from PIL import Image
import numpy as np


def get_image(upload):
    img = Image.open(upload)
    img_array = np.array(img)
    return img_array


def add_number(img):
    w, h = img.shape[:2]
    converted = img.copy()
    old_rgb = [converted[0][0][0], converted[0][0][1], converted[0][0][2]]
    for height in range(h):
        old_rgb = [converted[0][height][0], converted[0][height][1], converted[0][height][2]]
        for width in range(w):
            rgb = [converted[width][height][0], converted[width][height][1], converted[width][height][2]]
            if old_rgb != rgb:
                converted[width][height][0], converted[width][height][1], converted[width][height][2] = 0, 0, 0
            old_rgb = rgb
    for width in range(w):
        old_rgb = [converted[width][0][0], converted[width][0][1], converted[width][0][2]]
        for height in range(h):
            rgb = [converted[width][height][0], converted[width][height][1], converted[width][height][2]]
            if old_rgb != rgb:
                converted[width][height][0], converted[width][height][1], converted[width][height][2] = 0, 0, 0
            old_rgb = rgb
    return converted


if __name__ == "__main__":
    img = cv2.imread("sample/maikura.jpg")
    # img = get_image(img)
    img = add_number(img)
    # cv2.imshow("test", img)
    # cv2.waitKey()
    cv2.imwrite("sample/test2.bmp", img)
