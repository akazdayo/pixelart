import numpy as np
import cv2
from sklearn.cluster import KMeans


class AI:
    def __init__(self) -> None:
        pass

    def get_color(self, image, color, iter):
        img = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
        img = img.reshape((img.shape[0] * img.shape[1], 3))
        color_count = self.get_color_count(img)
        if color_count < color:
            color = color_count
        cluster = KMeans(n_clusters=color, max_iter=iter)
        cluster.fit(X=img)
        cluster_centers_arr = cluster.cluster_centers_.astype(int, copy=False)
        hexlist = []
        for rgb_arr in list(self.lab2rgb(cluster_centers_arr)):
            hexlist.append("#%02x%02x%02x" % tuple(rgb_arr))
        del img
        del cluster
        del cluster_centers_arr
        return hexlist

    def get_color_count(self, image):
        _, unique_counts = np.unique(image.reshape(-1, 3), axis=0, return_counts=True)
        unique_colors = len(unique_counts)
        return unique_colors

    def lab2rgb(self, image):
        # LAB色空間が入った配列
        lab_array = np.array(image, dtype=np.uint8)

        # 配列の形状を変更（OpenCVの色空間変換関数が3次元配列を必要とするため）
        lab_array = lab_array.reshape(-1, 1, 3)

        # LABからRGBへの変換
        rgb_array = cv2.cvtColor(lab_array, cv2.COLOR_Lab2RGB)  # numpy配列
        rgb_array = rgb_array.reshape(len(rgb_array), 3)
        return rgb_array
