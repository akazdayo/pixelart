import cv2
import numpy as np
from sklearn.cluster import KMeans


class AI:
    def __init__(self) -> None:
        pass

    def get_color(
        self,
        image: np.ndarray,
        color: int,
        iter_count: int,
    ) -> list[str]:
        if image.shape[2] == 4:
            rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        else:
            rgb = image
        lab_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
        flat_lab = lab_img.reshape((lab_img.shape[0] * lab_img.shape[1], 3))

        color_count = self.get_color_count(flat_lab)
        if color_count < color:
            color = color_count

        cluster = KMeans(n_clusters=color, max_iter=iter_count)
        cluster.fit(X=flat_lab)
        cluster_centers_arr = cluster.cluster_centers_.astype(int, copy=False)

        hexlist: list[str] = []
        for rgb_arr in list(self.lab2rgb(cluster_centers_arr)):
            hexlist.append("#%02x%02x%02x" % tuple(rgb_arr))

        del flat_lab
        del cluster
        del cluster_centers_arr
        return hexlist

    def get_color_count(self, image: np.ndarray) -> int:
        _, unique_counts = np.unique(image.reshape(-1, 3), axis=0, return_counts=True)
        unique_colors = len(unique_counts)
        return unique_colors

    def lab2rgb(self, image: np.ndarray) -> np.ndarray:
        lab_array = np.array(image, dtype=np.uint8)
        lab_array = lab_array.reshape(-1, 1, 3)
        rgb_array = cv2.cvtColor(lab_array, cv2.COLOR_Lab2RGB)
        rgb_array = rgb_array.reshape(len(rgb_array), 3)
        return rgb_array
