import csv
import cv2
import numpy as np
import pixelart_modules as pm
from numpy.typing import NDArray
from typing import cast


class Convert:
    def __init__(self) -> None:
        self.color_dict = {}

    def read_csv(self, path):
        with open(path) as f:
            reader = csv.reader(f)
            color = [[int(v) for v in row] for row in reader]
            return color

    def convert(self, img, option, custom=None) -> NDArray[np.uint64]:
        # 選択されたcsvファイルを読み込む
        color_palette = []
        if option != "Custom":
            color_palette = self.read_csv("./color/" + option + ".csv")
        else:
            if not custom:
                raise ValueError("Custom Palette is empty.")
            color_palette = custom

        # convert関数はRustに移しました。
        # https://github.com/akazdayo/pixelart-modules
        changed = cast(
            NDArray[np.uint64],
            pm.convert(img, np.array(color_palette, dtype=np.uint64)), # type: ignore
        )
        return changed

    def resize_image(self, image):
        img_size = image.shape[0] * image.shape[1]
        # 画像をFull HDよりも小さくする
        ratio = (img_size / 2073600) ** 0.5
        new_height = int(image.shape[0] / ratio)
        new_width = int(image.shape[1] / ratio)
        result = cv2.resize(image, (new_width, new_height))
        return result

    def delete_alpha(self, image):
        # 画像がアルファチャンネルを持っているか確認
        if image.shape[2] == 4:
            b = image[:, :, 0]
            g = image[:, :, 1]
            r = image[:, :, 2]
            a = image[:, :, 3]
            conv_a = a.copy()
            for i, x in enumerate(a):
                for j, y in enumerate(x):
                    if y != 0:
                        conv_a[i][j] = 255

            merged = cv2.merge([b, g, r, conv_a])
            return merged
        else:
            # アルファチャンネルがない場合はそのまま返す
            return image

    def delete_transparent_color(self, image):
        # 画像がアルファチャンネルを持っているか確認
        if image.shape[2] == 4:
            b = image[:, :, 0]
            g = image[:, :, 1]
            r = image[:, :, 2]
            a = image[:, :, 3]
            for i, x in enumerate(a):
                for j, y in enumerate(x):
                    if y == 0:
                        b[i][j] = 255
                        g[i][j] = 255
                        r[i][j] = 255

            merged = cv2.merge([b, g, r, a])
            return merged
        else:
            # アルファチャンネルがない場合はそのまま返す
            return image
