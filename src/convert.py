import csv
import cv2
import numpy as np
import pixelart_modules as pm
from numpy.typing import NDArray
from typing import cast
from src.ai import AI


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
        color_palette = []  # [[r, g, b], [r, g, b], ...]
        if option != "Custom":
            color_palette = self.read_csv("./color/" + option + ".csv")
        else:
            if not custom:
                raise ValueError("Custom Palette is empty.")
            color_palette = custom
        print(img.dtype)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
        # color_paletteをLAB色空間に変換
        color_palette_array = np.array([color_palette], dtype=np.uint8)
        color_palette_lab = cv2.cvtColor(color_palette_array, cv2.COLOR_RGB2Lab)[0]

        # LAB色空間の値をuint64として扱う
        color_palette_uint64 = np.array(color_palette_lab, dtype=np.uint64)

        # convert関数はRustに移しました。
        # https://github.com/akazdayo/pixelart-modules
        result = cast(
            NDArray[np.uint64],
            pm.convert(img, color_palette_uint64),  # type: ignore
        )

        # 結果をuint64型に変換してRGBに戻す
        changed = AI.lab2rgb(result)
        changed = cv2.cvtColor(changed, cv2.COLOR_RGB2BGR)
        changed = np.array(changed, dtype=np.uint64)  # 最終的な型を確保

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
