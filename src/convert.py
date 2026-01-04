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
            pm.convert(img, np.array(color_palette, dtype=np.uint64)),  # type: ignore
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
            bgr = image[:, :, :3]
            a = image[:, :, 3]
            # ベクトル化: 0以外の値を255に変換
            conv_a = np.where(a != 0, 255, a).astype(np.uint8)
            merged = cv2.merge([bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2], conv_a])
            return merged
        else:
            # アルファチャンネルがない場合はそのまま返す
            return image

    def delete_transparent_color(self, image):
        # 画像がアルファチャンネルを持っているか確認
        if image.shape[2] == 4:
            bgr = image[:, :, :3].copy()
            a = image[:, :, 3]
            # ベクトル化: アルファが0のピクセルを白(255)に変換
            mask = a == 0
            bgr[mask] = [255, 255, 255]
            merged = cv2.merge([bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2], a])
            return merged
        else:
            # アルファチャンネルがない場合はそのまま返す
            return image
