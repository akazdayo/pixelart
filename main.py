"""
Class Converter
・色の変換
・モザイク処理
・画像を配列に変換


Class Web
・描画(タイトル, アップロードボタン)
・描画(画像)
・画像の取得・Numpy配列に変換
・プログレスバー

main()
"""
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import csv


class Converter():
    def __init__(self) -> None:
        pass

    def read_csv(self, path):
        with open(path) as f:
            reader = csv.reader(f)
            color = [[int(v) for v in row] for row in reader]
            return color

    def color_change(self, rgb):
        # RGB値を取得
        r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
        color_pallet = self.read_csv("./color/pyxel.csv")

        # 最も近い色を見つける
        min_distance = float('inf')
        color_name = None
        for color in color_pallet:
            distance = (r - color[0]) ** 2 + (g - color[1]) ** 2 + (b - color[2]) ** 2
            if distance < min_distance:
                min_distance = distance
                color_name = color
        return color_name

    def mosaic(self, src, ratio=0.1):
        small = cv2.resize(src, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
        return cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

    def store_variable(self, picture, h, w):
        var = []
        for k in range(3):
            array = picture[:, :, k]
            for j in range(h):
                for i in range(w):
                    var.append(str(array[j, i]))
        return var

    def convert(self, img, rgb):
        h, w = img.shape[:2]
        changed = [[0]*(h*w)]*3
        print(str(h) + "," + str(w))
        # print(changed)

        for i in range(h*w):
            color = self.color_change(rgb)
            for j in range(3):  # 0 = R, 1 = G, 2 = B
                changed[j][i] = color[j]
        return changed

    def rgb2img(self, rgb, img):
        h, w = img.shape[:2]
        # RGBリストをNumPy配列に変換する
        array = np.zeros((h, w, 3), dtype=np.uint8)
        for k in range(3):
            for j in range(h):
                for i in range(w):
                    index = j * w + i
                    array[j, i, k] = int(rgb[k][index])
        return array


class Web():
    def __init__(self) -> None:
        self.col1, self.col2 = None, None
        self.draw_text()

    def draw_text(self):
        st.title("PixelArt-Converter")
        self.upload = st.file_uploader("以下からファイルアップロード", type=['jpg', 'png', 'webp'])
        self.col1, self.col2 = st.columns(2)
        self.col1.header("Original img")
        self.col2.header("Convert img")

    def draw_image(self, image):
        self.col2.image(image)

    def update_progress(self):
        pass

    def get_image(self):
        img = Image.open(self.upload)
        img_array = np.array(img)
        return img_array


if __name__ == "__main__":
    web = Web()
    converter = Converter()

    if web.upload != None:
        img = web.get_image()
        height, width = img.shape[:2]
        web.col1.image(img)
        img = converter.mosaic(img, 0.3)
        rgb = converter.store_variable(img, height, width)
        rgb = converter.convert(img, rgb)
        img = converter.rgb2img(rgb, img)
        web.col2.image(img)
