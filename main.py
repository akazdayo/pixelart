"""
Class Converter
・色の変換
・モザイク処理


Class Web
・描画(タイトル, アップロードボタン)
・画像の取得・Numpy配列に変換
・プログレスバー

main
・Converter呼び出し
・Web呼び出し
・画像を配列に変換
・変換後の画像を描画
"""
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import csv
import time


class Converter():
    def __init__(self) -> None:
        color_dict = []

    def read_csv(self, path):
        with open(path) as f:
            reader = csv.reader(f)
            color = [[int(v) for v in row] for row in reader]
            return color

    def color_change(self, r, g, b, option):
        # 選択されたcsvファイルを読み込む
        color_pallet = self.read_csv("./color/"+option+".csv")
        # 最も近い色を見つける
        min_distance = float('inf')
        color_name = None
        for color in color_pallet:
            distance = (int(r) - color[0]) ** 2 + (int(g) - color[1]) ** 2 + (int(b) - color[2]) ** 2
            if distance < min_distance:
                min_distance = distance
                color_name = color
        return color_name

    def mosaic(self, img, ratio=0.1):
        """# mosaic

        Args:
            img (_type_): _description_
            ratio (float, optional): _description_. Defaults to 0.1.

        Returns:
            _type_: _description_
        """
        small = cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
        return cv2.resize(small, img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

    def convert(self, img, option):
        w, h = img.shape[:2]
        changed = img.copy()
        for height in range(h):
            for width in range(w):
                color = self.color_change(img[width][height][0], img[width][height][1], img[width][height][2], option)
                changed[width][height][0] = color[0]  # 赤
                changed[width][height][1] = color[1]  # 緑
                changed[width][height][2] = color[2]  # 青
        return changed


class Web():
    def __init__(self) -> None:
        self.col1, self.col2 = None, None
        self.draw_text()

    def draw_text(self):
        st.set_page_config(
            page_title="Pixelart-Converter",
            page_icon="🖼️",
            layout="centered",
            initial_sidebar_state="expanded",
        )
        st.title("PixelArt-Converter")
        self.upload = st.file_uploader("Upload Image", type=['jpg', 'png', 'webp'])
        self.color = st.selectbox("Select color pallet", ('pyxel', 'pastel', 'warm', 'cold', 'rainbow', 'gold', 'pale'))
        self.slider = st.slider('Select ratio', 0.1, 1.0, 0.3, 0.05)

        self.col1, self.col2 = st.columns(2)
        self.col1.header("Original img")
        self.col2.header("Convert img")
        st.write("Source Code : https://github.com/akazdayo/pixelart")

    def update_progress(self):
        pass

    def get_image(self):
        img = Image.open(self.upload)
        img_array = np.array(img)
        return img_array


if __name__ == "__main__":
    web = Web()
    converter = Converter()
    default = False  # サンプル画像を一度のみ表示

    if web.upload != None:
        with st.spinner('Wait for it...'):
            img = web.get_image()
            height, width = img.shape[:2]
            cimg = img.copy()
            web.col1.image(img)
            cimg = converter.mosaic(cimg, web.slider)
            cimg = converter.convert(cimg, web.color)
            web.col2.image(cimg)
        st.success('Done!')
        st.balloons()
    elif default == False:
        with st.spinner('Wait for it...'):
            img = Image.open("./sample/irasutoya.png")
            img = np.array(img)
            default = True
            height, width = img.shape[:2]
            cimg = img.copy()
            web.col1.image(img)
            cimg = converter.mosaic(cimg, web.slider)
            cimg = converter.convert(cimg, web.color)
            web.col2.image(cimg)
        st.success('Done!')
        st.balloons()
