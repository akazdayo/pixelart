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
        color_dict = []

    def read_csv(self, path):
        with open(path) as f:
            reader = csv.reader(f)
            color = [[int(v) for v in row] for row in reader]
            return color

    def color_change(self, r, g, b, option):
        # print("R : "+r+"\nG : "+g+"\nB : "+b)
        # RGB値を取得
        # r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
        """
        if option == "Pyxel":
            color_pallet = self.read_csv("./color/pyxel.csv")
        elif option == "Pastel":
            color_pallet = self.read_csv("./color/pastel.csv")
        elif option == "Warm":
            color_pallet = self.read_csv("./color/warm.csv")
        elif option == "Cold":
            color_pallet = self.read_csv("./color/cold.csv")
        elif option == "Rainbow":
            color_pallet = self.read_csv("./color/rainbow.csv")
        """
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

    def mosaic(self, src, ratio=0.1):
        small = cv2.resize(src, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
        return cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

    def store_variable(self, picture, h, w):
        # 3つの色チャンネルごとに処理
        var = [[[0]*width]*height]*3
        for k in range(3):
            array = picture[:, :, k]
            # 画像の高さと幅でループ
            for i in range(h):
                for j in range(w):
                    # 各ピクセルの値を文字列に変換して格納
                    var[k][i][j] = (str(array[i, j]))
        return var

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

    def rgb2img(self, rgb, img):
        h, w = img.shape[:2]  # 画像の高さと幅を取得
        array = np.zeros((h, w, 3), dtype=np.uint8)  # 高さと幅と3つのチャンネル（RGB）を持つNumPy配列を作成
        for color in range(3):  # RGBチャンネルごとに処理
            for height in range(h):  # 高さ方向に処理
                for width in range(w):  # 幅方向に処理
                    # index = height * w + width
                    array[height, width, color] = int(rgb[color][height][width])  # RGBリストから値を取り出し、NumPy配列に代入
        return array  # 作成したNumPy配列を返す

    def convert_rgb_list_to_image(self, rgb_list, img):
        """
        RGBのリストから画像に変換する関数
        :param rgb_list: RGBのリスト
        :return: 画像
        """
        rgb_array = np.transpose(np.array(rgb_list), (1, 2, 0))  # リストをNumpy配列に変換し、軸を入れ替える
        print(rgb_array.shape)
        image = Image.fromarray(np.uint8(rgb_array))  # Numpy配列から画像に変換
        return image  # 画像を返す


class Web():
    def __init__(self) -> None:
        self.col1, self.col2 = None, None
        self.draw_text()

    def draw_text(self):
        st.title("PixelArt-Converter")
        self.upload = st.file_uploader("Upload Image", type=['jpg', 'png', 'webp'])
        self.color = st.selectbox("Select color pallet", ('pyxel', 'pastel', 'warm', 'cold', 'rainbow', 'gold', 'pale'))
        self.slider = st.slider('Select ratio', 0.1, 1.0, 0.5, 0.05)
        self.col1, self.col2 = st.columns(2)
        self.col1.header("Original img")
        self.col2.header("Convert img")
        st.write("Source Code : https://github.com/akazdayo/pixelart")

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
    else:
        img = Image.open("./sample/irasutoya.png")
        img = np.array(img)
    height, width = img.shape[:2]
    # rgb = [[[0]*width]*height]*3
    cimg = img.copy()
    web.col1.image(img)
    cimg = converter.mosaic(cimg, web.slider)
    # rgb = converter.store_variable(img, height, width)
    cimg = converter.convert(cimg, web.color)
    # img = converter.convert_rgb_list_to_image(rgb, img)
    web.col2.image(cimg)
