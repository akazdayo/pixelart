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
----
追加したいもの
csv追加したら自動的に読み込まれる(完成)
色を確認できるようにする(完成)
すでに変換したものは保存して、早くする(完成)
csvの追加
"""
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import cv2
from PIL import Image
import csv
import os
import pandas as pd


class Converter():
    def __init__(self) -> None:
        self.color_dict = {}
        self.counter = 0
        self.counterr = 0

    def read_csv(self, path):
        with open(path) as f:
            reader = csv.reader(f)
            color = [[int(v) for v in row] for row in reader]
            return color

    def color_change(self, r, g, b, color_pallet):
        if (r, g, b) in self.color_dict:
            self.counter += 1
            return self.color_dict[(r, g, b)]
        # 最も近い色を見つける
        min_distance = float('inf')
        color_name = None
        for color in color_pallet:
            distance = (int(r) - color[0]) ** 2 + (int(g) - color[1]) ** 2 + (int(b) - color[2]) ** 2
            if distance < min_distance:
                min_distance = distance
                color_name = color
        self.color_dict[(r, g, b)] = color_name
        self.counterr += 1
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

    def convert(self, img, option, custom=None):
        w, h = img.shape[:2]
        changed = img.copy()
        # 選択されたcsvファイルを読み込む
        color_pallet = []
        if option != "Custom":
            color_pallet = self.read_csv("./color/"+option+".csv")
        else:
            if custom == [] or custom == None:
                return
            color_pallet = [item[0] for item in custom]

        for height in range(h):
            for width in range(w):
                color = self.color_change(img[width][height][0], img[width][height][1], img[width][height][2], color_pallet)
                changed[width][height][0] = color[0]  # 赤
                changed[width][height][1] = color[1]  # 緑
                changed[width][height][2] = color[2]  # 青
        return changed

    def anime_filter(self, img, K=20):
        # アルファチャンネルを分離
        bgr = img[:, :, :3]
        if len(img[0][0]) == 4:
            alpha = img[:, :, 3]

        # グレースケール変換
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # ぼかしでノイズ低減
        edge = cv2.blur(gray, (3, 3))

        # Cannyアルゴリズムで輪郭抽出
        edge = cv2.Canny(edge, 50, 150, apertureSize=3)

        # 輪郭画像をRGB色空間に変換
        edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

        # 画像の減色処理
        # bgr = np.array(bgr/K, dtype=np.uint8)
        # bgr = np.array(bgr*K, dtype=np.uint8)

        # 差分を返す
        result = cv2.subtract(bgr, edge)

        # アルファチャンネルを結合して返す
        if len(img[0][0]) == 4:
            return np.dstack([result, alpha])
        else:
            return result


class Web():
    def __init__(self) -> None:
        self.col1, self.col2 = None, None
        self.draw_text()

    def file_dir(self):
        filedir = os.listdir("./color")
        for i in range(len(filedir)):
            filedir[i] = filedir[i].replace(".csv", "")
        filedir = tuple(filedir)
        return filedir

    def draw_text(self):
        st.set_page_config(
            page_title="Pixelart-Converter",
            page_icon="🖼️",
            layout="centered",
            initial_sidebar_state="expanded",
        )
        # file_dir = os.listdir("./color")
        fdir = self.file_dir()
        st.title("PixelArt-Converter")
        self.upload = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png', 'webp'])
        self.color = st.selectbox("Select color pallet", fdir)
        self.slider = st.slider('Select ratio', 0.01, 1.0, 0.3, 0.01)
        self.custom = st.checkbox('Custom Pallet')
        self.share()

        self.col1, self.col2 = st.columns(2)
        self.col1.header("Original img")
        self.col2.header("Convert img")
        with st.expander("Custom pallet"):
            self.custom_pallet()

        with st.expander("Experimental Features"):
            self.experimental()
        st.write("Source Code : https://github.com/akazdayo/pixelart")

    def share(self):
        components.html(
            """
<a href="https://twitter.com/share?ref_src=twsrc%5Etfw" class="twitter-share-button" data-show-count="false" data-text="PixelArt-Converter\nFascinating tool to convert images into pixel art!\n" data-url="https://pixelart.streamlit.app" data-hashtags="pixelart,streamlit">Tweet</a><script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
            """,
            height=30,
        )

    def custom_pallet(self):
        st.title("Add pallet")
        _ = st.color_picker('Pick A Color', '#ffffff')
        df, col2 = st.columns(2)
        col2.title("hello")
        df = pd.DataFrame(
            [
                {"R": 255, "G": 0, "B": 0},
                {"R": 0, "G": 255, "B": 0},
                {"R": 0, "G": 0, "B": 255},
                {"R": 0, "G": 0, "B": 0},
                {"R": 255, "G": 255, "B": 255},
            ]
        )
        self.edited_df = st.experimental_data_editor(df, num_rows="dynamic")
        self.rgblist = list()
        for i in range(len(self.edited_df.loc[self.edited_df["R"].keys()])):
            self.rgblist.append([])
            self.rgblist[i].append((self.edited_df.loc[self.edited_df.index[i]]["R"],
                                    self.edited_df.loc[self.edited_df.index[i]]["G"],
                                    self.edited_df.loc[self.edited_df.index[i]]["B"]))

    def experimental(self):
        st.write("""
            The following features are experimental and subject to errors and bugs.
            """)

        self.edge_filter = st.checkbox('Anime Filter')
        self.no_convert = st.checkbox('No Color Convert')

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
            if web.custom:
                img = web.get_image()
                height, width = img.shape[:2]
                cimg = img.copy()
                web.col1.image(img)
                cimg = converter.mosaic(cimg, web.slider)
                if web.no_convert == False:
                    cimg = converter.convert(cimg, "Custom", web.rgblist)
                if web.edge_filter:
                    cimg = converter.anime_filter(cimg)
                web.col2.image(cimg, use_column_width=True)
            else:
                img = web.get_image()
                height, width = img.shape[:2]
                cimg = img.copy()
                web.col1.image(img)
                cimg = converter.mosaic(cimg, web.slider)
                if web.no_convert == False:
                    cimg = converter.convert(cimg, web.color)
                if web.edge_filter:
                    cimg = converter.anime_filter(cimg)
                web.col2.image(cimg, use_column_width=True)
        st.success('Done!', icon="✅")
    elif default == False:
        with st.spinner('Wait for it...'):
            if web.custom:
                img = Image.open("sample/irasutoya.png")
                img = np.array(img)
                height, width = img.shape[:2]
                cimg = img.copy()
                web.col1.image(img)
                cimg = converter.mosaic(cimg, web.slider)
                cimg = converter.convert(cimg, "Custom", web.rgblist)
                if web.edge_filter:
                    cimg = converter.anime_filter(cimg)
                web.col2.image(cimg, use_column_width=True)
            else:
                img = Image.open("sample/irasutoya.png")
                img = np.array(img)
                height, width = img.shape[:2]
                cimg = img.copy()
                web.col1.image(img)
                cimg = converter.mosaic(cimg, web.slider)
                cimg = converter.convert(cimg, web.color)
                if web.edge_filter:
                    cimg = converter.anime_filter(cimg)
                web.col2.image(cimg, use_column_width=True)
        st.success('Done!', icon="✅")
