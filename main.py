import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import cv2
from PIL import Image
import csv
import os
import pandas as pd
from sklearn.cluster import KMeans
import warnings
import gc

warnings.simplefilter('ignore')


class Converter():
    def __init__(self) -> None:
        self.color_dict = {}

    def read_csv(self, path):
        with open(path) as f:
            reader = csv.reader(f)
            color = [[int(v) for v in row] for row in reader]
            return color

    def color_change(self, r, g, b, color_palette):
        if (r, g, b) in self.color_dict:
            return self.color_dict[(r, g, b)]
        # 最も近い色を見つける
        min_distance = float('inf')
        color_name = None
        for color in color_palette:
            distance = (int(r) - color[0]) ** 2 + (int(g) -
                                                   color[1]) ** 2 + (int(b) - color[2]) ** 2
            if distance < min_distance:
                min_distance = distance
                color_name = color
        self.color_dict[(r, g, b)] = color_name
        return color_name

    def mosaic(self, img, ratio=0.1):
        small = cv2.resize(img, None, fx=ratio, fy=ratio,
                           interpolation=cv2.INTER_NEAREST)
        return cv2.resize(small, img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        # return small

    def convert(self, img, option, custom=None):
        w, h = img.shape[:2]
        changed = img.copy()
        # 選択されたcsvファイルを読み込む
        color_palette = []
        if option != "Custom":
            color_palette = self.read_csv("./color/"+option+".csv")
        else:
            if custom == [] or custom == None:
                return
            color_palette = custom

        for height in range(h):
            for width in range(w):
                color = self.color_change(
                    img[width][height][0], img[width][height][1], img[width][height][2], color_palette)
                changed[width][height][0] = color[0]  # 赤
                changed[width][height][1] = color[1]  # 緑
                changed[width][height][2] = color[2]  # 青
        return changed

    def anime_filter(self, img, th1=50, th2=150):
        # アルファチャンネルを分離
        bgr = img[:, :, :3]
        if len(img[0][0]) == 4:
            alpha = img[:, :, 3]

        # グレースケール変換
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # ぼかしでノイズ低減
        edge = cv2.blur(gray, (3, 3))

        # Cannyアルゴリズムで輪郭抽出
        edge = cv2.Canny(edge, th1, th2, apertureSize=3)

        # 輪郭画像をRGB色空間に変換
        edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

        # 差分を返す
        result = cv2.subtract(bgr, edge)

        # アルファチャンネルを結合して返す
        if len(img[0][0]) == 4:
            return np.dstack([result, alpha])
        else:
            return result

    def pxdog(self, img, size, p, sigma, eps, phi, k=1.6):
        eps /= 255
        g1 = cv2.GaussianBlur(img, (size, size), sigma)
        g2 = cv2.GaussianBlur(img, (size, size), sigma*k)
        d = (1 + p) * g1 - p * g2
        d /= d.max()
        e = 1 + np.tanh(phi*(d-eps))
        e[e >= 1] = 1
        return e * 255

    def new_anime_filter(self, img):
        # アルファチャンネルを分離
        bg_image = img[:, :, :3]
        if len(img[0][0]) == 4:
            alpha = img[:, :, 3]
        image = bg_image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.array(image, dtype=np.float64)
        image = self.pxdog(image, 17, 40, 1.4, 0, 15)
        _, image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
        a = np.array(image, np.uint8)
        image = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
        image = cv2.bitwise_not(image)
        result = cv2.subtract(bg_image, image)
        # アルファチャンネルを結合して返す
        if len(img[0][0]) == 4:
            return np.dstack([result, alpha])
        else:
            return result

    def decreaseColor(self, img):
        dst = img.copy()

        idx = np.where((0 <= img) & (64 > img))
        dst[idx] = 32
        idx = np.where((64 <= img) & (128 > img))
        dst[idx] = 96
        idx = np.where((128 <= img) & (192 > img))
        dst[idx] = 160
        idx = np.where((192 <= img) & (256 > img))
        dst[idx] = 224

        return dst

    def resize_image(self, img):
        img_size = img.shape[0] * img.shape[1]
        if img_size > 2073600:
            ratio = (img_size / 2073600) ** 0.5
            new_height = int(img.shape[0] / ratio)
            new_width = int(img.shape[1] / ratio)
            img = cv2.resize(img, (new_width, new_height))
        return img

    def half_img(self, img):
        h, w = img.shape[:2]
        cx = 0
        cy = 0
        completed = []
        for _ in range(2):
            split_pic = img[cy:cy+int(h), cx:cx+int(w/2), :]
            completed.append(split_pic)
            cy = cy+int(h)
            cy = 0
            cx = cx+int(w/2)
        return completed

    def combine_img(self, img1, img2):
        # 画像の高さと幅を取得
        height, width, _ = img1.shape

        # 新しい画像を作成する
        new_img = np.zeros((height, width*2, 3), np.uint8)

        # RGBからBGRに変換
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

        # 画像を結合する
        new_img[:, :width] = img1
        new_img[:, width:] = cv2.resize(img2, (width, height))

        new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)

        return new_img

    def add_number(self, img):
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
        fdir = self.file_dir()
        st.title("PixelArt-Converter")
        self.message = st.empty()
        self.use_ai = False
        self.upload = st.file_uploader(
            "Upload Image", type=['jpg', 'jpeg', 'png', 'webp', 'jfif'])
        self.color = st.selectbox(
            "Select color Palette", fdir, disabled=self.use_ai)
        self.slider = st.slider('Select ratio', 0.01, 1.0, 0.3, 0.01)
        self.custom = st.checkbox('Custom Palette')
        self.use_ai = st.checkbox('Use AI', True)
        self.share()

        self.col1, self.col2 = st.columns(2)
        self.col1.header("Original img")
        self.col2.header("Convert img")
        self.now = st.empty()

        with st.expander("More Options", True):
            self.more_options()
        with st.expander("Custom Palette"):
            self.custom_palette()
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

    def hex_to_rgb(self, hex_code):
        hex_code = hex_code.replace("#", "")
        r = int(hex_code[0:2], 16)
        g = int(hex_code[2:4], 16)
        b = int(hex_code[4:6], 16)
        return [r, g, b]

    def hex_to_rgblist(self, hex_list):
        rgb_values = []
        for hex_code in hex_list:
            if hex_code != None:
                rgb_values.append(self.hex_to_rgb(hex_code[1:]))
        return rgb_values

    def custom_palette(self, df=pd.DataFrame(
        [
            {"hex": "#FF0000"},
            {"hex": "#00FF00"},
            {"hex": "#0000FF"},
            {"hex": "#FFFFFF"},
            {"hex": "#000000"},
        ]
    )):
        st.title("Add Palette")
        # _ = st.color_picker('Pick A Color', '#ffffff')
        col1, col2 = st.columns(2)
        self.edited_df = col1.data_editor(df, num_rows="dynamic")
        self.rgblist = list()
        for i in range(len(self.edited_df.loc[self.edited_df["hex"].keys()])):
            self.rgblist.append([])
            self.rgblist[i].append(
                (self.edited_df.loc[self.edited_df.index[i]]["hex"]))
        self.show_custom(col2)

    def show_custom(self, col):
        color_palette = [item[0] for item in self.rgblist]
        color_palette = self.hex_to_rgblist(color_palette)
        rgb = []
        for i in color_palette:
            color = np.zeros((50, 50, 3), dtype=np.uint8)
            color[:, :] = [i[0], i[1], i[2]]
            col.image(color)
            rgb.append(i)
        self.rgblist = rgb

    def experimental(self):
        st.write("""
            The following features are experimental and subject to errors and bugs.
            """)
        st.title("AI")
        self.color_number = st.slider(
            "AI Color", 1, 20, 8, 1, help="Number of colors")
        self.ai_iter = st.slider("AI Number of attempts", 1, 3000, 150, 1,
                                 help="Maximum number of iterations of the k-means algorithm for a single run.")

    def more_options(self):
        st.title("Anime Filter")

        st.subheader("LoG Filter")
        px_col_log, smooth_col_log = st.columns(2)
        self.smooth_log_filter = px_col_log.checkbox('Smooth LoG Filter')
        self.px_log_filter = smooth_col_log.checkbox('Pixel LoG Filter', True)

        st.subheader("Canny Filter", help="deprecated")
        smooth_col_canny, px_col_canny,  = st.columns(2)

        smooth_col_canny.subheader('Smooth Edge')
        self.smooth_canny_filter = smooth_col_canny.checkbox('Smooth Canny Filter')
        self.anime_th1 = smooth_col_canny.slider('Select threhsold1(minVal)', 0.0, 500.0, 0.0, 5.0,
                                                 help="The smaller the value, the more edges there are.(using cv2.Canny)", disabled=not self.smooth_canny_filter)
        self.anime_th2 = smooth_col_canny.slider('Select threhsold2(maxVal)', 0.0, 500.0, 0.0, 5.0,
                                                 help="The smaller the value, the more edges there are.(using cv2.Canny)", disabled=not self.smooth_canny_filter)

        px_col_canny.subheader('Pixel Edge')
        self.pixel_canny_edge = px_col_canny.checkbox("Pixel Canny Filter")
        self.px_th1 = px_col_canny.slider('Select Pixel threhsold1(minVal)', 0.0, 500.0, 100.0, 5.0,
                                          help="The smaller the value, the more edges there are.(using cv2.Canny)", disabled=not self.pixel_canny_edge)
        self.px_th2 = px_col_canny.slider('Select Pixel threhsold2(maxVal)', 0.0, 500.0, 100.0, 5.0,
                                          help="The smaller the value, the more edges there are.(using cv2.Canny)", disabled=not self.pixel_canny_edge)

        st.title("Convert Setting")
        self.no_convert = st.checkbox('No Color Convert')
        self.decreaseColor = st.checkbox('decrease Color')

    def get_image(self, upload):
        img = Image.open(upload)
        img_array = np.array(img)
        return img_array


@st.cache_resource
def getMainColor(img, color, iter):
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    img = img.reshape(
        (img.shape[0] * img.shape[1], 3))
    cluster = KMeans(n_clusters=color, max_iter=iter)
    cluster.fit(X=img)
    cluster_centers_arr = cluster.cluster_centers_.astype(
        int, copy=False)
    hexlist = []

    for rgb_arr in cluster_centers_arr:
        hexlist.append('#%02x%02x%02x' % tuple(rgb_arr))
    del img
    del cluster
    del cluster_centers_arr
    return hexlist


if __name__ == "__main__":
    web = Web()
    converter = Converter()
    if web.upload != None:
        img = web.get_image(web.upload)
    else:
        img = web.get_image("sample/irasutoya.png")
    height, width = img.shape[:2]
    if height*width < 2073600:
        pass
    else:
        img = converter.resize_image(img)
        web.message.warning("""
The size of the image has been reduced because the file size is too large.\n
Image size is reduced if the number of pixels exceeds 2K (2,073,600).
        """)
    cimg = img.copy()
    # del img
    del web.upload
    web.col1.image(cimg)
    if web.pixel_canny_edge:
        web.now.write("### Pixel Edge in progress")
        cimg = converter.anime_filter(cimg, web.px_th1, web.px_th2)
    if web.px_log_filter:
        web.now.write("### Pixel Edge in progress")
        cimg = converter.new_anime_filter(cimg)
    web.now.write("### Now mosaic")
    cimg = converter.mosaic(cimg, web.slider)
    if web.no_convert == False:
        if web.custom or web.use_ai:
            if web.use_ai:
                web.now.write("### AI Palette in progress")
                ai_color = getMainColor(
                    cimg, web.color_number, web.ai_iter)
                web.custom_palette(pd.DataFrame(
                    {"hex": c} for c in ai_color))
            web.now.write("### Color Convert in progress")
            cimg = converter.convert(cimg, "Custom", web.rgblist)
        else:
            web.now.write("### Color Convert in progress")
            cimg = converter.convert(cimg, web.color)
    if web.decreaseColor:
        web.now.write("### Decrease Color in progress")
        cimg = converter.decreaseColor(cimg)
    if web.smooth_canny_filter:
        web.now.write("### Edge filter in progress")
        cimg = converter.anime_filter(cimg, web.anime_th1, web.anime_th2)
    if web.smooth_log_filter:
        web.now.write("### Edge filter in progress")
        cimg = converter.new_anime_filter(cimg)
    web.col2.image(cimg, use_column_width=True)
    web.now.write("")
    del converter.color_dict
    gc.collect()
