from PIL import Image
import numpy as np
import pandas as pd
import streamlit as st
import cv2
import math
import datetime
import threading


class Converter():
    def __init__(self) -> None:
        # self.file_name = 'data/img/dango.png'
        # self.pic = cv2.imread(self.file_name)
        # self.h, self.w = self.pic.shape[:2]
        # self.pic = cv2.cvtColor(self.pic, cv2.COLOR_BGR2RGB)
        # self.pic = self.mosaic(src=self.pic)
        # self.RGB = [[], [], []]
        pass

    def rgb_to_hsv(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return hsv

    def detect_color(self, rgb):
        # RGB値を取得
        r, g, b = rgb[0], rgb[1], rgb[2]

        # 色名を定義するためのデータベース
        color_names = {
            (255, 127, 127): (255, 0, 0),  # 赤
            (255, 191, 127): (255, 165, 0),  # オレンジ
            (255, 255, 127): (255, 255, 0),  # 黄色
            (127, 255, 127): (0, 128, 0),  # 緑
            (127, 191, 255): (0, 0, 255),  # 青
            (127, 127, 255): (128, 0, 128),  # 紫
            (0, 0, 0): (0, 0, 0),  # 黒
            (255, 255, 255): (255, 255, 255),  # 白
            (128, 128, 128): (128, 128, 128)  # 灰色
        }
        pyxel_color = [
            (0, 0, 0),
            (43, 51, 95),
            (126, 32, 114),
            (25, 149, 156),
            (139, 72, 82),
            (57, 92, 152),
            (169, 193, 255),
            (238, 238, 238),
            (212, 24, 108),
            (211, 132, 65),
            (233, 195, 91),
            (112, 198, 169),
            (118, 150, 222),
            (163, 163, 163),
            (255, 151, 152),
            (237, 199, 176)
        ]
        """
        color_names = {
            (255, 127, 127): (255, 0, 0),  # 赤
            (255, 191, 127): (255, 165, 0),  # オレンジ
            (255, 255, 127): (255, 255, 0),  # 黄色
            (127, 255, 127): (0, 128, 0),  # 緑
            (127, 191, 255): (0, 0, 255),  # 青
            (127, 127, 255): (128, 0, 128),  # 紫
            (0, 0, 0): (0, 0, 0),  # 黒
            (255, 255, 255): (255, 255, 255),  # 白
            (128, 128, 128): (128, 128, 128)  # 灰色
        }
        """

        # 最も近い色を見つける
        min_distance = float('inf')
        color_name = None
        for color in pyxel_color:
            distance = (r - color[0]) ** 2 + (g - color[1]) ** 2 + (b - color[2]) ** 2
            if distance < min_distance:
                min_distance = distance
                color_name = color
        return color_name

    def mosaic(self, src, ratio=0.1):
        small = cv2.resize(src, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
        return cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

    def generate(self, img, count):
        color = self.detect_color(rgb=[int(img[0][count]), int(img[1][count]), int(img[2][count])])
        img[0][count] = color[0]
        img[1][count] = color[1]
        img[2][count] = color[2]
        return img

    def store_variable(self, picture, variables, h, w):
        for k in range(3):
            array = picture[:, :, k]
            for j in range(h):
                for i in range(w):
                    variables[k].append(str(array[j, i]))
        return variables

    def image_save(self, img, h, w):
        # RGBリストをNumPy配列に変換する
        array = np.zeros((h, w, 3), dtype=np.uint8)
        for k in range(3):
            for j in range(h):
                for i in range(w):
                    index = j * w + i
                    array[j, i, k] = int(img[k][index])

        # NumPy配列からPillowのImageオブジェクトを作成する
        image = Image.fromarray(array)

        # 画像を保存する
        # image.save('./data/restored.png')
        return array


class Web():
    def __init__(self) -> None:
        self.converter = Converter()
        st.title("PixelArt-Converter")
        # self.my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        # アップローダー
        self.my_upload = st.file_uploader("以下からファイルアップロード", type=['jpg', 'png', 'webp'])
        # 解像度
        self.number = st.number_input('Insert a number', min_value=0.01, max_value=1.00, value=0.50)
        # カラム設定
        self.col1, self.col2 = st.columns(2)

        self.col1.header("Original image")
        self.col2.header("convert image")
        self.speed_chart = []
        # self.chart_thread = threading.Thread(target=self.progress_chart)

    def progressbar(self, h, w, image):
        i = 0
        process = st.empty()
        my_bar = st.progress(0)
        my_bar.text(i)
        percent_complete = 0
        start_time = datetime.datetime.now()
        self.chart = st.empty()

        for i in range(h * w):
            image = self.converter.generate(img=image, count=i)
            percent_complete += 1
            progress = int(percent_complete / (h * w) * 100)
            elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
            current_progress = percent_complete / (h * w)
            average_progress = current_progress / elapsed_time if elapsed_time > 0 else 0.0
            remaining_time = datetime.timedelta(seconds=int((1 - current_progress) / average_progress)
                                                ) if average_progress > 0 else datetime.timedelta(seconds=0)
            speed = int(i / elapsed_time) if elapsed_time > 0 else 0
            self.speed_chart.append(speed)
            process.text("{:.2f}% ({}/{}) - {} remaining - {} iterations/s".format(current_progress *
                         100, percent_complete, h * w, remaining_time, speed))
            self.progress_chart()
            my_bar.progress(progress)
        return image

    def progress_chart(self):
        chart_data = pd.DataFrame(
            self.speed_chart)

        self.chart.line_chart(chart_data)


def main():
    web = Web()
    converter = Converter()
    image = [[], [], []]
    if web.my_upload is not None:
        img = Image.open(web.my_upload,)  # streamlitから読み込む
        img_array = np.array(img)  # nparrayに変換
        web.col1.image(img_array, use_column_width=None)  # 読み込んだ画像を表示
        upload = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)  # RGBからBGRに変換
        height, width = upload.shape[:2]
        upload = converter.mosaic(src=upload, ratio=web.number)
        image = converter.store_variable(upload, image, h=height, w=width)
        image = web.progressbar(h=height, w=width, image=image)  # generate
        array = converter.image_save(img=image, h=height, w=width)
        img_bytes = cv2.imencode('.jpg', array)[1].tobytes()
        img_array2 = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        web.col2.image(img_array2)


main()
