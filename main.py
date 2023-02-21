from PIL import Image
import numpy as np
import streamlit as st
import cv2


class Converter():
    def __init__(self) -> None:
        # self.file_name = 'data/img/dango.png'
        # self.pic = cv2.imread(self.file_name)
        # self.h, self.w = self.pic.shape[:2]
        # self.pic = cv2.cvtColor(self.pic, cv2.COLOR_BGR2RGB)
        # self.pic = self.mosaic(src=self.pic)
        # self.RGB = [[], [], []]
        pass

    def detect_color(self, rgb):
        # RGB値を取得
        r, g, b = rgb[0], rgb[1], rgb[2]

        # 色名を定義するためのデータベース
        color_names = {
            (255, 127, 127): (255, 127, 127),  # 赤
            (255, 191, 127): (255, 191, 127),  # オレンジ
            (255, 255, 127): (255, 255, 127),  # 黄色
            (127, 255, 127): (127, 255, 127),  # 緑
            (127, 191, 255): (127, 191, 255),  # 青
            (127, 127, 255): (127, 127, 255),  # 紫
            (0, 0, 0): (0, 0, 0),  # 黒
            (255, 255, 255): (255, 255, 255),  # 白
            (128, 128, 128): (128, 128, 128)  # 灰色
        }
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
        for name, color in color_names.items():
            distance = (r - color[0]) ** 2 + (g - color[1]) ** 2 + (b - color[2]) ** 2
            if distance < min_distance:
                min_distance = distance
                color_name = name
        return color_name

    def mosaic(self, src, ratio=0.1):
        small = cv2.resize(src, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
        return cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

    def generate(self, img, count):
        print(count)
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
        self.number = st.number_input('Insert a number', min_value=0.01, max_value=1.00)
        # カラム設定
        col1, self.col2 = st.columns(2)

        col1.header("Original image")
        self.col2.header("convert image")

    def progressbar(self, h, w, image):
        i = 0
        process = st.empty()
        my_bar = st.progress(0)
        my_bar.text(i)
        percent_complete = 0
        for i in range(h*w):
            image = self.converter.generate(img=image, count=i)
            percent_complete += 1
            progress = int(percent_complete/(h*w)*100)
            process.text(str(percent_complete/(h*w)*100)+"%")
            my_bar.progress(progress)

        return image


def main():
    web = Web()
    converter = Converter()
    image = [[], [], []]
    if web.my_upload is not None:
        img = Image.open(web.my_upload,)
        img_array = np.array(img)
        st.image(img_array, use_column_width=None)
        upload = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        height, width = upload.shape[:2]
        upload = converter.mosaic(src=upload, ratio=web.number)
        image = converter.store_variable(upload, image, h=height, w=width)
        image = web.progressbar(h=height, w=width, image=image)  # generate
        array = converter.image_save(img=image, h=height, w=width)
        img_bytes = cv2.imencode('.jpg', array)[1].tobytes()
        img_array2 = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        web.col2.image(img_array2)


main()
