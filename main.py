import cv2
from PIL import Image
import numpy as np
import streamlit as st
from concurrent.futures.thread import ThreadPoolExecutor


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
        # RGB値の各要素を変数に格納します。
        r, g, b = rgb[0], rgb[1], rgb[2]

        # RGB値を0~1の範囲に変換します。
        r /= 255
        g /= 255
        b /= 255

        # RGB値の輝度（明度）を計算します。
        y = 0.2126 * r + 0.7152 * g + 0.0722 * b

        # 輝度が一定の値以下であれば黒と判定し、一定の値以上であれば白と判定します。
        if y <= 0.1:
            return [0, 0, 0]
        elif y >= 0.9:
            return [255, 255, 255]

        # 以下の処理は、色相（Hue）と彩度（Saturation）を計算する部分です。
        cmax = max(r, g, b)
        cmin = min(r, g, b)
        delta = cmax - cmin

        h = 0
        s = 0
        l = (cmax + cmin) / 2

        if delta != 0:
            if cmax == r:
                h = (g - b) / delta % 6
            elif cmax == g:
                h = (b - r) / delta + 2
            elif cmax == b:
                h = (r - g) / delta + 4

            s = delta / (1 - abs(2 * l - 1))

            if s < 0.1:  # gray
                return [228, 229, 227]
            elif h < 1:  # red
                return [255, 127, 127]
            elif h < 2:  # orange
                return [255, 191, 127]
            elif h < 3:  # yellow
                return [255, 255, 127]
            elif h < 4:  # green
                return [127, 255, 127]
            elif h < 5:  # blue
                return [127, 127, 255]
            elif h < 6:  # purple
                return [191, 127, 255]

        return [0, 0, 0]

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
        image.save('./data/restored.png')
        return array


class Web():
    def __init__(self) -> None:
        self.converter = Converter()
        st.title("PixelArt-Converter")
        # self.my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        # アップローダー
        self.my_upload = st.file_uploader("以下からファイルアップロード", type=['jpg', 'png'])
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
        percent = (h*w) / 100
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
