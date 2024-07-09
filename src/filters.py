import cv2
import numpy as np
from PIL import Image
from PIL import ImageEnhance


class EdgeFilter:
    def __init__(self) -> None:
        pass

    def canny(self, image, th1, th2):
        # アルファチャンネルを分離
        bgr = image[:, :, :3]
        if len(image[0][0]) == 4:
            alpha = image[:, :, 3]

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
        if len(image[0][0]) == 4:
            return np.dstack([result, alpha])
        else:
            return result

    def dog(self, img, scratch=False):
        def filter(img_array, size, p, sigma, eps, phi, k=1.6):
            eps /= 255
            g1 = cv2.GaussianBlur(img_array, (size, size), sigma)
            g2 = cv2.GaussianBlur(img_array, (size, size), sigma * k)
            d = (1 + p) * g1 - p * g2
            d /= d.max()
            e = 1 + np.tanh(phi * (d - eps))
            e[e >= 1] = 1
            return e * 255

        # アルファチャンネルを分離
        base_image = img[:, :, :3]
        if len(img[0][0]) == 4:
            alpha = img[:, :, 3]
        image = base_image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.array(image, dtype=np.float64)
        image = filter(image, 17, 40, 1.4, 0, 15)
        # 第一: 画像 第ニ: しきい値 第三: しきい値に当てはまったときになる値
        _, image = cv2.threshold(
            image, 200, 255, cv2.THRESH_BINARY_INV
        )  # しきい値 二値化
        a = np.array(image, np.uint8)
        image = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
        if scratch:
            image = cv2.bitwise_not(image)
        result = cv2.subtract(base_image, image)
        # アルファチャンネルを結合して返す
        if len(img[0][0]) == 4:
            return np.dstack([result, alpha])
        else:
            return result

    def morphology_gradient(self, base_image, op):
        gray_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)

        kernel = np.ones((2, 2), np.uint8)
        morphology_img = cv2.morphologyEx(
            gray_image, op, kernel)

        _, binary_image = cv2.threshold(
            morphology_img, 200, 255, cv2.THRESH_BINARY_INV)  # しきい値 二値化
        binary_image = np.array(binary_image, np.uint8)

        binary_image_bgr = cv2.cvtColor(morphology_img, cv2.COLOR_GRAY2BGR)
        result = cv2.subtract(base_image, binary_image_bgr)

        cv2.imwrite("result.jpg", binary_image_bgr)
        return result


class ImageEnhancer:
    def __init__(self) -> None:
        pass

    def saturation(self, image, value):
        img = Image.fromarray(image)
        enhancer = ImageEnhance.Color(img)
        result = enhancer.enhance(value)
        result = np.array(result)
        return result

    def brightness(self, image, value):
        img = Image.fromarray(image)
        enhancer = ImageEnhance.Brightness(img)
        result = enhancer.enhance(value)
        result = np.array(result)
        return result

    def sharpness(self, image, value):
        img = Image.fromarray(image)
        enhancer = ImageEnhance.Sharpness(img)
        result = enhancer.enhance(value)
        result = np.array(result)
        return result

    def mosaic(self, image, ratio):
        small = cv2.resize(
            image, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST
        )
        return small

    def decrease(self, image):
        dst = image.copy()

        idx = np.where((0 <= image) & (64 > image))
        dst[idx] = 32
        idx = np.where((64 <= image) & (128 > image))
        dst[idx] = 96
        idx = np.where((128 <= image) & (192 > image))
        dst[idx] = 160
        idx = np.where((192 <= image) & (256 > image))
        dst[idx] = 224

        return dst
