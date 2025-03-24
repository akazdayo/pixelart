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
        alpha = []
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
        alpha = []
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
        # a = self.morphology_dilate(a)
        image = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)

        if scratch:
            image = cv2.bitwise_not(image)
            image = self.morphology_erode(image)
        result = cv2.subtract(base_image, image)

        # アルファチャンネルを結合して返す
        if len(img[0][0]) == 4:
            return np.dstack([result, alpha])
        else:
            return result

    def morphology_dilate(self, image):
        dilate_kernel = np.ones((3, 3), np.uint8)
        dilate_filtered = cv2.morphologyEx(image, cv2.MORPH_DILATE, dilate_kernel)

        _, binary_image = cv2.threshold(
            dilate_filtered, 200, 255, cv2.THRESH_BINARY_INV
        )  # 二値化
        binary_image = np.array(binary_image, np.uint8)
        binary_image = cv2.bitwise_not(binary_image)

        return binary_image

    def morphology_erode(self, image):
        # kernel = np.ones((2, 2), np.uint8)
        # img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel_3 = np.array(
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ],
            dtype=np.uint8,
        )
        kernel_2 = np.array([[0, 1], [1, 0]], dtype=np.uint8)

        erode_filtered = np.array(
            cv2.morphologyEx(image, cv2.MORPH_ERODE, kernel_2, iterations=3), np.uint8
        )
        return erode_filtered

    @staticmethod
    def kuwahara(im, n):
        filt = np.zeros((2 * n - 1, 2 * n - 1))
        filt[:n, :n] = 1 / n**2
        filts = [
            np.roll(filt, (i * (n - 1), j * (n - 1)), axis=(0, 1))
            for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]
        ]
        u = np.array([cv2.filter2D(im, -1, f) for f in filts])
        u2 = [cv2.filter2D(im**2, -1, f) for f in filts]
        idx = np.argmin([(i - j**2).sum(2) for i, j in zip(u2, u)], 0)
        ix, iy = np.indices(im.shape[:2])
        return u[idx, ix, iy]

    def apply_kuwahara(self, image):
        filtered = self.kuwahara(image, 5)
        return filtered

    def median(self, image, size):
        return cv2.medianBlur(image, size)


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

    def contrast(self, image, value):
        img = Image.fromarray(image)
        enhancer = ImageEnhance.Contrast(img)
        result = enhancer.enhance(value)
        result = np.array(result)
        return result

    def slider_mosaic(self, image, ratio):
        small = cv2.resize(
            image, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST
        )
        return small

    def grid_mosaic(self, image, size):
        aspect = image.shape[0] / image.shape[1]
        small = cv2.resize(
            image, (size, int(size * aspect)), interpolation=cv2.INTER_NEAREST
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
