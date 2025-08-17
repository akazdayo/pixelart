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


class GridMask:
    def __init__(self) -> None:
        pass

    def add_grid(
        self, image, grid_size, line_color=(0, 0, 0), line_thickness=1, opacity=0.5
    ):
        """
        画像にグリッドマスクを追加する

        Args:
            image: 入力画像 (numpy array)
            grid_size: グリッドのサイズ (ピクセル数)
            line_color: グリッド線の色 (B, G, R)
            line_thickness: 線の太さ
            opacity: グリッドの透明度 (0.0-1.0)

        Returns:
            グリッド付きの画像
        """
        # アルファチャンネルを分離
        has_alpha = len(image.shape) == 3 and image.shape[2] == 4
        if has_alpha:
            bgr = image[:, :, :3].copy()
            alpha = image[:, :, 3]
        else:
            bgr = image.copy()

        # データ型をuint8に確保
        bgr = np.array(bgr, dtype=np.uint8)
        h, w = bgr.shape[:2]

        # 結果画像を元画像のコピーで初期化
        result = bgr.copy()

        # グリッド線を直接描画
        # 縦線を描画
        for x in range(0, w, grid_size):
            if x > 0:  # 左端は描画しない
                cv2.line(result, (x, 0), (x, h - 1), line_color, line_thickness)

        # 横線を描画
        for y in range(0, h, grid_size):
            if y > 0:  # 上端は描画しない
                cv2.line(result, (0, y), (w - 1, y), line_color, line_thickness)

        # 透明度を適用する場合は、グリッド線部分だけブレンド
        if opacity < 1.0:
            # グリッド線が描画された部分を検出
            diff = cv2.absdiff(bgr, result)
            grid_mask = np.any(diff > 0, axis=2)

            # 透明度を適用
            for c in range(3):
                result[:, :, c] = np.where(
                    grid_mask,
                    bgr[:, :, c] * (1 - opacity) + result[:, :, c] * opacity,
                    bgr[:, :, c],
                )

        # アルファチャンネルを結合して返す
        if has_alpha:
            return np.dstack([result, alpha])
        else:
            return result


class Dithering:
    def __init__(self) -> None:
        pass

    def floyd_steinberg(self, image, intensity=1.0):
        """Floyd-Steinberg dithering algorithm - adds noise pattern without changing colors"""
        # アルファチャンネルを分離
        has_alpha = len(image.shape) == 3 and image.shape[2] == 4
        if has_alpha:
            bgr = image[:, :, :3].copy()
            alpha = image[:, :, 3]
        else:
            bgr = image.copy()

        # float型に変換
        img = bgr.astype(np.float32)
        h, w = img.shape[:2]

        # グレースケール版を作成してディザパターンを計算
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        dithered_gray = gray.copy()

        for y in range(h):
            for x in range(w):
                old_val = dithered_gray[y, x]
                new_val = np.round(old_val / 255) * 255
                dithered_gray[y, x] = new_val
                error = old_val - new_val

                # エラーを周囲のピクセルに拡散
                if x + 1 < w:
                    dithered_gray[y, x + 1] += error * 7 / 16
                if y + 1 < h:
                    if x > 0:
                        dithered_gray[y + 1, x - 1] += error * 3 / 16
                    dithered_gray[y + 1, x] += error * 5 / 16
                    if x + 1 < w:
                        dithered_gray[y + 1, x + 1] += error * 1 / 16

        # ディザパターンを元の画像に適用（明度のみ調整）
        dithered_gray = np.clip(dithered_gray, 0, 255)
        pattern = (dithered_gray - gray) / 255.0
        pattern = np.expand_dims(pattern, axis=2)

        # 元の色にディザパターンを適用
        result = img + pattern * 50 * intensity  # パターンの強度を調整
        result = np.clip(result, 0, 255).astype(np.uint8)

        # アルファチャンネルを結合して返す
        if has_alpha:
            return np.dstack([result, alpha])
        else:
            return result

    def ordered_dither(self, image, matrix_size=4, intensity=1.0):
        """Ordered (Bayer) dithering algorithm"""
        # Bayer行列の定義
        bayer_matrices = {
            2: np.array([[0, 2], [3, 1]], dtype=np.float32) / 4,
            4: np.array(
                [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]],
                dtype=np.float32,
            )
            / 16,
            8: np.array(
                [
                    [0, 32, 8, 40, 2, 34, 10, 42],
                    [48, 16, 56, 24, 50, 18, 58, 26],
                    [12, 44, 4, 36, 14, 46, 6, 38],
                    [60, 28, 52, 20, 62, 30, 54, 22],
                    [3, 35, 11, 43, 1, 33, 9, 41],
                    [51, 19, 59, 27, 49, 17, 57, 25],
                    [15, 47, 7, 39, 13, 45, 5, 37],
                    [63, 31, 55, 23, 61, 29, 53, 21],
                ],
                dtype=np.float32,
            )
            / 64,
        }

        if matrix_size not in bayer_matrices:
            matrix_size = 4

        bayer_matrix = bayer_matrices[matrix_size]

        # アルファチャンネルを分離
        has_alpha = len(image.shape) == 3 and image.shape[2] == 4
        if has_alpha:
            bgr = image[:, :, :3].copy()
            alpha = image[:, :, 3]
        else:
            bgr = image.copy()

        img = bgr.astype(np.float32) / 255.0
        h, w = img.shape[:2]

        # Bayer行列をタイル状に配置
        threshold_map = np.tile(
            bayer_matrix, (h // matrix_size + 1, w // matrix_size + 1)
        )[:h, :w]
        threshold_map = np.expand_dims(threshold_map, axis=2)

        # ディザパターンを元の画像に適用（色は変更しない）
        threshold_map = (threshold_map - 0.5) * 0.1 * intensity  # 強度調整
        result = img + threshold_map
        result = (np.clip(result, 0, 1) * 255).astype(np.uint8)

        # アルファチャンネルを結合して返す
        if has_alpha:
            return np.dstack([result, alpha])
        else:
            return result

    def atkinson(self, image, intensity=1.0):
        """Atkinson dithering algorithm"""
        # アルファチャンネルを分離
        has_alpha = len(image.shape) == 3 and image.shape[2] == 4
        if has_alpha:
            bgr = image[:, :, :3].copy()
            alpha = image[:, :, 3]
        else:
            bgr = image.copy()

        # float型に変換
        img = bgr.astype(np.float32)
        h, w = img.shape[:2]

        # グレースケール版を作成してディザパターンを計算
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        dithered_gray = gray.copy()

        for y in range(h):
            for x in range(w):
                old_val = dithered_gray[y, x]
                new_val = np.round(old_val / 255) * 255
                dithered_gray[y, x] = new_val
                error = old_val - new_val

                # Atkinsonエラー拡散 (エラーの1/8ずつを6つの隣接ピクセルに拡散)
                diffusion_factor = 1 / 8

                if x + 1 < w:
                    dithered_gray[y, x + 1] += error * diffusion_factor
                if x + 2 < w:
                    dithered_gray[y, x + 2] += error * diffusion_factor
                if y + 1 < h:
                    if x > 0:
                        dithered_gray[y + 1, x - 1] += error * diffusion_factor
                    dithered_gray[y + 1, x] += error * diffusion_factor
                    if x + 1 < w:
                        dithered_gray[y + 1, x + 1] += error * diffusion_factor
                if y + 2 < h:
                    dithered_gray[y + 2, x] += error * diffusion_factor

        # ディザパターンを元の画像に適用（明度のみ調整）
        dithered_gray = np.clip(dithered_gray, 0, 255)
        pattern = (dithered_gray - gray) / 255.0
        pattern = np.expand_dims(pattern, axis=2)

        # 元の色にディザパターンを適用
        result = img + pattern * 50 * intensity  # パターンの強度を調整
        result = np.clip(result, 0, 255).astype(np.uint8)

        # アルファチャンネルを結合して返す
        if has_alpha:
            return np.dstack([result, alpha])
        else:
            return result
