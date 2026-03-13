import cv2
import numpy as np
from PIL import Image, ImageEnhance


class EdgeFilter:
    def __init__(self) -> None:
        pass

    def canny(self, image: np.ndarray, th1: float, th2: float) -> np.ndarray:
        bgr = image[:, :, :3]
        alpha: np.ndarray | list[float] = []
        if len(image[0][0]) == 4:
            alpha = image[:, :, 3]

        gray = cv2.cvtColor(bgr, cv2.COLOR_RGB2GRAY)
        edge = cv2.blur(gray, (3, 3))
        edge = cv2.Canny(edge, th1, th2, apertureSize=3)
        edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2RGB)
        result = cv2.subtract(bgr, edge)

        if len(image[0][0]) == 4:
            return np.dstack([result, alpha])
        return result

    def dog(self, img: np.ndarray, scratch: bool = False) -> np.ndarray:
        def _filter(
            img_array: np.ndarray,
            size: int,
            p: int,
            sigma: float,
            eps: float,
            phi: float,
            k: float = 1.6,
        ) -> np.ndarray:
            eps /= 255
            g1 = cv2.GaussianBlur(img_array, (size, size), sigma)
            g2 = cv2.GaussianBlur(img_array, (size, size), sigma * k)
            d = (1 + p) * g1 - p * g2
            d /= d.max()
            e = 1 + np.tanh(phi * (d - eps))
            e[e >= 1] = 1
            return e * 255

        base_image = img[:, :, :3]
        alpha: np.ndarray | list[float] = []
        if len(img[0][0]) == 4:
            alpha = img[:, :, 3]
        image = base_image.copy()

        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = np.array(image, dtype=np.float64)
        image = _filter(image, 17, 40, 1.4, 0, 15)
        _, image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)  # noqa: FBT003
        a = np.array(image, np.uint8)
        image = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)

        if scratch:
            image = cv2.bitwise_not(image)
            image = self.morphology_erode(image)

        result = cv2.subtract(base_image, image)
        if len(img[0][0]) == 4:
            return np.dstack([result, alpha])
        return result

    def morphology_dilate(self, image: np.ndarray) -> np.ndarray:
        dilate_kernel = np.ones((3, 3), np.uint8)
        dilate_filtered = cv2.morphologyEx(image, cv2.MORPH_DILATE, dilate_kernel)
        _, binary_image = cv2.threshold(
            dilate_filtered,
            200,
            255,
            cv2.THRESH_BINARY_INV,
        )
        binary_image = np.array(binary_image, np.uint8)
        binary_image = cv2.bitwise_not(binary_image)
        return binary_image

    def morphology_erode(self, image: np.ndarray) -> np.ndarray:
        kernel_2 = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        erode_filtered = np.array(
            cv2.morphologyEx(image, cv2.MORPH_ERODE, kernel_2, iterations=3),
            np.uint8,
        )
        return erode_filtered

    @staticmethod
    def kuwahara(im: np.ndarray, n: int) -> np.ndarray:
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

    def apply_kuwahara(self, image: np.ndarray) -> np.ndarray:
        filtered = self.kuwahara(image, 5)
        return filtered

    def median(self, image: np.ndarray, size: int) -> np.ndarray:
        return cv2.medianBlur(image, size)


class ImageEnhancer:
    def __init__(self) -> None:
        pass

    def saturation(self, image: np.ndarray, value: float) -> np.ndarray:
        img = Image.fromarray(image)
        enhancer = ImageEnhance.Color(img)
        result = enhancer.enhance(value)
        result = np.array(result)
        return result

    def brightness(self, image: np.ndarray, value: float) -> np.ndarray:
        img = Image.fromarray(image)
        enhancer = ImageEnhance.Brightness(img)
        result = enhancer.enhance(value)
        result = np.array(result)
        return result

    def sharpness(self, image: np.ndarray, value: float) -> np.ndarray:
        img = Image.fromarray(image)
        enhancer = ImageEnhance.Sharpness(img)
        result = enhancer.enhance(value)
        result = np.array(result)
        return result

    def contrast(self, image: np.ndarray, value: float) -> np.ndarray:
        img = Image.fromarray(image)
        enhancer = ImageEnhance.Contrast(img)
        result = enhancer.enhance(value)
        result = np.array(result)
        return result

    def slider_mosaic(self, image: np.ndarray, ratio: float) -> np.ndarray:
        new_width = max(1, int(round(image.shape[1] * ratio)))
        new_height = max(1, int(round(image.shape[0] * ratio)))
        small = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_NEAREST,
        )
        return small

    def grid_mosaic(self, image: np.ndarray, size: int) -> np.ndarray:
        aspect = image.shape[0] / image.shape[1]
        small = cv2.resize(
            image,
            (size, int(size * aspect)),
            interpolation=cv2.INTER_NEAREST,
        )
        return small

    def decrease(self, image: np.ndarray) -> np.ndarray:
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
        self,
        image: np.ndarray,
        grid_size: int,
        line_color: tuple[int, int, int] = (0, 0, 0),
        line_thickness: int = 1,
        opacity: float = 0.5,
    ) -> np.ndarray:
        has_alpha = len(image.shape) == 3 and image.shape[2] == 4
        alpha = np.zeros(image.shape[:2], dtype=np.uint8)
        if has_alpha:
            bgr = image[:, :, :3].copy()
            alpha = image[:, :, 3]
        else:
            bgr = image.copy()

        bgr = np.array(bgr, dtype=np.uint8)
        h, w = bgr.shape[:2]
        result = bgr.copy()

        for x in range(0, w, grid_size):
            if x > 0:
                cv2.line(result, (x, 0), (x, h - 1), line_color, line_thickness)

        for y in range(0, h, grid_size):
            if y > 0:
                cv2.line(result, (0, y), (w - 1, y), line_color, line_thickness)

        if opacity < 1.0:
            diff = cv2.absdiff(bgr, result)
            grid_mask = np.any(diff > 0, axis=2)

            for c in range(3):
                result[:, :, c] = np.where(
                    grid_mask,
                    bgr[:, :, c] * (1 - opacity) + result[:, :, c] * opacity,
                    bgr[:, :, c],
                )

        if has_alpha:
            return np.dstack([result, alpha])
        return result


class Dithering:
    def __init__(self) -> None:
        pass

    def floyd_steinberg(self, image: np.ndarray, intensity: float = 1.0) -> np.ndarray:
        has_alpha = len(image.shape) == 3 and image.shape[2] == 4
        alpha = np.zeros(image.shape[:2], dtype=np.uint8)
        if has_alpha:
            bgr = image[:, :, :3].copy()
            alpha = image[:, :, 3]
        else:
            bgr = image.copy()

        img = bgr.astype(np.float32)
        h, w = img.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_RGB2GRAY).astype(np.float32)
        dithered_gray = gray.copy()

        for y in range(h):
            for x in range(w):
                old_val = dithered_gray[y, x]
                new_val = np.round(old_val / 255) * 255
                dithered_gray[y, x] = new_val
                error = old_val - new_val

                if x + 1 < w:
                    dithered_gray[y, x + 1] += error * 7 / 16
                if y + 1 < h:
                    if x > 0:
                        dithered_gray[y + 1, x - 1] += error * 3 / 16
                    dithered_gray[y + 1, x] += error * 5 / 16
                    if x + 1 < w:
                        dithered_gray[y + 1, x + 1] += error * 1 / 16

        dithered_gray = np.clip(dithered_gray, 0, 255)
        pattern = (dithered_gray - gray) / 255.0
        pattern = np.expand_dims(pattern, axis=2)
        result = img + pattern * 50 * intensity
        result = np.clip(result, 0, 255).astype(np.uint8)

        if has_alpha:
            return np.dstack([result, alpha])
        return result

    def ordered_dither(
        self,
        image: np.ndarray,
        matrix_size: int = 4,
        intensity: float = 1.0,
    ) -> np.ndarray:
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

        has_alpha = len(image.shape) == 3 and image.shape[2] == 4
        alpha = np.zeros(image.shape[:2], dtype=np.uint8)
        if has_alpha:
            bgr = image[:, :, :3].copy()
            alpha = image[:, :, 3]
        else:
            bgr = image.copy()

        img = bgr.astype(np.float32) / 255.0
        h, w = img.shape[:2]
        threshold_map = np.tile(
            bayer_matrix,
            (h // matrix_size + 1, w // matrix_size + 1),
        )[:h, :w]
        threshold_map = np.expand_dims(threshold_map, axis=2)

        threshold_map = (threshold_map - 0.5) * 0.1 * intensity
        result = img + threshold_map
        result = (np.clip(result, 0, 1) * 255).astype(np.uint8)

        if has_alpha:
            return np.dstack([result, alpha])
        return result

    def atkinson(self, image: np.ndarray, intensity: float = 1.0) -> np.ndarray:
        has_alpha = len(image.shape) == 3 and image.shape[2] == 4
        alpha = np.zeros(image.shape[:2], dtype=np.uint8)
        if has_alpha:
            bgr = image[:, :, :3].copy()
            alpha = image[:, :, 3]
        else:
            bgr = image.copy()

        img = bgr.astype(np.float32)
        h, w = img.shape[:2]

        gray = cv2.cvtColor(bgr, cv2.COLOR_RGB2GRAY).astype(np.float32)
        dithered_gray = gray.copy()

        for y in range(h):
            for x in range(w):
                old_val = dithered_gray[y, x]
                new_val = np.round(old_val / 255) * 255
                dithered_gray[y, x] = new_val
                error = old_val - new_val

                if x + 1 < w:
                    dithered_gray[y, x + 1] += error * (1 / 8)
                if x + 2 < w:
                    dithered_gray[y, x + 2] += error * (1 / 8)
                if y + 1 < h:
                    if x > 0:
                        dithered_gray[y + 1, x - 1] += error * (1 / 8)
                    dithered_gray[y + 1, x] += error * (1 / 8)
                    if x + 1 < w:
                        dithered_gray[y + 1, x + 1] += error * (1 / 8)
                if y + 2 < h:
                    dithered_gray[y + 2, x] += error * (1 / 8)

        dithered_gray = np.clip(dithered_gray, 0, 255)
        pattern = (dithered_gray - gray) / 255.0
        pattern = np.expand_dims(pattern, axis=2)
        result = img + pattern * 50 * intensity
        result = np.clip(result, 0, 255).astype(np.uint8)

        if has_alpha:
            return np.dstack([result, alpha])
        return result
