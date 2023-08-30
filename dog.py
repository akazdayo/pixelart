import cv2
import numpy as np


class Edge():
    def __init__(self) -> None:
        pass

    def _convolve2d(self, image, kernel):
        shape = (image.shape[0] - kernel.shape[0] + 1, image.shape[1] - kernel.shape[1] + 1) + kernel.shape
        strides = image.strides * 2
        strided_image = np.lib.stride_tricks.as_strided(image, shape, strides)
        return np.einsum('kl,ijkl->ij', kernel, strided_image)

    def _convolve2d_multichannel(self, image, kernel):
        convolved_image = np.empty((image.shape[0] - kernel.shape[0] + 1, image.shape[1] - kernel.shape[1] + 1, image.shape[2]))
        for i in range(image.shape[2]):
            convolved_image[:, :, i] = self._convolve2d(image[:, :, i], kernel)
        return convolved_image

    def _pad_singlechannel_image(self, image, kernel_shape, boundary):
        return np.pad(image, ((int(kernel_shape[0] / 2),), (int(kernel_shape[1] / 2),)), boundary)

    def _pad_multichannel_image(self, image, kernel_shape, boundary):
        return np.pad(image, ((int(kernel_shape[0] / 2),), (int(kernel_shape[1] / 2),), (0,)), boundary)

    def convolve2d(self, image, kernel, boundary='edge'):
        if image.ndim == 2:
            pad_image = self._pad_singlechannel_image(image, kernel.shape, boundary) if boundary is not None else image
            return self._convolve2d(pad_image, kernel)
        elif image.ndim == 3:
            pad_image = self._pad_multichannel_image(image, kernel.shape, boundary) if boundary is not None else image
            return self._convolve2d_multichannel(pad_image, kernel)

    def create_log_kernel(self, size=(13, 13), sigma=2):
        center = ((size[0] - 1) / 2, (size[1] - 1) / 2)
        sigma2 = sigma * sigma
        sigma6 = sigma2 * sigma2 * sigma2
        sigma2_2 = sigma2 * 2.0
        sigma2_2_inv = 1.0 / sigma2_2
        sigma6_2pi_inv = 1.0 / (sigma6 * 2.0 * np.pi)

        def calc_weight(y, x):
            sqDist = (x - center[1]) ** 2 + (y - center[0]) ** 2
            return (sqDist - sigma2_2) * sigma6_2pi_inv * np.exp(-sqDist * sigma2_2_inv)
        return np.fromfunction(calc_weight, size)

    def detect_zero_crossing(self, image):
        pad_image = np.pad(image, 1, 'edge')
        shape = image.shape + (3, 3)
        strides = pad_image.strides * 2
        strided_image = np.lib.stride_tricks.as_strided(pad_image, shape, strides).reshape(shape[0], shape[1], 9)
        return np.apply_along_axis(
            lambda array: 1 if array[3] * array[5] < 0 or array[2] * array[6] < 0 else 0,
            2, strided_image)


edge = Edge()

original_image = cv2.imread("sample/cat.jfif")
original_image = np.array(original_image)
if np.issubdtype(original_image.dtype, np.integer):
    original_image = original_image / np.iinfo(original_image.dtype).max
    gray_image = 0.2116 * original_image[:, :, 0] + 0.7152 * original_image[:, :, 1] + 0.0722 * original_image[:, :, 2]


log_kernel3 = edge.create_log_kernel(size=(41, 41), sigma=7)
log_image3 = edge.convolve2d(gray_image, log_kernel3)
print(log_image3)

zero_crossing_image3 = edge.detect_zero_crossing(log_image3) * 255
print(np.amax(zero_crossing_image3))

cv2.imwrite("sample/test2.png", zero_crossing_image3)
