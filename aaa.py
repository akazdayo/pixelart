import numpy as np
from ctypes import cdll, c_int, POINTER, c_uint8

# Load the library
a = cdll.LoadLibrary("./libs/libpixelart.so")

# Call the add function
b = a.add(1,2)
print(b)

# Create a numpy array
arr = np.array([[1,2,3], [255,255,255]], dtype=np.uint8)
arr_h, arr_w = arr.shape

arr2 = np.array([[1,2,3],[255,255,255]], dtype=np.uint8)
arr2_h, arr2_w = arr2.shape

arr3 = np.array([[3,2,1],[255,255,255]], dtype=np.uint8)
arr3_h, arr3_w = arr3.shape

# Get the pointer to the numpy array data
data = arr.ctypes.data_as(POINTER((c_uint8 * arr_w) * arr_h))
data2 = arr2.ctypes.data_as(POINTER((c_uint8 * arr2_w) * arr2_h))
data3 = arr3.ctypes.data_as(POINTER((c_uint8 * arr3_w) * arr3_h))

# Call the color_change function
c = a.color_change(1,2,3,data, data2, data3)