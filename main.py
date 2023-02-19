import cv2
from PIL import Image
import numpy as np


def mosaic(src, ratio=0.1):
    small = cv2.resize(src, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)


file_name = 'rabosann.png'
pic = cv2.imread('data/sample/'+file_name)
h, w = pic.shape[:2]
pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
pic = mosaic(src=pic,ratio=0.5)
RGB = [[], [], []]



def detect_color(rgb):
    # RGB値の各要素を変数に格納します。
    r, g, b = rgb[0], rgb[1], rgb[2]
    
    # RGB値を0~1の範囲に変換します。
    r /= 255
    g /= 255
    b /= 255
    
    # RGB値の輝度（明度）を計算します。
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    
    # 輝度が一定の値以下であれば黒と判定し、一定の値以上であれば白と判定します。
    if y <= 0.1:#black
        return [0,0,0]
    elif y >= 0.8:#white
        return [255,255,255]
    
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
        if s < 0.1:#gray
            #return [255,255,255]
            return [228,229,227]
        elif h < 1:#red
            return [255,127,127]
        elif h < 2:#orange
            return [255,191,127]
        elif h < 3:#yellow
            return [255,255,127]
        elif h < 4:#green
            return [127,255,127]
        elif h < 5.5 and h >= 4.5:  # purple
            return [191, 127, 255]
        elif h < 2.5 and h >= 0.5:  # blue
            return [127, 127, 255]
    
    return [0,0,0]





for k in range(3):
    array = pic[:, :, k]
    for j in range(h):
        for i in range(w):
            RGB[k].append(str(array[j, i]))


for i in range(h*w):
    color = detect_color(rgb=[int(RGB[0][i]),int(RGB[1][i]),int(RGB[2][i])])
    print(i)
    RGB[0][i] = color[0]
    RGB[1][i] = color[1]
    RGB[2][i] = color[2]

    
"""debug用
color = detect_color(rgb=[int(RGB[0][33101]),int(RGB[1][33101]),int(RGB[2][33101])])
print(str(RGB[0][33101])+":"+str(RGB[1][33101])+":"+str(RGB[2][33101]))

print(color)
"""
# RGBリストをNumPy配列に変換する
array = np.zeros((h, w, 3), dtype=np.uint8)
for k in range(3):
    for j in range(h):
        for i in range(w):
            index = j * w + i
            array[j, i, k] = int(RGB[k][index])

# NumPy配列からPillowのImageオブジェクトを作成する
image = Image.fromarray(array)

# 画像を保存する
image.save("data/converted/"+file_name + ".png")
