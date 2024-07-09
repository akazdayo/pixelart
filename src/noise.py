import cv2

INPUT_IMG = r"./ahri.jpg"

# オリジナル画像
src = cv2.imread(INPUT_IMG)
# ノイズ除去した画像
result = cv2.fastNlMeansDenoising(src, h=5)

# 画像保存
cv2.imwrite("output2.png", result)
