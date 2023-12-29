import csv
import cv2


class Convert:
    def __init__(self) -> None:
        self.color_dict = {}

    def read_csv(self, path):
        with open(path) as f:
            reader = csv.reader(f)
            color = [[int(v) for v in row] for row in reader]
            return color

    def convert(self, img, option, custom=None):
        def color_change(r, g, b, color_palette):
            if (r, g, b) in self.color_dict:
                return self.color_dict[(r, g, b)]
            # 最も近い色を見つける
            min_distance = float('inf')
            color_name = None
            for color in color_palette:
                # ユークリッド距離
                # 差分を取って2乗すると距離になる。
                distance = (int(r) - color[0]) ** 2 + (int(g) -
                                                       color[1]) ** 2 + (int(b) - color[2]) ** 2
                if distance < min_distance:
                    min_distance = distance
                    color_name = color
            self.color_dict[(r, g, b)] = color_name
            return color_name

        w, h = img.shape[:2]
        changed = img.copy()
        # 選択されたcsvファイルを読み込む
        color_palette = []
        if option != "Custom":
            color_palette = self.read_csv("./color/"+option+".csv")
        else:
            if custom == [] or custom == None:
                return
            color_palette = custom

        for height in range(h):
            for width in range(w):
                color = color_change(
                    img[width][height][0], img[width][height][1], img[width][height][2], color_palette)
                changed[width][height][0] = color[0]  # 赤
                changed[width][height][1] = color[1]  # 緑
                changed[width][height][2] = color[2]  # 青
        return changed

    def resize_image(self, image):
        img_size = image.shape[0] * image.shape[1]
        if img_size > 2073600:
            # 画像をFull HDよりも小さくする
            # 面積から辺の比に直す。
            # 面積比 相似比 検索
            ratio = (img_size / 2073600) ** 0.5
            new_height = int(image.shape[0] / ratio)
            new_width = int(image.shape[1] / ratio)
            result = cv2.resize(image, (new_width, new_height))
        return result
