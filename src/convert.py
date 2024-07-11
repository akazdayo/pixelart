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
            min_distance = float("inf")
            color_name = None
            for color in color_palette:
                # ユークリッド距離
                # 差分を取って2乗すると距離になる。
                distance = (
                    (int(r) - color[0]) ** 2
                    + (int(g) - color[1]) ** 2
                    + (int(b) - color[2]) ** 2
                )
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
            color_palette = self.read_csv("./color/" + option + ".csv")
        else:
            if custom == [] or custom is None:
                return
            color_palette = custom

        for height in range(h):
            for width in range(w):
                color = color_change(
                    img[width][height][0],
                    img[width][height][1],
                    img[width][height][2],
                    color_palette,
                )
                changed[width][height][0] = color[0]  # 赤
                changed[width][height][1] = color[1]  # 緑
                changed[width][height][2] = color[2]  # 青
        return changed

    def resize_image(self, image):
        img_size = image.shape[0] * image.shape[1]
        # 画像をFull HDよりも小さくする
        ratio = (img_size / 2073600) ** 0.5
        new_height = int(image.shape[0] / ratio)
        new_width = int(image.shape[1] / ratio)
        result = cv2.resize(image, (new_width, new_height))
        return result

    def delete_alpha(self, image):
        # 画像がアルファチャンネルを持っているか確認
        if image.shape[2] == 4:
            b = image[:, :, 0]
            g = image[:, :, 1]
            r = image[:, :, 2]
            a = image[:, :, 3]
            conv_a = a.copy()
            for i, x in enumerate(a):
                for j, y in enumerate(x):
                    if y != 0:
                        conv_a[i][j] = 255

            merged = cv2.merge([b, g, r, conv_a])
            return merged
        else:
            # アルファチャンネルがない場合はそのまま返す
            return image

    def delete_transparent_color(self, image):
        # 画像がアルファチャンネルを持っているか確認
        if image.shape[2] == 4:
            b = image[:, :, 0]
            g = image[:, :, 1]
            r = image[:, :, 2]
            a = image[:, :, 3]
            for i, x in enumerate(a):
                for j, y in enumerate(x):
                    if y == 0:
                        b[i][j] = 255
                        g[i][j] = 255
                        r[i][j] = 255

            merged = cv2.merge([b, g, r, a])
            return merged
        else:
            # アルファチャンネルがない場合はそのまま返す
            return image
