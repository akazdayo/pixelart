import numpy as np
import cv2
from sklearn.cluster import KMeans


class AI:
    def __init__(self) -> None:
        pass

    def get_color(self, image, color, iter):
        img = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
        img = img.reshape((img.shape[0] * img.shape[1], 3))
        color_count = self.get_color_count(img)
        if color_count < color:
            color = color_count
        cluster = KMeans(n_clusters=color, max_iter=iter)
        cluster.fit(X=img)
        cluster_centers_arr = cluster.cluster_centers_.astype(int, copy=False)
        # クラスタリング結果をRGBに変換
        rgb_colors = self.lab2rgb(cluster_centers_arr)
        hexlist = []

        # LAB色空間での分布を表示
        import matplotlib.pyplot as plt
        import os

        # 結果表示用の3D図を作成
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        # サンプリングして元のデータポイントをプロット
        sample_size = min(1000, len(img))
        sample_indices = np.random.choice(len(img), sample_size, replace=False)
        sample_points = img[sample_indices]

        # クラスタごとに色分けしてプロットし、セントロイドまでの線を描画
        labels = cluster.labels_[sample_indices]
        for i in range(color):
            mask = labels == i
            points = sample_points[mask]
            if len(points) > 0:
                hex_color = "#%02x%02x%02x" % tuple(rgb_colors[i])
                hexlist.append(hex_color)

                # データポイントをプロット
                ax.scatter(
                    xs=points[:, 0],
                    ys=points[:, 1],
                    zs=points[:, 2],
                    **{
                        "c": hex_color,
                        "alpha": 0.6,
                        "s": 20,
                        "marker": "o",
                        "label": f"Cluster {i + 1}",
                    },
                )

                # セントロイドまでの線を描画
                centroid = cluster_centers_arr[i]
                for point in points:
                    ax.plot(
                        [point[0], centroid[0]],
                        [point[1], centroid[1]],
                        [point[2], centroid[2]],
                        c=hex_color,
                        alpha=0.8,
                        linestyle=(0, (1, 1)),  # より明確な点線パターン
                        linewidth=1.2,
                    )

        # クラスタ中心をプロット
        ax.scatter(
            xs=cluster_centers_arr[:, 0],
            ys=cluster_centers_arr[:, 1],
            zs=cluster_centers_arr[:, 2],
            **{"c": "red", "marker": "*", "s": 200, "label": "Cluster Centers"},
        )

        # グラフの設定
        ax.set(
            xlabel="L* (Lightness)",
            ylabel="a* (Green-Red)",
            zlabel="b* (Blue-Yellow)",
            title="K-means Clustering in LAB Color Space",
        )
        ax.legend()

        plt.tight_layout()

        # プレビュー画像を保存
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(
            os.path.join(output_dir, "clustering_preview.png"),
            bbox_inches="tight",
            dpi=300,
        )

        # プレビューを表示
        # plt.show()

        del img
        del cluster
        del cluster_centers_arr
        return hexlist

    def get_color_count(self, image):
        _, unique_counts = np.unique(image.reshape(-1, 3), axis=0, return_counts=True)
        unique_colors = len(unique_counts)
        return unique_colors

    @staticmethod
    def lab2rgb(image):
        # LAB色空間が入った配列
        lab_array = np.array(image, dtype=np.uint8)

        # 配列の形状を変更（OpenCVの色空間変換関数が3次元配列を必要とするため）
        lab_array = lab_array.reshape(-1, 1, 3)

        # LABからRGBへの変換
        rgb_array = cv2.cvtColor(lab_array, cv2.COLOR_Lab2RGB)  # numpy配列
        rgb_array = rgb_array.reshape(len(rgb_array), 3)
        return rgb_array
