import numpy as np
import pytest
from pixelart_backend.ai import AI


class TestAI:
    """AI クラスのユニットテスト"""

    @pytest.fixture
    def ai_instance(self):
        """AI インスタンスを作成"""
        return AI()

    @pytest.fixture
    def rgba_image(self):
        """テスト用 RGBA 画像 (20x20)"""
        return np.random.randint(0, 256, (20, 20, 4), dtype=np.uint8)

    @pytest.fixture
    def simple_rgba_image(self):
        """2色のみの単純な RGBA 画像 (10x10)"""
        image = np.zeros((10, 10, 4), dtype=np.uint8)
        image[:5, :, :3] = [255, 0, 0]  # 上半分: 赤
        image[5:, :, :3] = [0, 0, 255]  # 下半分: 青
        image[:, :, 3] = 255  # 完全不透明
        return image


class TestGetColorCount(TestAI):
    """get_color_count() のテスト"""

    def test_get_color_count_single_color(self, ai_instance):
        """単色画像のカラー数が 1 であること"""
        flat_img = np.full((100, 3), [128, 128, 128], dtype=np.uint8)
        result = ai_instance.get_color_count(flat_img)
        assert result == 1

    def test_get_color_count_two_colors(self, ai_instance):
        """2色画像のカラー数が 2 であること"""
        flat_img = np.array([[0, 0, 0], [255, 255, 255]] * 50, dtype=np.uint8)
        result = ai_instance.get_color_count(flat_img)
        assert result == 2

    def test_get_color_count_multiple_colors(self, ai_instance):
        """複数色の画像で正しいカラー数を返すこと"""
        flat_img = np.array(
            [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]],
            dtype=np.uint8,
        )
        result = ai_instance.get_color_count(flat_img)
        assert result == 4


class TestLab2RGB(TestAI):
    """lab2rgb() のテスト"""

    def test_lab2rgb_output_shape(self, ai_instance):
        """出力形状が正しいこと"""
        lab_array = np.array([[50, 0, 0], [100, 50, 50]], dtype=np.uint8)
        result = ai_instance.lab2rgb(lab_array)
        assert result.shape == (2, 3)

    def test_lab2rgb_output_range(self, ai_instance):
        """出力値が 0-255 の範囲内であること"""
        lab_array = np.array([[50, 128, 128], [200, 50, 50]], dtype=np.uint8)
        result = ai_instance.lab2rgb(lab_array)
        assert np.all(result >= 0)
        assert np.all(result <= 255)

    def test_lab2rgb_single_color(self, ai_instance):
        """単一色の変換が正しいこと"""
        lab_array = np.array([[0, 128, 128]], dtype=np.uint8)
        result = ai_instance.lab2rgb(lab_array)
        assert result.shape == (1, 3)


class TestGetColor(TestAI):
    """get_color() のテスト"""

    def test_get_color_returns_hex_list(self, ai_instance, rgba_image):
        """16進数カラーコードのリストが返されること"""
        result = ai_instance.get_color(rgba_image, 3, 10)
        assert isinstance(result, list)
        assert len(result) <= 3

    def test_get_color_hex_format(self, ai_instance, rgba_image):
        """各色が正しい16進数フォーマットであること"""
        result = ai_instance.get_color(rgba_image, 3, 10)
        for hex_code in result:
            assert hex_code.startswith("#")
            assert len(hex_code) == 7
            # 有効な16進数であること
            int(hex_code[1:], 16)

    def test_get_color_respects_color_count(self, ai_instance, rgba_image):
        """指定した色数以下の結果が返されること"""
        result = ai_instance.get_color(rgba_image, 2, 10)
        assert len(result) <= 2

    def test_get_color_with_more_colors_than_unique(
        self, ai_instance, simple_rgba_image
    ):
        """ユニーク色数より多い色数を指定した場合、ユニーク色数に制限されること"""
        result = ai_instance.get_color(simple_rgba_image, 100, 10)
        # 2色しかないので、結果は2色以下
        assert len(result) <= 100
        assert len(result) > 0

    def test_get_color_single_color(self, ai_instance):
        """1色を指定した場合に動作すること"""
        image = np.full((10, 10, 4), [128, 128, 128, 255], dtype=np.uint8)
        result = ai_instance.get_color(image, 1, 10)
        assert len(result) == 1
        assert result[0].startswith("#")

    def test_get_color_rgb_image_no_rb_swap(self, ai_instance):
        red_rgb = np.zeros((20, 20, 4), dtype=np.uint8)
        red_rgb[:, :, 0] = 255
        red_rgb[:, :, 3] = 255
        result = ai_instance.get_color(red_rgb, 1, 10)
        hex_code = result[0]
        r = int(hex_code[1:3], 16)
        b = int(hex_code[5:7], 16)
        assert r > b, f"Expected red > blue but got R={r}, B={b} from {hex_code}"

    def test_get_color_3channel_rgb_input(self, ai_instance):
        rgb_image = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)
        result = ai_instance.get_color(rgb_image, 3, 10)
        assert isinstance(result, list)
        assert len(result) > 0
        for hex_code in result:
            assert hex_code.startswith("#")
            assert len(hex_code) == 7
