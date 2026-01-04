import numpy as np
import pytest
from src.convert import Convert


class TestConvert:
    """Convert クラスのユニットテスト"""

    @pytest.fixture
    def converter(self):
        """Convert インスタンスを作成"""
        return Convert()

    @pytest.fixture
    def rgb_image(self):
        """テスト用 RGB 画像 (100x100)"""
        return np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    @pytest.fixture
    def rgba_image(self):
        """テスト用 RGBA 画像 (100x100)"""
        return np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8)

    @pytest.fixture
    def large_image(self):
        """Full HD より大きい画像 (2000x2000)"""
        return np.random.randint(0, 256, (2000, 2000, 3), dtype=np.uint8)


class TestReadCsv(TestConvert):
    """read_csv() のテスト"""

    def test_read_csv_returns_list_of_rgb_values(self, converter):
        """CSV ファイルを正しく読み込めること"""
        result = converter.read_csv("./color/pyxel.csv")

        assert isinstance(result, list)
        assert len(result) > 0
        # 各行が RGB 値 (3要素のリスト) であること
        for row in result:
            assert len(row) == 3
            assert all(isinstance(v, int) for v in row)
            assert all(0 <= v <= 255 for v in row)

    def test_read_csv_pyxel_has_16_colors(self, converter):
        """pyxel パレットが 16 色であること"""
        result = converter.read_csv("./color/pyxel.csv")
        assert len(result) == 16

    def test_read_csv_first_color_is_black(self, converter):
        """pyxel パレットの最初の色が黒であること"""
        result = converter.read_csv("./color/pyxel.csv")
        assert result[0] == [0, 0, 0]

    def test_read_csv_file_not_found(self, converter):
        """存在しないファイルで FileNotFoundError が発生すること"""
        with pytest.raises(FileNotFoundError):
            converter.read_csv("./color/nonexistent.csv")


class TestConvertMethod(TestConvert):
    """convert() のテスト"""

    def test_convert_with_custom_palette(self, converter, rgb_image):
        """カスタムパレットで変換できること"""
        custom_palette = [[0, 0, 0], [255, 255, 255], [255, 0, 0]]
        result = converter.convert(rgb_image, "Custom", custom=custom_palette)

        assert result.shape == rgb_image.shape
        assert result.dtype == np.uint8 or result.dtype == np.uint64

    def test_convert_with_preset_palette(self, converter, rgb_image):
        """プリセットパレット (pyxel) で変換できること"""
        result = converter.convert(rgb_image, "pyxel")

        assert result.shape == rgb_image.shape

    def test_convert_custom_empty_raises_error(self, converter, rgb_image):
        """空のカスタムパレットで ValueError が発生すること"""
        with pytest.raises(ValueError, match="Custom Palette is empty"):
            converter.convert(rgb_image, "Custom", custom=None)

    def test_convert_custom_empty_list_raises_error(self, converter, rgb_image):
        """空リストのカスタムパレットで ValueError が発生すること"""
        with pytest.raises(ValueError, match="Custom Palette is empty"):
            converter.convert(rgb_image, "Custom", custom=[])

    def test_convert_output_colors_in_palette(self, converter):
        """出力画像の色がパレットに含まれること"""
        # 単色の画像を作成
        image = np.full((50, 50, 3), [128, 128, 128], dtype=np.uint8)
        custom_palette = [[0, 0, 0], [255, 255, 255]]
        result = converter.convert(image, "Custom", custom=custom_palette)

        # 出力の各ピクセルがパレットの色であること
        unique_colors = np.unique(result.reshape(-1, 3), axis=0).tolist()
        for color in unique_colors:
            assert color in custom_palette


class TestResizeImage(TestConvert):
    """resize_image() のテスト"""

    def test_resize_large_image(self, converter, large_image):
        """大きい画像がリサイズされること"""
        result = converter.resize_image(large_image)

        # Full HD (1920x1080 = 2,073,600 ピクセル) 以下になること
        assert result.shape[0] * result.shape[1] <= 2073600

    def test_resize_preserves_aspect_ratio(self, converter):
        """アスペクト比が維持されること"""
        # 2:1 のアスペクト比の画像
        image = np.random.randint(0, 256, (2000, 4000, 3), dtype=np.uint8)
        result = converter.resize_image(image)

        original_ratio = image.shape[1] / image.shape[0]
        result_ratio = result.shape[1] / result.shape[0]

        assert abs(original_ratio - result_ratio) < 0.01

    def test_resize_preserves_channels(self, converter, large_image):
        """チャンネル数が維持されること"""
        result = converter.resize_image(large_image)
        assert result.shape[2] == large_image.shape[2]

    def test_resize_returns_uint8(self, converter, large_image):
        """リサイズ後も uint8 であること"""
        result = converter.resize_image(large_image)
        assert result.dtype == np.uint8


class TestDeleteAlpha(TestConvert):
    """delete_alpha() のテスト"""

    def test_delete_alpha_converts_semitransparent_to_opaque(self, converter):
        """半透明ピクセルが不透明になること"""
        # 半透明のピクセルを含む画像
        image = np.zeros((10, 10, 4), dtype=np.uint8)
        image[:, :, 3] = 128  # アルファ = 128 (半透明)

        result = converter.delete_alpha(image)

        # アルファが 255 (不透明) になること
        assert np.all(result[:, :, 3] == 255)

    def test_delete_alpha_preserves_fully_transparent(self, converter):
        """完全透明 (alpha=0) は維持されること"""
        image = np.zeros((10, 10, 4), dtype=np.uint8)
        image[:, :, 3] = 0  # 完全透明

        result = converter.delete_alpha(image)

        assert np.all(result[:, :, 3] == 0)

    def test_delete_alpha_preserves_bgr(self, converter):
        """BGR チャンネルが維持されること"""
        image = np.zeros((10, 10, 4), dtype=np.uint8)
        image[:, :, 0] = 100  # B
        image[:, :, 1] = 150  # G
        image[:, :, 2] = 200  # R
        image[:, :, 3] = 128  # A

        result = converter.delete_alpha(image)

        assert np.all(result[:, :, 0] == 100)
        assert np.all(result[:, :, 1] == 150)
        assert np.all(result[:, :, 2] == 200)

    def test_delete_alpha_rgb_image_unchanged(self, converter, rgb_image):
        """RGB 画像はそのまま返されること"""
        result = converter.delete_alpha(rgb_image)
        np.testing.assert_array_equal(result, rgb_image)

    def test_delete_alpha_mixed_transparency(self, converter):
        """透明と半透明が混在する画像"""
        image = np.zeros((10, 10, 4), dtype=np.uint8)
        image[:5, :, 3] = 0  # 上半分: 透明
        image[5:, :, 3] = 100  # 下半分: 半透明

        result = converter.delete_alpha(image)

        assert np.all(result[:5, :, 3] == 0)  # 透明は維持
        assert np.all(result[5:, :, 3] == 255)  # 半透明は不透明に


class TestDeleteTransparentColor(TestConvert):
    """delete_transparent_color() のテスト"""

    def test_delete_transparent_color_sets_white(self, converter):
        """透明ピクセルの色が白になること"""
        image = np.zeros((10, 10, 4), dtype=np.uint8)
        image[:, :, 0] = 100  # B
        image[:, :, 1] = 100  # G
        image[:, :, 2] = 100  # R
        image[:, :, 3] = 0  # 完全透明

        result = converter.delete_transparent_color(image)

        # BGR が白 (255, 255, 255) になること
        assert np.all(result[:, :, 0] == 255)
        assert np.all(result[:, :, 1] == 255)
        assert np.all(result[:, :, 2] == 255)

    def test_delete_transparent_color_preserves_alpha(self, converter):
        """アルファチャンネルが維持されること"""
        image = np.zeros((10, 10, 4), dtype=np.uint8)
        image[:, :, 3] = 0  # 完全透明

        result = converter.delete_transparent_color(image)

        assert np.all(result[:, :, 3] == 0)

    def test_delete_transparent_color_opaque_unchanged(self, converter):
        """不透明ピクセルの色は変更されないこと"""
        image = np.zeros((10, 10, 4), dtype=np.uint8)
        image[:, :, 0] = 100  # B
        image[:, :, 1] = 150  # G
        image[:, :, 2] = 200  # R
        image[:, :, 3] = 255  # 不透明

        result = converter.delete_transparent_color(image)

        assert np.all(result[:, :, 0] == 100)
        assert np.all(result[:, :, 1] == 150)
        assert np.all(result[:, :, 2] == 200)

    def test_delete_transparent_color_rgb_unchanged(self, converter, rgb_image):
        """RGB 画像はそのまま返されること"""
        result = converter.delete_transparent_color(rgb_image)
        np.testing.assert_array_equal(result, rgb_image)

    def test_delete_transparent_color_mixed(self, converter):
        """透明と不透明が混在する画像"""
        image = np.zeros((10, 10, 4), dtype=np.uint8)
        image[:, :, 0] = 50
        image[:, :, 1] = 100
        image[:, :, 2] = 150
        image[:5, :, 3] = 0  # 上半分: 透明
        image[5:, :, 3] = 255  # 下半分: 不透明

        result = converter.delete_transparent_color(image)

        # 透明部分は白に
        assert np.all(result[:5, :, 0] == 255)
        assert np.all(result[:5, :, 1] == 255)
        assert np.all(result[:5, :, 2] == 255)
        # 不透明部分は元の色を維持
        assert np.all(result[5:, :, 0] == 50)
        assert np.all(result[5:, :, 1] == 100)
        assert np.all(result[5:, :, 2] == 150)
