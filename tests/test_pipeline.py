import base64
import numpy as np
import cv2
import pytest
from pixelart_backend.pipeline import (
    ProcessingOptions,
    ProcessingResult,
    _hex_to_rgb,
    cv_to_base64,
    process_image,
)


class TestCvToBase64:
    """cv_to_base64() のテスト"""

    @pytest.fixture
    def bgr_image(self):
        """テスト用 BGR 画像 (10x10)"""
        return np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)

    @pytest.fixture
    def bgra_image(self):
        """テスト用 BGRA 画像 (10x10)"""
        return np.random.randint(0, 256, (10, 10, 4), dtype=np.uint8)

    def test_returns_string(self, bgr_image):
        """文字列が返されること"""
        result = cv_to_base64(bgr_image)
        assert isinstance(result, str)

    def test_valid_base64(self, bgr_image):
        """有効な base64 エンコードであること"""
        result = cv_to_base64(bgr_image)
        # base64 デコードが成功すること
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    def test_decodable_to_png(self, bgr_image):
        """デコードした結果が有効な PNG 画像であること"""
        result = cv_to_base64(bgr_image)
        decoded = base64.b64decode(result)
        # PNG ヘッダーチェック
        assert decoded[:4] == b"\x89PNG"

    def test_roundtrip_bgr(self, bgr_image):
        """BGR 画像のラウンドトリップが成功すること"""
        result = cv_to_base64(bgr_image)
        decoded = base64.b64decode(result)
        img_array = np.frombuffer(decoded, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        assert img is not None
        assert img.shape == bgr_image.shape
        np.testing.assert_array_equal(img, bgr_image)

    def test_bgra_image(self, bgra_image):
        """BGRA 画像でも動作すること"""
        result = cv_to_base64(bgra_image)
        decoded = base64.b64decode(result)
        assert decoded[:4] == b"\x89PNG"

    def test_roundtrip_bgra(self, bgra_image):
        """BGRA 画像のラウンドトリップが成功すること"""
        result = cv_to_base64(bgra_image)
        decoded = base64.b64decode(result)
        img_array = np.frombuffer(decoded, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        assert img is not None
        assert img.shape == bgra_image.shape
        np.testing.assert_array_equal(img, bgra_image)


class TestProcessingOptions:
    """ProcessingOptions のテスト"""

    def test_default_values(self):
        """デフォルト値が正しく設定されること"""
        opts = ProcessingOptions()
        assert opts.palette == "pyxel"
        assert opts.custom_palette == []
        assert opts.no_convert is False
        assert opts.mosaic_mode == "slider"
        assert opts.slider_ratio == 0.3
        assert opts.grid_size == 256
        assert opts.saturation == 1.0
        assert opts.brightness == 1.0
        assert opts.contrast == 1.0
        assert opts.sharpness == 1.0
        assert opts.dithering is False
        assert opts.enable_grid is False
        assert opts.delete_alpha is False
        assert opts.decrease_color is False

    def test_custom_values(self):
        """カスタム値で初期化できること"""
        opts = ProcessingOptions(
            palette="cold",
            saturation=1.5,
            brightness=0.8,
            dithering=True,
            dithering_method="Ordered",
            dither_matrix_size=8,
        )
        assert opts.palette == "cold"
        assert opts.saturation == 1.5
        assert opts.brightness == 0.8
        assert opts.dithering is True
        assert opts.dithering_method == "Ordered"
        assert opts.dither_matrix_size == 8


class TestProcessingResult:
    """ProcessingResult のテスト"""

    def test_basic_construction(self):
        """基本的なコンストラクタが動作すること"""
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        result = ProcessingResult(image=img, was_resized=False)
        assert result.image is img
        assert result.was_resized is False
        assert result.ai_hex_colors is None

    def test_with_ai_colors(self):
        """AI カラー付きで構築できること"""
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        result = ProcessingResult(
            image=img,
            was_resized=True,
            ai_hex_colors=["#ff0000", "#00ff00"],
        )
        assert result.was_resized is True
        assert result.ai_hex_colors == ["#ff0000", "#00ff00"]


class TestProcessImage:
    """process_image() のテスト"""

    @pytest.fixture
    def rgb_image(self):
        """テスト用 RGB 画像 (50x50)"""
        return np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)

    @pytest.fixture
    def rgba_image(self):
        """テスト用 RGBA 画像 (50x50)"""
        return np.random.randint(0, 256, (50, 50, 4), dtype=np.uint8)

    def test_default_options_returns_result(self, rgb_image):
        """デフォルトオプションで ProcessingResult が返されること"""
        result = process_image(rgb_image)
        assert isinstance(result, ProcessingResult)
        assert isinstance(result.image, np.ndarray)
        assert result.was_resized is False

    def test_no_convert_mode(self, rgb_image):
        """no_convert=True でカラー変換がスキップされること"""
        opts = ProcessingOptions(no_convert=True)
        result = process_image(rgb_image, opts)
        assert isinstance(result, ProcessingResult)
        assert result.image.dtype == np.uint8

    def test_greyscale_input(self):
        """グレースケール画像が処理できること"""
        gray = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        opts = ProcessingOptions(no_convert=True)
        result = process_image(gray, opts)
        # グレースケールが BGR に変換されること
        assert result.image.ndim == 3

    def test_none_options_uses_defaults(self, rgb_image):
        """options=None でデフォルトが使用されること"""
        result = process_image(rgb_image, None)
        assert isinstance(result, ProcessingResult)

    def test_grid_mosaic_mode(self, rgb_image):
        """grid モザイクモードで処理できること"""
        opts = ProcessingOptions(mosaic_mode="grid", grid_size=10, no_convert=True)
        result = process_image(rgb_image, opts)
        assert isinstance(result.image, np.ndarray)

    def test_slider_mosaic_mode(self, rgb_image):
        """slider モザイクモードで処理できること"""
        opts = ProcessingOptions(
            mosaic_mode="slider", slider_ratio=0.5, no_convert=True
        )
        result = process_image(rgb_image, opts)
        assert isinstance(result.image, np.ndarray)

    def test_dithering_floyd_steinberg(self, rgb_image):
        """Floyd-Steinberg ディザリングが適用できること"""
        opts = ProcessingOptions(
            dithering=True,
            dithering_method="Floyd-Steinberg",
            dither_intensity=0.5,
            no_convert=True,
            no_expand=True,
        )
        result = process_image(rgb_image, opts)
        assert result.image.dtype == np.uint8

    def test_dithering_ordered(self, rgb_image):
        """Ordered ディザリングが適用できること"""
        opts = ProcessingOptions(
            dithering=True,
            dithering_method="Ordered",
            dither_matrix_size=4,
            dither_intensity=0.5,
            no_convert=True,
            no_expand=True,
        )
        result = process_image(rgb_image, opts)
        assert result.image.dtype == np.uint8

    def test_dithering_atkinson(self, rgb_image):
        """Atkinson ディザリングが適用できること"""
        opts = ProcessingOptions(
            dithering=True,
            dithering_method="Atkinson",
            dither_intensity=0.5,
            no_convert=True,
            no_expand=True,
        )
        result = process_image(rgb_image, opts)
        assert result.image.dtype == np.uint8

    def test_enhancements(self, rgb_image):
        """画像強調パラメータが適用できること"""
        opts = ProcessingOptions(
            saturation=1.5,
            brightness=1.2,
            contrast=0.8,
            sharpness=1.3,
            no_convert=True,
        )
        result = process_image(rgb_image, opts)
        assert isinstance(result.image, np.ndarray)

    def test_custom_palette(self, rgb_image):
        """カスタムパレットで変換できること"""
        opts = ProcessingOptions(
            palette="Custom",
            custom_palette=[[0, 0, 0], [255, 255, 255]],
        )
        result = process_image(rgb_image, opts)
        assert isinstance(result.image, np.ndarray)

    def test_custom_palette_output_colors_are_in_palette(self):
        image = np.full((24, 24, 3), [128, 128, 128], dtype=np.uint8)
        custom_palette = [[0, 0, 0], [255, 255, 255]]
        opts = ProcessingOptions(
            palette="Custom",
            custom_palette=custom_palette,
            slider_ratio=1.0,
            no_expand=True,
        )
        result = process_image(image, opts)

        unique_colors = np.unique(result.image.reshape(-1, 3), axis=0).tolist()
        for color in unique_colors:
            assert color in custom_palette

    def test_preset_palette(self, rgb_image):
        """プリセットパレットで変換できること"""
        opts = ProcessingOptions(palette="pyxel")
        result = process_image(rgb_image, opts)
        assert isinstance(result.image, np.ndarray)

    def test_no_expand(self, rgb_image):
        """no_expand=True で元サイズにリサイズしないこと"""
        opts = ProcessingOptions(
            no_expand=True,
            slider_ratio=0.5,
            no_convert=True,
        )
        result = process_image(rgb_image, opts)
        # モザイクで縮小されているはず
        assert result.image.shape[0] < rgb_image.shape[0]

    def test_decrease_color(self, rgb_image):
        """decrease_color が適用できること"""
        opts = ProcessingOptions(no_convert=True, decrease_color=True)
        result = process_image(rgb_image, opts)
        assert result.image.dtype == np.uint8

    def test_delete_alpha(self, rgba_image):
        """delete_alpha が適用できること"""
        opts = ProcessingOptions(no_convert=True, delete_alpha=True)
        result = process_image(rgba_image, opts)
        assert isinstance(result.image, np.ndarray)

    def test_grid_overlay(self, rgb_image):
        """グリッドオーバーレイが適用できること"""
        opts = ProcessingOptions(
            no_convert=True,
            enable_grid=True,
            grid_line_color=(0, 0, 0),
            grid_opacity=0.5,
        )
        result = process_image(rgb_image, opts)
        assert isinstance(result.image, np.ndarray)

    def test_grid_overlay_grid_mode(self, rgb_image):
        """grid モードでグリッドオーバーレイが適用できること"""
        opts = ProcessingOptions(
            mosaic_mode="grid",
            grid_size=10,
            no_convert=True,
            enable_grid=True,
        )
        result = process_image(rgb_image, opts)
        assert isinstance(result.image, np.ndarray)

    def test_large_image_is_resized(self):
        """FullHD を超える画像がリサイズされること"""
        large = np.random.randint(0, 256, (2000, 2000, 3), dtype=np.uint8)
        opts = ProcessingOptions(no_convert=True)
        result = process_image(large, opts)
        assert result.was_resized is True

    def test_small_image_not_resized(self, rgb_image):
        """小さい画像がリサイズされないこと"""
        opts = ProcessingOptions(no_convert=True)
        result = process_image(rgb_image, opts)
        assert result.was_resized is False

    def test_dog_filter(self, rgb_image):
        """DoG フィルタが適用できること"""
        opts = ProcessingOptions(dog_filter=True, no_convert=True)
        result = process_image(rgb_image, opts)
        assert isinstance(result.image, np.ndarray)

    def test_scratch_filter(self, rgb_image):
        """scratch フィルタが適用できること"""
        opts = ProcessingOptions(scratch=True, no_convert=True)
        result = process_image(rgb_image, opts)
        assert isinstance(result.image, np.ndarray)

    def test_scratch_disables_dog(self, rgb_image):
        """scratch=True の時 dog_filter が無視されること"""
        opts = ProcessingOptions(scratch=True, dog_filter=True, no_convert=True)
        result = process_image(rgb_image, opts)
        assert isinstance(result.image, np.ndarray)

    def test_morphology_filter(self, rgb_image):
        """morphology フィルタが適用できること"""
        opts = ProcessingOptions(morphology=True, no_convert=True)
        result = process_image(rgb_image, opts)
        assert isinstance(result.image, np.ndarray)

    def test_scratch_disables_morphology(self, rgb_image):
        """scratch=True の時 morphology が無視されること"""
        opts = ProcessingOptions(scratch=True, morphology=True, no_convert=True)
        result = process_image(rgb_image, opts)
        assert isinstance(result.image, np.ndarray)

    def test_kuwahara_filter(self, rgb_image):
        """kuwahara フィルタが適用できること"""
        opts = ProcessingOptions(kuwahara=True, no_convert=True)
        result = process_image(rgb_image, opts)
        assert isinstance(result.image, np.ndarray)

    def test_median_filter(self, rgb_image):
        """median フィルタが適用できること"""
        opts = ProcessingOptions(median=True, no_convert=True)
        result = process_image(rgb_image, opts)
        assert isinstance(result.image, np.ndarray)

    def test_delete_transparent(self, rgba_image):
        """delete_transparent が適用できること"""
        opts = ProcessingOptions(delete_transparent=True, no_convert=True)
        result = process_image(rgba_image, opts)
        assert isinstance(result.image, np.ndarray)

    def test_ai_palette(self, rgb_image, monkeypatch):
        ai_hex_colors = ["#000000", "#ffffff", "#ff0000"]

        def fake_get_color(self, image, color_num, iter):
            return ai_hex_colors

        monkeypatch.setattr("pixelart_backend.pipeline.AI.get_color", fake_get_color)

        opts = ProcessingOptions(
            palette="AI",
            ai_colors=3,
            ai_iterations=10,
            slider_ratio=1.0,
            no_expand=True,
        )
        result = process_image(rgb_image, opts)

        assert isinstance(result, ProcessingResult)
        assert result.ai_hex_colors == ai_hex_colors

        expected_palette = [[0, 0, 0], [255, 255, 255], [255, 0, 0]]
        unique_colors = np.unique(result.image.reshape(-1, 3), axis=0).tolist()
        for color in unique_colors:
            assert color in expected_palette

    def test_ai_palette_uniform_red_image(self):
        red_rgb = np.zeros((24, 24, 3), dtype=np.uint8)
        red_rgb[:, :, 0] = 255

        opts = ProcessingOptions(
            palette="AI",
            ai_colors=1,
            ai_iterations=20,
            slider_ratio=1.0,
            no_expand=True,
        )
        result = process_image(red_rgb, opts)

        assert result.ai_hex_colors is not None
        hex_code = result.ai_hex_colors[0]
        r = int(hex_code[1:3], 16)
        b = int(hex_code[5:7], 16)
        assert r > b

        unique_color = np.unique(result.image.reshape(-1, 3), axis=0)
        assert unique_color.shape[0] == 1
        assert int(unique_color[0][0]) >= int(unique_color[0][2])


class TestHexToRgb:
    def test_hex_to_rgb_with_hash_prefix(self):
        assert _hex_to_rgb("#1a2b3c") == [26, 43, 60]

    def test_hex_to_rgb_without_hash_prefix(self):
        assert _hex_to_rgb("ABCDEF") == [171, 205, 239]
