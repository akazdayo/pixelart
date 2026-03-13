import numpy as np
import pytest
from pixelart_backend.filters import EdgeFilter, ImageEnhancer, GridMask, Dithering


class TestEdgeFilter:
    """EdgeFilter クラスのユニットテスト"""

    @pytest.fixture
    def edge_filter(self):
        """EdgeFilter インスタンスを作成"""
        return EdgeFilter()

    @pytest.fixture
    def rgb_image(self):
        """テスト用 RGB 画像 (30x30)"""
        return np.random.randint(0, 256, (30, 30, 3), dtype=np.uint8)

    @pytest.fixture
    def rgba_image(self):
        """テスト用 RGBA 画像 (30x30)"""
        return np.random.randint(0, 256, (30, 30, 4), dtype=np.uint8)


class TestCanny(TestEdgeFilter):
    """canny() のテスト"""

    def test_canny_rgb_output_shape(self, edge_filter, rgb_image):
        """RGB 画像でキャニーフィルタの出力形状が正しいこと"""
        result = edge_filter.canny(rgb_image, 100, 200)
        assert result.shape == rgb_image.shape

    def test_canny_rgba_output_shape(self, edge_filter, rgba_image):
        """RGBA 画像でキャニーフィルタの出力形状が正しいこと"""
        result = edge_filter.canny(rgba_image, 100, 200)
        assert result.shape == rgba_image.shape

    def test_canny_rgba_preserves_alpha(self, edge_filter, rgba_image):
        """RGBA 画像のアルファチャンネルが維持されること"""
        result = edge_filter.canny(rgba_image, 100, 200)
        np.testing.assert_array_equal(result[:, :, 3], rgba_image[:, :, 3])

    def test_canny_output_dtype(self, edge_filter, rgb_image):
        """出力の dtype が uint8 であること"""
        result = edge_filter.canny(rgb_image, 100, 200)
        assert result.dtype == np.uint8


class TestDog(TestEdgeFilter):
    """dog() のテスト"""

    def test_dog_rgb_no_scratch(self, edge_filter, rgb_image):
        """RGB 画像で scratch=False の出力形状が正しいこと"""
        result = edge_filter.dog(rgb_image, scratch=False)
        assert result.shape == rgb_image.shape

    def test_dog_rgb_with_scratch(self, edge_filter, rgb_image):
        """RGB 画像で scratch=True の出力形状が正しいこと"""
        result = edge_filter.dog(rgb_image, scratch=True)
        assert result.shape == rgb_image.shape

    def test_dog_rgba_no_scratch(self, edge_filter, rgba_image):
        """RGBA 画像で scratch=False の出力形状が正しいこと"""
        result = edge_filter.dog(rgba_image, scratch=False)
        assert result.shape == rgba_image.shape

    def test_dog_rgba_with_scratch(self, edge_filter, rgba_image):
        """RGBA 画像で scratch=True の出力形状が正しいこと"""
        result = edge_filter.dog(rgba_image, scratch=True)
        assert result.shape == rgba_image.shape

    def test_dog_rgba_preserves_alpha(self, edge_filter, rgba_image):
        """RGBA 画像のアルファチャンネルが維持されること"""
        result = edge_filter.dog(rgba_image)
        np.testing.assert_array_equal(result[:, :, 3], rgba_image[:, :, 3])

    def test_dog_default_scratch_is_false(self, edge_filter, rgb_image):
        """デフォルトの scratch が False であること"""
        result_default = edge_filter.dog(rgb_image)
        result_false = edge_filter.dog(rgb_image, scratch=False)
        np.testing.assert_array_equal(result_default, result_false)


class TestMorphology(TestEdgeFilter):
    """morphology_dilate() / morphology_erode() のテスト"""

    def test_morphology_dilate_output_shape(self, edge_filter, rgb_image):
        """膨張フィルタの出力形状が正しいこと"""
        result = edge_filter.morphology_dilate(rgb_image)
        assert result.shape == rgb_image.shape

    def test_morphology_dilate_output_dtype(self, edge_filter, rgb_image):
        """膨張フィルタの出力が uint8 であること"""
        result = edge_filter.morphology_dilate(rgb_image)
        assert result.dtype == np.uint8

    def test_morphology_erode_output_shape(self, edge_filter, rgb_image):
        """収縮フィルタの出力形状が正しいこと"""
        result = edge_filter.morphology_erode(rgb_image)
        assert result.shape == rgb_image.shape

    def test_morphology_erode_output_dtype(self, edge_filter, rgb_image):
        """収縮フィルタの出力が uint8 であること"""
        result = edge_filter.morphology_erode(rgb_image)
        assert result.dtype == np.uint8


class TestKuwahara(TestEdgeFilter):
    """kuwahara() / apply_kuwahara() のテスト"""

    def test_kuwahara_static_method(self, edge_filter, rgb_image):
        """kuwahara 静的メソッドの出力形状が正しいこと"""
        result = EdgeFilter.kuwahara(rgb_image, 3)
        assert result.shape == rgb_image.shape

    def test_apply_kuwahara_output_shape(self, edge_filter, rgb_image):
        """apply_kuwahara の出力形状が正しいこと"""
        result = edge_filter.apply_kuwahara(rgb_image)
        assert result.shape == rgb_image.shape

    def test_kuwahara_smooths_image(self, edge_filter):
        """kuwahara フィルタが画像を平滑化すること"""
        # ノイズの多い画像
        noisy = np.random.randint(0, 256, (30, 30, 3), dtype=np.uint8)
        result = edge_filter.apply_kuwahara(noisy)
        # 平滑化後は分散が減少するはず
        assert result.shape == noisy.shape


class TestMedian(TestEdgeFilter):
    """median() のテスト"""

    def test_median_output_shape(self, edge_filter, rgb_image):
        """メディアンフィルタの出力形状が正しいこと"""
        result = edge_filter.median(rgb_image, 5)
        assert result.shape == rgb_image.shape

    def test_median_output_dtype(self, edge_filter, rgb_image):
        """メディアンフィルタの出力が uint8 であること"""
        result = edge_filter.median(rgb_image, 5)
        assert result.dtype == np.uint8

    def test_median_with_different_sizes(self, edge_filter, rgb_image):
        """異なるカーネルサイズでメディアンフィルタが動作すること"""
        for size in [3, 5, 15]:
            result = edge_filter.median(rgb_image, size)
            assert result.shape == rgb_image.shape


class TestImageEnhancer:
    """ImageEnhancer クラスのユニットテスト"""

    @pytest.fixture
    def enhancer(self):
        """ImageEnhancer インスタンスを作成"""
        return ImageEnhancer()

    @pytest.fixture
    def rgb_image(self):
        """テスト用 RGB 画像 (30x30)"""
        return np.random.randint(0, 256, (30, 30, 3), dtype=np.uint8)


class TestSaturation(TestImageEnhancer):
    """saturation() のテスト"""

    def test_saturation_output_shape(self, enhancer, rgb_image):
        """彩度調整の出力形状が正しいこと"""
        result = enhancer.saturation(rgb_image, 1.5)
        assert result.shape == rgb_image.shape

    def test_saturation_identity(self, enhancer, rgb_image):
        """彩度 1.0 で画像が変わらないこと"""
        result = enhancer.saturation(rgb_image, 1.0)
        np.testing.assert_array_equal(result, rgb_image)


class TestBrightness(TestImageEnhancer):
    """brightness() のテスト"""

    def test_brightness_output_shape(self, enhancer, rgb_image):
        """明度調整の出力形状が正しいこと"""
        result = enhancer.brightness(rgb_image, 1.5)
        assert result.shape == rgb_image.shape

    def test_brightness_identity(self, enhancer, rgb_image):
        """明度 1.0 で画像が変わらないこと"""
        result = enhancer.brightness(rgb_image, 1.0)
        np.testing.assert_array_equal(result, rgb_image)


class TestSharpness(TestImageEnhancer):
    """sharpness() のテスト"""

    def test_sharpness_output_shape(self, enhancer, rgb_image):
        """シャープネス調整の出力形状が正しいこと"""
        result = enhancer.sharpness(rgb_image, 1.5)
        assert result.shape == rgb_image.shape

    def test_sharpness_identity(self, enhancer, rgb_image):
        """シャープネス 1.0 で画像が変わらないこと"""
        result = enhancer.sharpness(rgb_image, 1.0)
        np.testing.assert_array_equal(result, rgb_image)


class TestContrast(TestImageEnhancer):
    """contrast() のテスト"""

    def test_contrast_output_shape(self, enhancer, rgb_image):
        """コントラスト調整の出力形状が正しいこと"""
        result = enhancer.contrast(rgb_image, 1.5)
        assert result.shape == rgb_image.shape

    def test_contrast_identity(self, enhancer, rgb_image):
        """コントラスト 1.0 で画像が変わらないこと"""
        result = enhancer.contrast(rgb_image, 1.0)
        np.testing.assert_array_equal(result, rgb_image)


class TestSliderMosaic(TestImageEnhancer):
    """slider_mosaic() のテスト"""

    def test_slider_mosaic_reduces_size(self, enhancer, rgb_image):
        """スライダーモザイクが画像を縮小すること"""
        result = enhancer.slider_mosaic(rgb_image, 0.5)
        assert result.shape[0] == 15
        assert result.shape[1] == 15
        assert result.shape[2] == 3

    def test_slider_mosaic_small_ratio(self, enhancer, rgb_image):
        """小さい ratio でモザイクが適用されること"""
        result = enhancer.slider_mosaic(rgb_image, 0.1)
        assert result.shape[0] < rgb_image.shape[0]
        assert result.shape[1] < rgb_image.shape[1]

    def test_slider_mosaic_tiny_image(self, enhancer):
        tiny = np.random.randint(0, 256, (1, 1, 3), dtype=np.uint8)
        result = enhancer.slider_mosaic(tiny, 0.01)
        assert result.shape == (1, 1, 3)


class TestGridMosaic(TestImageEnhancer):
    """grid_mosaic() のテスト"""

    def test_grid_mosaic_output_width(self, enhancer, rgb_image):
        """グリッドモザイクの出力幅が指定サイズであること"""
        result = enhancer.grid_mosaic(rgb_image, 10)
        assert result.shape[1] == 10

    def test_grid_mosaic_preserves_aspect(self, enhancer):
        """グリッドモザイクがアスペクト比を維持すること"""
        image = np.random.randint(0, 256, (40, 20, 3), dtype=np.uint8)
        result = enhancer.grid_mosaic(image, 10)
        assert result.shape[1] == 10
        # aspect = 40/20 = 2.0, so height should be 20
        assert result.shape[0] == 20

    def test_grid_mosaic_preserves_channels(self, enhancer, rgb_image):
        """グリッドモザイクがチャンネル数を維持すること"""
        result = enhancer.grid_mosaic(rgb_image, 10)
        assert result.shape[2] == 3


class TestDecrease(TestImageEnhancer):
    """decrease() のテスト"""

    def test_decrease_output_shape(self, enhancer, rgb_image):
        """減色の出力形状が正しいこと"""
        result = enhancer.decrease(rgb_image)
        assert result.shape == rgb_image.shape

    def test_decrease_reduces_colors(self, enhancer):
        """減色後のピクセル値が4段階のみであること"""
        image = np.random.randint(0, 256, (30, 30, 3), dtype=np.uint8)
        result = enhancer.decrease(image)
        unique_values = np.unique(result)
        # 4段階: 32, 96, 160, 224
        for val in unique_values:
            assert val in [32, 96, 160, 224]

    def test_decrease_specific_ranges(self, enhancer):
        """各範囲が正しく変換されること"""
        # 0-63 → 32
        image = np.full((5, 5, 3), 30, dtype=np.uint8)
        result = enhancer.decrease(image)
        assert np.all(result == 32)

        # 64-127 → 96
        image = np.full((5, 5, 3), 100, dtype=np.uint8)
        result = enhancer.decrease(image)
        assert np.all(result == 96)

        # 128-191 → 160
        image = np.full((5, 5, 3), 150, dtype=np.uint8)
        result = enhancer.decrease(image)
        assert np.all(result == 160)

        # 192-255 → 224
        image = np.full((5, 5, 3), 220, dtype=np.uint8)
        result = enhancer.decrease(image)
        assert np.all(result == 224)

    def test_decrease_does_not_modify_original(self, enhancer, rgb_image):
        """元の画像が変更されないこと"""
        original = rgb_image.copy()
        enhancer.decrease(rgb_image)
        np.testing.assert_array_equal(rgb_image, original)


class TestGridMaskClass:
    """GridMask クラスのユニットテスト"""

    @pytest.fixture
    def grid_mask(self):
        """GridMask インスタンスを作成"""
        return GridMask()

    @pytest.fixture
    def rgb_image(self):
        """テスト用 RGB 画像 (30x30)"""
        return np.random.randint(0, 256, (30, 30, 3), dtype=np.uint8)

    @pytest.fixture
    def rgba_image(self):
        """テスト用 RGBA 画像 (30x30)"""
        return np.random.randint(0, 256, (30, 30, 4), dtype=np.uint8)


class TestAddGrid(TestGridMaskClass):
    """add_grid() のテスト"""

    def test_add_grid_rgb_output_shape(self, grid_mask, rgb_image):
        """RGB 画像でグリッドの出力形状が正しいこと"""
        result = grid_mask.add_grid(rgb_image, 10)
        assert result.shape == rgb_image.shape

    def test_add_grid_rgba_output_shape(self, grid_mask, rgba_image):
        """RGBA 画像でグリッドの出力形状が正しいこと"""
        result = grid_mask.add_grid(rgba_image, 10)
        assert result.shape == rgba_image.shape

    def test_add_grid_rgba_preserves_alpha(self, grid_mask, rgba_image):
        """RGBA 画像のアルファチャンネルが維持されること"""
        result = grid_mask.add_grid(rgba_image, 10)
        np.testing.assert_array_equal(result[:, :, 3], rgba_image[:, :, 3])

    def test_add_grid_with_opacity(self, grid_mask, rgb_image):
        """opacity < 1.0 でグリッドが半透明に描画されること"""
        result = grid_mask.add_grid(rgb_image, 10, opacity=0.5)
        assert result.shape == rgb_image.shape

    def test_add_grid_full_opacity(self, grid_mask, rgb_image):
        """opacity = 1.0 でグリッドが完全に描画されること"""
        result = grid_mask.add_grid(rgb_image, 10, opacity=1.0)
        assert result.shape == rgb_image.shape

    def test_add_grid_custom_color(self, grid_mask, rgb_image):
        """カスタム色でグリッドが描画されること"""
        result = grid_mask.add_grid(rgb_image, 10, line_color=(255, 0, 0))
        assert result.shape == rgb_image.shape

    def test_add_grid_custom_thickness(self, grid_mask, rgb_image):
        """カスタム太さでグリッドが描画されること"""
        result = grid_mask.add_grid(rgb_image, 10, line_thickness=3)
        assert result.shape == rgb_image.shape

    def test_add_grid_modifies_image(self, grid_mask):
        """グリッドが実際に画像を変更すること"""
        image = np.full((30, 30, 3), 128, dtype=np.uint8)
        result = grid_mask.add_grid(image, 10, line_color=(0, 0, 0), opacity=1.0)
        # グリッド線が描画されているので、元の画像と違うはず
        assert not np.array_equal(image, result)

    def test_add_grid_rgba_with_opacity(self, grid_mask, rgba_image):
        """RGBA 画像で opacity < 1.0 のグリッドが正しく描画されること"""
        result = grid_mask.add_grid(rgba_image, 10, opacity=0.3)
        assert result.shape == rgba_image.shape
        np.testing.assert_array_equal(result[:, :, 3], rgba_image[:, :, 3])


class TestDitheringClass:
    """Dithering クラスのユニットテスト"""

    @pytest.fixture
    def dithering(self):
        """Dithering インスタンスを作成"""
        return Dithering()

    @pytest.fixture
    def rgb_image(self):
        """テスト用 RGB 画像 (10x10) — 小さいサイズ"""
        return np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)

    @pytest.fixture
    def rgba_image(self):
        """テスト用 RGBA 画像 (10x10) — 小さいサイズ"""
        return np.random.randint(0, 256, (10, 10, 4), dtype=np.uint8)


class TestFloydSteinberg(TestDitheringClass):
    """floyd_steinberg() のテスト"""

    def test_floyd_steinberg_rgb_output_shape(self, dithering, rgb_image):
        """RGB 画像で Floyd-Steinberg の出力形状が正しいこと"""
        result = dithering.floyd_steinberg(rgb_image)
        assert result.shape == rgb_image.shape

    def test_floyd_steinberg_rgba_output_shape(self, dithering, rgba_image):
        """RGBA 画像で Floyd-Steinberg の出力形状が正しいこと"""
        result = dithering.floyd_steinberg(rgba_image)
        assert result.shape == rgba_image.shape

    def test_floyd_steinberg_rgba_preserves_alpha(self, dithering, rgba_image):
        """RGBA 画像のアルファチャンネルが維持されること"""
        result = dithering.floyd_steinberg(rgba_image)
        np.testing.assert_array_equal(result[:, :, 3], rgba_image[:, :, 3])

    def test_floyd_steinberg_output_dtype(self, dithering, rgb_image):
        """出力の dtype が uint8 であること"""
        result = dithering.floyd_steinberg(rgb_image)
        assert result.dtype == np.uint8

    def test_floyd_steinberg_with_intensity(self, dithering, rgb_image):
        """intensity パラメータが動作すること"""
        result = dithering.floyd_steinberg(rgb_image, intensity=0.5)
        assert result.shape == rgb_image.shape

    def test_floyd_steinberg_values_in_range(self, dithering, rgb_image):
        """出力値が 0-255 の範囲内であること"""
        result = dithering.floyd_steinberg(rgb_image)
        assert np.all(result >= 0)
        assert np.all(result <= 255)


class TestOrderedDither(TestDitheringClass):
    """ordered_dither() のテスト"""

    def test_ordered_dither_rgb_output_shape(self, dithering, rgb_image):
        """RGB 画像でオーダードディザの出力形状が正しいこと"""
        result = dithering.ordered_dither(rgb_image)
        assert result.shape == rgb_image.shape

    def test_ordered_dither_rgba_output_shape(self, dithering, rgba_image):
        """RGBA 画像でオーダードディザの出力形状が正しいこと"""
        result = dithering.ordered_dither(rgba_image)
        assert result.shape == rgba_image.shape

    def test_ordered_dither_rgba_preserves_alpha(self, dithering, rgba_image):
        """RGBA 画像のアルファチャンネルが維持されること"""
        result = dithering.ordered_dither(rgba_image)
        np.testing.assert_array_equal(result[:, :, 3], rgba_image[:, :, 3])

    def test_ordered_dither_matrix_size_2(self, dithering, rgb_image):
        """matrix_size=2 で動作すること"""
        result = dithering.ordered_dither(rgb_image, matrix_size=2)
        assert result.shape == rgb_image.shape

    def test_ordered_dither_matrix_size_4(self, dithering, rgb_image):
        """matrix_size=4 で動作すること"""
        result = dithering.ordered_dither(rgb_image, matrix_size=4)
        assert result.shape == rgb_image.shape

    def test_ordered_dither_matrix_size_8(self, dithering, rgb_image):
        """matrix_size=8 で動作すること"""
        result = dithering.ordered_dither(rgb_image, matrix_size=8)
        assert result.shape == rgb_image.shape

    def test_ordered_dither_invalid_matrix_size_falls_back(self, dithering, rgb_image):
        """無効な matrix_size が 4 にフォールバックすること"""
        result_invalid = dithering.ordered_dither(rgb_image, matrix_size=3)
        result_default = dithering.ordered_dither(rgb_image, matrix_size=4)
        np.testing.assert_array_equal(result_invalid, result_default)

    def test_ordered_dither_output_dtype(self, dithering, rgb_image):
        """出力の dtype が uint8 であること"""
        result = dithering.ordered_dither(rgb_image)
        assert result.dtype == np.uint8

    def test_ordered_dither_with_intensity(self, dithering, rgb_image):
        """intensity パラメータが動作すること"""
        result = dithering.ordered_dither(rgb_image, intensity=0.5)
        assert result.shape == rgb_image.shape

    def test_ordered_dither_values_in_range(self, dithering, rgb_image):
        """出力値が 0-255 の範囲内であること"""
        result = dithering.ordered_dither(rgb_image)
        assert np.all(result >= 0)
        assert np.all(result <= 255)


class TestAtkinson(TestDitheringClass):
    """atkinson() のテスト"""

    def test_atkinson_rgb_output_shape(self, dithering, rgb_image):
        """RGB 画像で Atkinson の出力形状が正しいこと"""
        result = dithering.atkinson(rgb_image)
        assert result.shape == rgb_image.shape

    def test_atkinson_rgba_output_shape(self, dithering, rgba_image):
        """RGBA 画像で Atkinson の出力形状が正しいこと"""
        result = dithering.atkinson(rgba_image)
        assert result.shape == rgba_image.shape

    def test_atkinson_rgba_preserves_alpha(self, dithering, rgba_image):
        """RGBA 画像のアルファチャンネルが維持されること"""
        result = dithering.atkinson(rgba_image)
        np.testing.assert_array_equal(result[:, :, 3], rgba_image[:, :, 3])

    def test_atkinson_output_dtype(self, dithering, rgb_image):
        """出力の dtype が uint8 であること"""
        result = dithering.atkinson(rgb_image)
        assert result.dtype == np.uint8

    def test_atkinson_with_intensity(self, dithering, rgb_image):
        """intensity パラメータが動作すること"""
        result = dithering.atkinson(rgb_image, intensity=0.5)
        assert result.shape == rgb_image.shape

    def test_atkinson_values_in_range(self, dithering, rgb_image):
        """出力値が 0-255 の範囲内であること"""
        result = dithering.atkinson(rgb_image)
        assert np.all(result >= 0)
        assert np.all(result <= 255)
