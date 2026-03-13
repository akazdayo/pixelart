import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from pixelart_backend.convert import Convert, list_palette_names, load_palette_rows, PALETTE_SUFFIX

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


class TestCandidatePaths:
    """_candidate_paths() のテスト"""

    def test_empty_string_raises_file_not_found(self):
        """空文字列で FileNotFoundError が発生すること"""
        with pytest.raises(FileNotFoundError):
            Convert._candidate_paths("")

    def test_whitespace_only_raises_file_not_found(self):
        """空白文字のみで FileNotFoundError が発生すること"""
        with pytest.raises(FileNotFoundError):
            Convert._candidate_paths("   ")


class TestCsvStems:
    """_csv_stems() のテスト"""

    def test_csv_stems_returns_names(self):
        """CSV ファイルの stem が返されること"""
        mock_files = []
        for name in ["pyxel.csv", "cold.csv", "readme.txt"]:
            f = MagicMock()
            f.is_file.return_value = True
            f.name = name
            mock_files.append(f)
        # ディレクトリ
        d = MagicMock()
        d.is_file.return_value = False
        d.name = "subdir"
        mock_files.append(d)

        result = Convert._csv_stems(mock_files)
        assert "pyxel" in result
        assert "cold" in result
        # .txt は含まれない
        assert "readme" not in result
        # ディレクトリは含まれない
        assert "subdir" not in result

    def test_csv_stems_empty_list(self):
        """空リストで空リストが返されること"""
        result = Convert._csv_stems([])
        assert result == []


class TestListColorPalettes:
    """list_color_palettes() のテスト"""

    def test_list_color_palettes_returns_tuple(self):
        """タプルが返されること"""
        result = Convert.list_color_palettes()
        assert isinstance(result, tuple)

    def test_list_color_palettes_includes_pyxel(self):
        """pyxel が含まれること"""
        result = Convert.list_color_palettes()
        assert "pyxel" in result

    def test_list_color_palettes_includes_known_palettes(self):
        """既知のパレットが含まれること"""
        result = Convert.list_color_palettes()
        for name in ["cold", "gold", "pale", "pastel", "warm", "rainbow"]:
            assert name in result

    def test_list_color_palettes_sorted(self):
        """ソートされて返されること"""
        result = Convert.list_color_palettes()
        assert result == tuple(sorted(result))

    def test_list_color_palettes_package_color_dir_not_exist(self):
        """package の color/ ディレクトリが存在しない場合でも動作すること"""
        with patch("pixelart_backend.convert.files") as mock_files:
            mock_package_root = MagicMock()
            mock_package_color = MagicMock()
            mock_package_color.is_dir.return_value = False
            mock_package_root.joinpath.return_value = mock_package_color
            mock_files.return_value = mock_package_root
            result = Convert.list_color_palettes()
            assert isinstance(result, tuple)
            # project の color/ からは取得できるので空ではない
            assert len(result) > 0

    def test_list_color_palettes_package_color_dir_exists(self):
        """package の color/ ディレクトリが存在する場合にパレットが追加されること"""
        with patch("pixelart_backend.convert.files") as mock_files:
            # package color dir has CSV files
            mock_csv_file = MagicMock()
            mock_csv_file.is_file.return_value = True
            mock_csv_file.name = "pkg_only.csv"

            mock_package_color = MagicMock()
            mock_package_color.is_dir.return_value = True
            mock_package_color.iterdir.return_value = iter([mock_csv_file])

            mock_package_root = MagicMock()
            mock_package_root.joinpath.return_value = mock_package_color
            mock_files.return_value = mock_package_root

            result = Convert.list_color_palettes()
            assert isinstance(result, tuple)
            assert "pkg_only" in result


class TestResolvePalettePathFallbacks(TestConvert):
    """_resolve_palette_path() のフォールバックパスのテスト"""

    def test_resolve_via_package_resource(self):
        """パッケージリソースから解決できること (line 72)"""
        with patch("pixelart_backend.convert.files") as mock_files:
            mock_resource = MagicMock()
            mock_resource.is_file.return_value = True
            mock_package_root = MagicMock()
            mock_package_root.joinpath.return_value = mock_resource
            mock_files.return_value = mock_package_root
            # 直接パスは存在しないように設定
            with patch.object(Path, "is_file", return_value=False):
                result = Convert._resolve_palette_path("some_palette.csv")
                assert result is mock_resource

    def test_resolve_via_project_root_fallback(self):
        """プロジェクトルートのフォールバックで解決できること (line 76)"""
        # _resolve_palette_path は candidates を順に試す
        # 「直接パス」と「パッケージリソース」が見つからず、「プロジェクトルート」で見つかるケース
        project_root = Path(__file__).resolve().parent.parent
        # pyxel.csv はプロジェクトの color/ にある
        # candidate "pyxel.csv" → direct_path(pyxel.csv).is_file() = False (CWD に pyxel.csv はない)
        # → package_root.joinpath("pyxel.csv").is_file() = False
        # → project_root / "pyxel.csv" → False
        # → project_root / "color" / "pyxel.csv" → True (line 81 ではなく)
        # candidate "color/pyxel.csv" → direct_path(color/pyxel.csv).is_file() = True
        # ↑ CWD がプロジェクトルートなので直接パスで見つかってしまう
        #
        # project_root フォールバック (line 76) を叩くには:
        # direct_path も package_root も見つからない、が project_root / candidate で見つかるケース
        with patch("pixelart_backend.convert.files") as mock_files:
            mock_resource = MagicMock()
            mock_resource.is_file.return_value = False
            mock_package_root = MagicMock()
            mock_package_root.joinpath.return_value = mock_resource
            mock_files.return_value = mock_package_root

            call_count = [0]
            orig_is_file = Path.is_file

            def selective_is_file(self_path):
                s = str(self_path)
                # 直接パスは見つからない
                if s == "color/pyxel.csv" or s == "pyxel.csv":
                    return False
                return orig_is_file(self_path)

            with patch.object(Path, "is_file", selective_is_file):
                result = Convert._resolve_palette_path("color/pyxel.csv")
                # project_root / "color/pyxel.csv" で解決される
                assert Path(str(result)).name == "pyxel.csv"


class TestResolvePalettePathInjection(TestConvert):
    """_resolve_palette_path() のテスト (injectable roots)"""

    def test_resolve_via_injected_project_root(self, tmp_path):
        """_project_root を注入してプロジェクトルートフォールバックで解決できること"""
        color_dir = tmp_path / "color"
        color_dir.mkdir()
        csv_file = color_dir / "test_pal.csv"
        csv_file.write_text("0,0,0")

        mock_pkg = MagicMock()
        mock_pkg.joinpath.return_value = MagicMock(is_file=MagicMock(return_value=False))

        result = Convert._resolve_palette_path(
            "color/test_pal.csv",
            _project_root=tmp_path,
            _package_root=mock_pkg,
        )
        assert str(result).endswith("test_pal.csv")

    def test_resolve_via_injected_package_root(self, tmp_path):
        """_package_root を注入してパッケージリソースで解決できること"""
        color_dir = tmp_path / "color"
        color_dir.mkdir()
        csv_file = color_dir / "pkg_pal.csv"
        csv_file.write_text("255,0,0")

        mock_pkg = MagicMock()
        mock_resource = MagicMock()
        mock_resource.is_file.return_value = True
        mock_pkg.joinpath.return_value = mock_resource

        # direct path won't exist; package_root.joinpath will find it
        result = Convert._resolve_palette_path(
            "color/pkg_pal.csv",
            _project_root=tmp_path / "nonexistent",
            _package_root=mock_pkg,
        )
        assert result is mock_resource

    def test_resolve_not_found_with_injection(self, tmp_path):
        """injectable roots でも見つからない場合は FileNotFoundError"""
        mock_pkg = MagicMock()
        mock_pkg.joinpath.return_value = MagicMock(is_file=MagicMock(return_value=False))

        with pytest.raises(FileNotFoundError):
            Convert._resolve_palette_path(
                "nonexistent.csv",
                _project_root=tmp_path,
                _package_root=mock_pkg,
            )


class TestDoubleSuffixBug(TestConvert):
    """.csv.csv 二重サフィックスバグの回帰テスト"""

    def test_convert_with_csv_suffix_no_double(self, converter, rgb_image):
        """パレット名に .csv が付いていても .csv.csv にならないこと"""
        # pyxel.csv should resolve correctly, not look for pyxel.csv.csv
        result = converter.convert(rgb_image, "pyxel.csv")
        assert result.shape == rgb_image.shape

    def test_convert_without_csv_suffix(self, converter, rgb_image):
        """.csv なしのパレット名でも動作すること"""
        result = converter.convert(rgb_image, "pyxel")
        assert result.shape == rgb_image.shape


class TestDeleteAlphaNonMutating(TestConvert):
    """delete_alpha() が入力を変更しないことのテスト"""

    def test_delete_alpha_does_not_mutate_input(self, converter):
        """入力画像が変更されないこと"""
        image = np.zeros((10, 10, 4), dtype=np.uint8)
        image[:, :, 0] = 100
        image[:, :, 1] = 150
        image[:, :, 2] = 200
        image[:, :, 3] = 128
        original = image.copy()

        converter.delete_alpha(image)

        np.testing.assert_array_equal(image, original)

class TestModuleLevelFunctions:
    """モジュールレベル関数のテスト"""

    def test_list_palette_names_returns_tuple(self):
        """list_palette_names() がタプルを返すこと"""
        result = list_palette_names()
        assert isinstance(result, tuple)
        assert "pyxel" in result

    def test_list_palette_names_matches_class_method(self):
        """list_palette_names() が Convert.list_color_palettes() と同じ結果を返すこと"""
        assert list_palette_names() == Convert.list_color_palettes()

    def test_load_palette_rows_with_suffix(self):
        """load_palette_rows() が .csv 付きで動作すること"""
        result = load_palette_rows("pyxel.csv")
        assert isinstance(result, list)
        assert len(result) == 16

    def test_load_palette_rows_without_suffix(self):
        """load_palette_rows() が .csv なしで動作すること"""
        result = load_palette_rows("pyxel")
        assert isinstance(result, list)
        assert len(result) == 16

    def test_load_palette_rows_rgb_values(self):
        """load_palette_rows() が正しい RGB 値を返すこと"""
        result = load_palette_rows("pyxel")
        for row in result:
            assert len(row) == 3
            assert all(0 <= v <= 255 for v in row)
