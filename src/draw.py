import streamlit as st
import streamlit.components.v1 as components
import os
import numpy as np
from PIL import Image
import uuid


class Web:
    def __init__(self) -> None:
        self.draw_text()

    def file_dir(self):
        """Get file names from the color palette directory

        Returns:
            tuple: A tuple of file names without the .csv extension
        """
        return tuple(os.path.splitext(f)[0] for f in os.listdir("./color"))

    def setup_page(self):
        """ページの基本設定を行う"""
        st.set_page_config(
            page_title="Pixelart-Converter",
            page_icon="🖼️",
            layout="centered",
            initial_sidebar_state="expanded",
        )
        st.title("PixelArt-Converter")
        self.message = st.empty()

    def setup_image_upload(self):
        """画像アップロード部分のUIを設定"""
        self.upload = st.file_uploader(
            "Upload Image", type=["jpg", "jpeg", "png", "webp", "jfif"]
        )
        self.col1, self.col2 = st.columns(2)
        st.write(
            "Link copy of image is not available. Please copy or download the image."
        )

    def setup_color_settings(self):
        """色設定関連のUIを設定"""
        self.color = st.selectbox(
            "Select color Palette",
            (
                "AI",
                "cold",
                "gold",
                "pale",
                "pastel",
                "pyxel",
                "rainbow",
                "warm",
                "Custom Palette",
            ),
        )

    def setup_pixel_settings(self):
        """ピクセル設定関連のUIを設定"""
        self.pixel_dropdown = st.selectbox(
            "Select Pixel Size", ("Pixel Grid", "Slider")
        )

        self.pixel_grid = st.number_input(
            "Select Pixel Grid",
            1,
            512,
            256,
            disabled=self.pixel_dropdown != "Pixel Grid",
        )
        self.slider = st.slider(
            "Select Mosaic Ratio",
            0.01,
            0.5,
            0.3,
            0.01,
            disabled=self.pixel_dropdown != "Slider",
        )

    def setup_layout(self):
        """レイアウト関連の設定"""
        self.col1.header("Original img")
        self.col2.header("Convert img")
        self.now = st.empty()

        with st.expander("More Options", True):
            self.more_options()
        with st.expander("Custom Palette"):
            self.custom_palette()
        with st.expander("Experimental Features"):
            self.experimental()

        st.write("Source Code : https://github.com/akazdayo/pixelart")

    def draw_text(self):
        """UIの初期化とレイアウトの設定"""
        self.setup_page()
        self.setup_image_upload()
        self.setup_color_settings()
        self.setup_pixel_settings()
        self.share()
        self.setup_layout()

    def share(self):
        with st.sidebar:
            components.html(
                """
    <a href="https://twitter.com/share?ref_src=twsrc%5Etfw" class="twitter-share-button" data-show-count="false" data-text="PixelArt-Converter\nFascinating tool to convert images into pixel art!\n By @akazdayo" data-url="https://pixelart.streamlit.app" data-hashtags="pixelart,streamlit">Tweet</a><script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                """,
                height=30,
            )

    def hex_to_rgb(self, hex_codes):
        """Convert hexadecimal color codes to RGB format

        Args:
            hex_codes (str or list): Single hex code or list of hex codes

        Returns:
            list: Single RGB value or list of RGB values in [R, G, B] format
        """
        if isinstance(hex_codes, str):
            hex_codes = [hex_codes]

        rgb_values = []
        for hex_code in hex_codes:
            if hex_code and isinstance(hex_code, str):
                hex_code = hex_code.replace("#", "")
                try:
                    rgb_values.append(
                        [
                            int(hex_code[0:2], 16),
                            int(hex_code[2:4], 16),
                            int(hex_code[4:6], 16),
                        ]
                    )
                except (ValueError, IndexError):
                    continue

        return rgb_values[0] if len(rgb_values) == 1 else rgb_values

    def custom_palette(
        self, custom_palette=None
    ):
        if custom_palette:
            palette = custom_palette
        else:
            palette = [
                '#431d48', '#f1c391', '#faf2d5', '#d68679', '#648db6', '#0d020f', '#ad5767', '#73376c']

        """カスタムパレットを作成するUIを表示し、RGB値のリストを生成する"""

        st.title("Add Palette")

        if st.button("Add Palette", key=str(uuid.uuid4())):
            # 新しいカラーピッカーを追加
            new_color = f"#{uuid.uuid4().hex[:6]}"
            palette.append(new_color)

        # カスタムカラーピッカーUI
        color_inputs = []
        for i, default_color in enumerate(palette):
            color = st.color_picker(
                f"Color {i + 1}", default_color)
            color_inputs.append(color)

        # RGB値のリストを生成
        self.rgblist = self.hex_to_rgb(color_inputs)

    def experimental(self):
        st.write("""
            The following features are experimental and subject to errors and bugs.
            """)
        st.title("AI")
        self.color_number = st.slider(
            "AI Color", 1, 20, 8, 1, help="Number of colors")
        self.ai_iter = st.slider(
            "AI Number of attempts",
            1,
            3000,
            150,
            1,
            help="Maximum number of iterations of the k-means algorithm for a single run.",
        )
        self.delete_alpha = st.checkbox(
            "Delete Alpha Channel",
            False,
            help="Remove the image transparency except perfect transparency",
        )

    def more_options(self):
        st.title("Anime Filter")
        st.write(
            "Simultaneous application of the Canny and DoG filters is deprecated.")

        st.subheader("DoG Filter")
        self.px_dog_filter = st.checkbox("Pixel DoG Filter", True)

        st.title("Other Filters")
        self.morphology = st.checkbox("Morphology Filter", False)
        self.kuwahara = st.checkbox("Kuwahara Filter", False)
        self.median = st.checkbox("Median Filter", False)
        self.delete_transparent = st.checkbox(
            "Delete transparent color", False)

        st.title("Convert Setting")
        self.no_expand = st.checkbox("No Expand Image")
        self.scratch = st.checkbox("Scratch Filter")
        self.no_convert = st.checkbox("No Color Convert")
        self.decreaseColor = st.checkbox("decrease Color")
        self.saturation = st.slider("Select Saturation", 0.0, 5.0, 1.1, 0.1)
        self.brightness = st.slider("Select Brightness", 0.0, 2.0, 1.0, 0.1)
        self.contrast = st.slider("Select Contrast", 0.0, 2.0, 1.0, 0.1)
        self.sharpness = st.slider("Select Sharpness", 0.0, 2.0, 1.0, 0.1)

    def get_image(self, upload):
        """Open and convert an uploaded image to a numpy array

        Args:
            upload: File object containing the uploaded image

        Returns:
            numpy.ndarray: Image data as a numpy array
        """
        img = Image.open(upload)
        img_array = np.array(img)
        return img_array
