import streamlit as st
import streamlit.components.v1 as components
import os
import numpy as np
import pandas as pd
from PIL import Image


class Web:
    def __init__(self) -> None:
        self.draw_text()

    def file_dir(self):
        filedir = os.listdir("./color")
        for i in range(len(filedir)):
            filedir[i] = filedir[i].replace(".csv", "")
        filedir = tuple(filedir)
        return filedir

    def draw_text(self):
        st.set_page_config(
            page_title="Pixelart-Converter",
            page_icon="🖼️",
            layout="centered",
            initial_sidebar_state="expanded",
        )
        st.title("PixelArt-Converter")
        self.message = st.empty()
        self.upload = st.file_uploader(
            "Upload Image", type=["jpg", "jpeg", "png", "webp", "jfif"]
        )
        self.col1, self.col2 = st.columns(2)
        st.write("""Link copy is not available.Please copy or download the image.""")
        self.color = st.selectbox(  # TODO: このあたりわかりにくいから将来的に修正したい
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

        # pixelの大きさを調整する
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

        self.share()

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

    def share(self):
        with st.sidebar:
            components.html(
                """
    <a href="https://twitter.com/share?ref_src=twsrc%5Etfw" class="twitter-share-button" data-show-count="false" data-text="PixelArt-Converter\nFascinating tool to convert images into pixel art!\n By @akazdayo" data-url="https://pixelart.streamlit.app" data-hashtags="pixelart,streamlit">Tweet</a><script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                """,
                height=30,
            )

    def hex_to_rgb(self, hex_code):
        hex_code = hex_code.replace("#", "")
        r = int(hex_code[0:2], 16)
        g = int(hex_code[2:4], 16)
        b = int(hex_code[4:6], 16)
        return [r, g, b]

    def hex_to_rgblist(self, hex_list):
        rgb_values = []
        for hex_code in hex_list:
            if hex_code is not None:
                rgb_values.append(self.hex_to_rgb(hex_code[1:]))
        return rgb_values

    def custom_palette(  # TODO: pandasからndarrayに変更したい
        self,
        df=pd.DataFrame(
            [
                {"hex": "#FF0000"},
                {"hex": "#00FF00"},
                {"hex": "#0000FF"},
                {"hex": "#FFFFFF"},
                {"hex": "#000000"},
            ]
        ),
    ):
        st.title("Add Palette")
        col1, col2 = st.columns(2)
        self.edited_df = col1.data_editor(df, num_rows="dynamic")
        self.rgblist = list()
        for i in range(len(self.edited_df.loc[self.edited_df["hex"].keys()])):
            self.rgblist.append([])
            self.rgblist[i].append((self.edited_df.loc[self.edited_df.index[i]]["hex"]))
        self.show_custom(col2)

    def show_custom(self, col):
        color_palette = [item[0] for item in self.rgblist]
        color_palette = self.hex_to_rgblist(color_palette)
        rgb = []
        for i in color_palette:
            color = np.zeros((50, 50, 3), dtype=np.uint8)
            color[:, :] = [i[0], i[1], i[2]]
            col.image(color)
            rgb.append(i)
        self.rgblist = rgb

    def experimental(self):
        st.write("""
            The following features are experimental and subject to errors and bugs.
            """)
        st.title("AI")
        self.color_number = st.slider("AI Color", 1, 20, 8, 1, help="Number of colors")
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
        st.write("Simultaneous application of the Canny and DoG filters is deprecated.")

        st.subheader("DoG Filter")
        self.px_dog_filter = st.checkbox("Pixel DoG Filter", True)

        st.title("Other Filters")
        self.morphology = st.checkbox("Morphology Filter", False)
        self.kuwahara = st.checkbox("Kuwahara Filter", False)
        self.median = st.checkbox("Median Filter", False)
        self.delete_transparent = st.checkbox("Delete transparent color", False)

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
        img = Image.open(upload)
        img_array = np.array(img)
        return img_array
