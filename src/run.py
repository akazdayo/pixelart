from typing import cast
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from numpy.typing import NDArray
from pixelart_backend.pipeline import ProcessingOptions, cv_to_base64, process_image

warning_message = """
The size of the image has been reduced because the file size is too large.\n
Image size is reduced if the number of pixels exceeds FullHD (2,073,600).
"""

MAX_PIXELS = 2_073_600


def main(web):
    if web.upload is not None:
        img = web.get_image(web.upload)
    else:
        img = web.get_image("sample/cat_and_dog.jpg")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    height, width = img.shape[:2]
    if height * width > MAX_PIXELS:
        web.message.warning(warning_message)

    del web.upload

    if img.ndim == 3 and img.shape[2] == 4:
        orig_for_encode = cast(
            NDArray[np.uint8],
            cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA),
        )
    else:
        orig_for_encode = cast(
            NDArray[np.uint8],
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
        )
    encoded_orig = cv_to_base64(orig_for_encode)
    web.col1.image(f"data:image/png;base64,{encoded_orig}", width="stretch")

    palette = web.color
    if palette == "Custom Palette":
        palette = "Custom"

    mosaic_mode = "slider" if web.pixel_dropdown == "Slider" else "grid"
    grid_line_color = (
        tuple(web.hex_to_rgb(web.grid_color)) if web.enable_grid else (0, 0, 0)
    )

    opts = ProcessingOptions(
        palette=palette,
        custom_palette=web.rgblist if palette == "Custom" else [],
        no_convert=web.no_convert,
        ai_colors=web.color_number,
        ai_iterations=web.ai_iter,
        mosaic_mode=mosaic_mode,
        slider_ratio=web.slider,
        grid_size=web.pixel_grid,
        no_expand=web.no_expand,
        dog_filter=web.px_dog_filter,
        scratch=web.scratch,
        morphology=web.morphology,
        kuwahara=web.kuwahara,
        median=web.median,
        delete_transparent=web.delete_transparent,
        saturation=web.saturation,
        brightness=web.brightness,
        contrast=web.contrast,
        sharpness=web.sharpness,
        dithering=web.dithering,
        dithering_method=web.dithering_method,
        dither_matrix_size=web.dither_matrix_size,
        dither_intensity=web.dither_intensity,
        decrease_color=web.decreaseColor,
        delete_alpha=web.delete_alpha,
        enable_grid=web.enable_grid,
        grid_line_color=grid_line_color,
        grid_line_thickness=web.grid_line_thickness,
        grid_opacity=web.grid_opacity,
    )

    web.now.write("### Processing...")
    result = process_image(img, opts)

    if result.was_resized:
        web.message.warning(warning_message)

    if result.ai_hex_colors is not None:
        with st.expander("AI Palette"):
            web.custom_palette(pd.DataFrame({"hex": c} for c in result.ai_hex_colors))

    cimg = result.image
    if cimg.dtype != "uint8":
        cimg = cast(NDArray[np.uint8], cv2.convertScaleAbs(cimg))
    if cimg.ndim == 3 and cimg.shape[2] == 4:
        cimg_for_encode = cast(
            NDArray[np.uint8],
            cv2.cvtColor(cimg, cv2.COLOR_RGBA2BGRA),
        )
    else:
        cimg_for_encode = cast(
            NDArray[np.uint8],
            cv2.cvtColor(cimg, cv2.COLOR_RGB2BGR),
        )
    encoded_img = cv_to_base64(cimg_for_encode)

    web.col2.image(f"data:image/png;base64,{encoded_img}", width="stretch")
    st.sidebar.image(f"data:image/png;base64,{encoded_img}", width="stretch")
    web.now.write("")
