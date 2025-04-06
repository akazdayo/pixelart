import streamlit as st
import cv2
import gc
import src.ai as ai
import src.convert as convert
import src.filters as filters
import base64

warning_message = """
The size of the image has been reduced because the file size is too large.\n
Image size is reduced if the number of pixels exceeds FullHD (2,073,600).
"""


def cv_to_base64(img):
    _, encoded = cv2.imencode(".png", img)
    img_str = base64.b64encode(encoded).decode("ascii")

    return img_str


def main(web):
    # instance
    ai_palette = ai.AI()
    conv = convert.Convert()
    edges = filters.EdgeFilter()
    enhance = filters.ImageEnhancer()

    if web.upload is not None:
        img = web.get_image(web.upload)

    else:
        img = web.get_image("sample/cat_and_dog.jpg")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    height, width = img.shape[:2]
    if height * width < 2073600:
        pass
    else:
        img = conv.resize_image(img)
        web.message.warning(warning_message)

    cimg = img.copy()
    del web.upload

    encoded_img = cv_to_base64(cv2.cvtColor(cimg, cv2.COLOR_BGR2RGB))

    web.col1.image(
        f"data:image/png;base64,{encoded_img}", use_container_width=True)

    if web.saturation != 1:
        cimg = enhance.saturation(cimg, web.saturation)

    if web.brightness != 1:
        cimg = enhance.brightness(cimg, web.brightness)

    if web.contrast != 1:
        cimg = enhance.contrast(cimg, web.contrast)

    if web.sharpness != 1:
        cimg = enhance.sharpness(cimg, web.sharpness)

    if web.delete_transparent:
        cimg = conv.delete_transparent_color(cimg)

    if web.scratch:
        cimg = edges.dog(cimg, True)

    if web.median:
        cimg = edges.median(cimg, 15)

    if web.kuwahara:
        cimg = edges.apply_kuwahara(cimg)

    if web.px_dog_filter and not web.scratch:
        web.now.write("### Pixel Edge in progress")
        cimg = edges.dog(cimg)

    if web.morphology and not web.scratch:
        # MorphologyとScratchは同時適用するとキモくなる
        cimg = edges.morphology_erode(cimg)

    web.now.write("### Now mosaic")
    # cimg = enhance.mosaic(cimg, web.slider)
    if web.pixel_dropdown == "Slider":
        cimg = enhance.slider_mosaic(cimg, web.slider)
    else:
        cimg = enhance.grid_mosaic(cimg, web.pixel_grid)

    if not web.no_convert:
        if web.color == "Custom Palette" or web.color == "AI":
            if web.color == "AI" and web.color != "Custom Palette":
                web.now.write("### AI Palette in progress")
                ai_color = ai_palette.get_color(
                    cimg, web.color_number, web.ai_iter)
                with st.expander("AI Palette"):
                    web.custom_palette(ai_color)

            web.now.write("### Color Convert in progress")
            cimg = conv.convert(cimg, "Custom", web.rgblist)

        else:
            web.now.write("### Color Convert in progress")
            cimg = conv.convert(cimg, web.color)

    if not web.no_expand:
        cimg = cv2.resize(
            cimg, img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

    if web.decreaseColor:
        web.now.write("### Decrease Color in progress")
        cimg = enhance.decrease(cimg)

    if web.delete_alpha:
        web.now.write("### Delete Alpha in progress")
        cimg = conv.delete_alpha(cimg)

    encoded_img = cv_to_base64(cimg)

    # Convert BGR to RGB
    if cimg.dtype != "uint8":
        cimg = cv2.convertScaleAbs(cimg)
    cimg_rgb = cv2.cvtColor(cimg, cv2.COLOR_BGR2RGB)
    encoded_img = cv_to_base64(cimg_rgb)

    web.col2.image(
        f"data:image/png;base64,{encoded_img}", use_container_width=True)
    st.sidebar.image(
        f"data:image/png;base64,{encoded_img}", use_container_width=True)
    web.now.write("")
    del conv.color_dict
    gc.collect()
