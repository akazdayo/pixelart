import streamlit as st
import cv2
import pandas as pd
import warnings
import gc
import src.ai as ai
import src.convert as convert
import src.filters as filters
import src.draw as draw

warnings.simplefilter('ignore')


def main():
    # instance
    web = draw.Web()
    ai_palette = ai.AI()
    conv = convert.Convert()
    edges = filters.EdgeFilter()
    enhance = filters.ImageEnhancer()

    if web.upload != None:
        img = web.get_image(web.upload)
    else:
        img = web.get_image("sample/irasutoya.png")

    height, width = img.shape[:2]
    if height*width < 2100000:
        pass
    else:
        img = conv.resize_image(img)
        web.message.warning("""
The size of the image has been reduced because the file size is too large.\n
Image size is reduced if the number of pixels exceeds FullHD (2,073,600).
        """)

    cimg = img.copy()
    del web.upload
    web.col1.image(cimg)

    if web.saturation != 1:
        cimg = enhance.saturation(
            cimg, web.saturation)

    if web.brightness != 1:
        cimg = enhance.brightness(
            cimg, web.brightness)

    if web.sharpness != 1:
        cimg = enhance.sharpness(
            cimg, web.sharpness)

    if web.scratch:
        cimg = edges.dog(cimg, True)

    if web.pixel_canny_edge:
        web.now.write("### Pixel Edge in progress")
        cimg = edges.canny(cimg, web.px_th1, web.px_th2)

    if web.px_dog_filter:
        web.now.write("### Pixel Edge in progress")
        cimg = edges.dog(cimg)

    web.now.write("### Now mosaic")

    if web.slider != 1:
        cimg = enhance.mosaic(cimg, web.slider)

    if web.no_convert == False:
        if web.color == "Custom Palette" or web.color == 'AI':
            if web.color == 'AI' and web.color != "Custom Palette":
                web.now.write("### AI Palette in progress")
                ai_color = ai_palette.get_color(
                    cimg, web.color_number, web.ai_iter)

                with st.expander("AI Palette"):
                    web.custom_palette(pd.DataFrame(
                        {"hex": c} for c in ai_color))

            web.now.write("### Color Convert in progress")
            cimg = conv.convert(cimg, "Custom", web.rgblist)

        else:
            web.now.write("### Color Convert in progress")
            cimg = conv.convert(cimg, web.color)

    if web.no_expand == False:
        cimg = cv2.resize(cimg, img.shape[:2][::-1],
                          interpolation=cv2.INTER_NEAREST)

    if web.decreaseColor:
        web.now.write("### Decrease Color in progress")
        cimg = enhance.decrease(cimg)

    if web.smooth_canny_filter:
        web.now.write("### Edge filter in progress")
        cimg = edges.canny(cimg, web.anime_th1, web.anime_th2)

    if web.smooth_dog_filter:
        web.now.write("### Edge filter in progress")
        cimg = edges.dog(cimg)

    web.col2.image(cimg, use_column_width=True)
    st.sidebar.image(cimg, use_column_width=True)
    web.now.write("")
    del conv.color_dict
    gc.collect()
