# pixelart-converter
Language : [日本語](README-ja.md)  
Fascinating tool to convert images into pixel art!  
[pixelart-converter](https://pixelart.streamlit.app)

# Basic functions
## color palette
This site converts colors.  
Select the color palette to use when converting colors.  
Pyxel is the color used in the library called [Pyxel](https://github.com/kitao/pyxel).  
![Color palette](./image/palette.png)
### AI Palette
Create a palette dedicated to images entered using KMeans.  
https://pixabay.com/vectors/homes-the-needle-village-mountains-8194751/
![AI](./image/ai.jpg)

## Mosaic ratio
This is a slider that can be adjusted in increments of 0.01. The lower the number, the larger the dot.
![Select ratio](./image/ratio.png)

## Custom palette
You can create your own Colorpalette.  
Enter the colors you want to add to the palette in the table using color codes.  
The colors entered in the table will be displayed on the right side.  
It is easier to select a color from the color picker above the table, copy the color code, and enter it.  
Color picker is not supported.
![Custom palette](./image/custom.png)

## Tweet
A button to tweet to Twitter.  
It does not support attaching images.  
When attaching an image, please copy the image or download the image and attach it.  
If you do not turn off the tracker blocker, it may not be displayed.  

# More Options
## Anime Filter
Add edges.  
![animefilter_on](./image/ai.jpg)
![animefilter_off](./image/noanime.jpg)
### threhsold
Value of AnimeFilter (edge processing).  
The smaller the value, the more edges are processed.  
#### threhsold 1
Specifies the amount of edges.
#### threhsold 2
Specifies the length of edges.

## No Color Convert
Disables the color palette.  
![no_convert](./image/no_convert.jpg)


## decrease Color
Decrease color.  
Basically, it is used at the same time as ``No Color Convert``.
![decrease_color](./image/decrease.jpg)

# Experimental Features
This is not an official feature yet, so there may be bugs or errors.  
## AI
### AI Color
Changes the number of colors used when using the AI Color palette.

# Color Sample
Displays the colors in the default color palette  
![color_sample](./image/sample.png)

# Develop
```
$ rye sync
$ rye run streamlit run main.py
```
## What is pixelart-modules?
The conversion process is written in Rust.
https://github.com/akazdayo/pixelart-modules