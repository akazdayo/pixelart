# pixelart-converter
Language : [日本語](README-ja.md)  
Fascinating tool to convert images into pixel art!  
[pixelart-converter](https://pixelart.streamlit.app)

# Basic functions
## colorpallet
This site converts colors.  
Select the color palette to use when converting colors.  
Pyxel is the color used in the library called [Pyxel](https://github.com/kitao/pyxel).  
![Color pallet](./image/pallet.png)

## ratio
This is a slider that can be adjusted in increments of 0.01. The lower the number, the larger the dot.
![Select ratio](./image/ratio.png)

## Custom Pallet
You can create your own ColorPallet.  
Enter the colors you want to add to the palette in the table using color codes.  
The colors entered in the table will be displayed on the right side.  
It is easier to select a color from the color picker above the table, copy the color code, and enter it.  
Color picker is not supported.
![Custom pallet](./image/custom.png)

## Tweet
A button to tweet to Twitter.  
It does not support attaching images.  
When attaching an image, please copy the image or download the image and attach it.  
If you do not turn off the tracker blocker, it may not be displayed.  

# Experimental Features
This is not an official feature yet, so there may be bugs or errors.  

## Anime Filter
Add edges.  
![animefilter_on](./image/anime.png)
![animefilter_off](./image/anime2.jpg)

## No Color Convert
Disables the color palette.  
![no_convert](./image/no_convert.jpg)


## decrease Color
Decrease color.  
Basically used with ``No Color Convert``.
![decrease_color](./image/decrease.jpg)

## threhsold
Value of AnimeFilter (edge processing).  
The smaller the value, the more edges are processed.  
### threhsold 1
Specifies the amount of edges.
### threhsold 2
Specifies the length of edges.

# Color Sample
Displays the colors in the default color palette  
![color_sample](./image/sample.png)
