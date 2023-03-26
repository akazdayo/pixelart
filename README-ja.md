# PixelArt-Converter
Language : [English](README.md)  
# 基本機能
## colorpallet
このサイトでは、色を変換しています。  
色を変換するときに使用するカラーパレットを選択します。  
Pyxelは[Pyxel](https://github.com/kitao/pyxel)というライブラリで使用されている色です。  
![Color pallet](./image/pallet.png)

## ratio
0.01ずつ調整できるスライダーで、数字が少なくなるほどドットが大きくなります。
![Select ratio](./image/ratio.png)

## Custom Pallet
ColorPalletを自分で作成できます。  
表の中にパレットに追加したい色をカラーコードで入力します。  
表に入力した色は右側に表示されます。  
表の上にあるカラーピッカーから色を選択してカラーコードをコピーし、入力すると楽です。  
※スポイトには対応していません
![Custom pallet](./image/custom.png)

## Tweet
ツイッターにツイートするボタンです。  
画像の添付には対応していません。  
※画像を添付する際、画像のコピーまたは画像をダウンロードして添付してください  
※トラッカーブロックをオフにしないと表示されない可能性があります。  

# More Options
## Anime Filter
エッジを追加します。  
アニメっぽくなるかもしれないです。
![animefilter_on](./image/anime.png)
![animefilter_off](./image/anime2.jpg)

## No Color Convert
カラーパレットを使用しないようにします。  
![no_convert](./image/no_convert.jpg)

## decrease Color
減色処理をします。  
基本的には、``No Color Convert``と一緒に使用します。
![decrease_color](./image/decrease.jpg)

## threhsold
AnimeFilter(エッジ処理)の値です。  
値が小さいほどエッジが多くなります。  
### threhsold 1
エッジの量を指定します。
### threhsold 2
エッジの長さを指定します。

# Experimental Features
まだ正式な機能ではないので、バグや、エラーが発生する可能性があります。  
## Pixel Edge
エッジをドットで生成します。

# Color Sample
デフォルトのカラーパレットに含まれている色を表示します  
![color_sample](./image/sample.png)
