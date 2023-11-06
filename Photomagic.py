import streamlit as st
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import os
import io
import seaborn as sns
import pandas as pd
import base64
# Streamlitアプリの設定
st.set_page_config(
    page_title="PhotoMagic",
    page_icon=":camera:",
)

# 背景色をカスタマイズ
st.markdown(
    """
    <style>
    body {
        background-color: #e6e6fa; /* 薄い紫色 */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlitアプリのタイトルを設定
st.title("IMAGE PROCESS")

st.caption("写真をアップロードして、フィルタを適用し、クリエイティブな画像にしよう!")

st.subheader("画像処理を瞬時に")
st.caption("ImageProcessは、画像処理を簡単かつインタラクティブに行えるウェブアプリケーションです。\n"
        "このアプリを使用すると、アップロードした画像にさまざまな画像処理フィルタを適用でき、クリエイティブな画像にできます！")

# 画像処理関数の定義
# セピアフィルタを適用する関数
def apply_sepia_filter(image):
    return ImageOps.colorize(image.convert("L"), "#704214", "#C0A080")

# モノクロフィルタを適用する関数
def apply_monochrome_filter(image):
    return ImageOps.grayscale(image)

# ぼかしフィルタを適用する関数
def apply_blur_filter(image):
    return image.filter(ImageFilter.BLUR)

# エッジ検出フィルタを適用する関数
def apply_edge_detection(image):
    return image.filter(ImageFilter.FIND_EDGES)

# 色反転フィルタを適用する関数
def apply_color_inversion(image):
    # 画像をRGBモードに変換
    image_rgb = image.convert("RGB")
    
    # 色反転処理を適用
    inverted_image = ImageOps.invert(image_rgb)
    
    return inverted_image

# 明るさ調整フィルタを適用する関数
def apply_brightness_adjustment(image, brightness_factor):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(brightness_factor)

# モザイク効果を適用する関数
def apply_mosaic_effect(image, block_size):
    return image.resize(
        (image.width // block_size, image.height // block_size),
        resample=Image.NEAREST
    ).resize(
        (image.width, image.height),
        resample=Image.NEAREST
    )
# 画像のピクセル数を取得する関数
def get_image_size(image):
    width, height = image.size
    return width, height

# 画像をクロップする関数
def apply_crop(image, crop_area):
    width, height = image.size
    if crop_area == "上":
        return image.crop((0, 0, width, height // 2))
    elif crop_area == "下":
        return image.crop((0, height // 2, width, height))
    elif crop_area == "左":
        return image.crop((0, 0, width // 2, height))
    elif crop_area == "右":
        return image.crop((width // 2, 0, width, height))

# 画像の保存とダウンロードを行う関数
def save_and_download_image(image, file_name):
    try:
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_data = base64.b64encode(buffered.getvalue()).decode()  # Base64エンコード
        href = f'<a href="data:application/octet-stream;base64,{img_data}" download="{file_name}">ダウンロード</a>'
        st.markdown(href, unsafe_allow_html=True)
    except Exception as e:
        error_message = f"エラーが発生しました: {str(e)}"
        st.error(error_message)

# サイドバーに画像をアップロードするセクションを追加
st.sidebar.header("画像をアップロード")
uploaded_image = st.sidebar.file_uploader("画像を処理します", type=["jpg", "jpeg", "png"])

# オリジナル画像を保存するためのディレクトリを作成
os.makedirs("original_images", exist_ok=True)

# フィルタ適用後の画像を初期化
filtered_image = None

if uploaded_image is not None:
    # アップロードされた画像を表示
    image = Image.open(uploaded_image)
    st.image(image, caption="アップロードされた画像", use_column_width=True)

    # 画像のピクセル数を取得
    width, height = get_image_size(image)

    # ピクセル数を表示
    st.write(f"幅 (Width): {width} ピクセル")
    st.write(f"高さ (Height): {height} ピクセル")

    # オリジナル画像を保存
    original_image_path = os.path.join("original_images", uploaded_image.name)
    with open(original_image_path, "wb") as f:
        # 画像をRGBモードに変換して保存
        image = image.convert("RGB")
        image.save(f, format="JPEG")

    # サイドバーに画像処理オプションを追加
    st.sidebar.header("画像処理オプション")
    selected_filter = st.sidebar.selectbox("フィルタを選択", ["なし", "セピア", "モノクロ", "ぼかし", "エッジ検出", "色反転", "明るさ調整", "モザイク", "クロップ"])

    if selected_filter != "なし":
        st.sidebar.header("フィルタ適用")
        if selected_filter == "セピア":
            filtered_image = apply_sepia_filter(image)
        elif selected_filter == "モノクロ":
            filtered_image = apply_monochrome_filter(image)
        elif selected_filter == "ぼかし":
            filtered_image = apply_blur_filter(image)
        elif selected_filter == "エッジ検出":
            filtered_image = apply_edge_detection(image)
        elif selected_filter == "色反転":
            filtered_image = apply_color_inversion(image)
        elif selected_filter == "明るさ調整":
            brightness_factor = st.sidebar.slider("明るさ調整", 0.1, 3.0, 1.0)
            filtered_image = apply_brightness_adjustment(image, brightness_factor)
        elif selected_filter == "モザイク":
            block_size = st.sidebar.slider("モザイクのブロックサイズ", 10, 100, 20)
            filtered_image = apply_mosaic_effect(image, block_size)
        elif selected_filter == "クロップ":
            crop_area = st.sidebar.selectbox("クロップエリアを選択", ["上", "下", "左", "右"])
            filtered_image = apply_crop(image, crop_area)

    if filtered_image is not None:
        try:
            # フィルタ適用後の画像を表示
            st.image(filtered_image, caption="フィルタ適用後の画像", use_column_width=True)
        except NameError:
            st.error("フィルタ適用後の画像が見つかりません。画像フィルタを選択して処理を行ってください。")

# 新しい比較機能の追加
if filtered_image is not None and st.button("オリジナル画像との比較"):
    original_image = Image.open(original_image_path)
    st.image(original_image, caption="オリジナル画像", use_column_width=True)
    st.image(filtered_image, caption="フィルタ適用後の画像", use_column_width=True)

# フィルタ適用後の画像を保存とダウンロード
if filtered_image is not None and st.button("画像を保存"):
    save_and_download_image(filtered_image, "filtered_image.jpg")

st.set_option('deprecation.showPyplotGlobalUse', False)
