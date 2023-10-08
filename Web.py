import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import io
import matplotlib.pyplot as plt
import base64

# Matplotlibのバックエンドを設定
plt.switch_backend('agg')

# アイコンを設定
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
st.title("PhotoMagic")

# 画像処理関数の定義
# セピアフィルタを適用する関数
def apply_sepia_filter(image):
    kernel = np.array([[0.393, 0.769, 0.189],
                      [0.349, 0.686, 0.168],
                      [0.272, 0.534, 0.131]])
    sepia_image = cv2.transform(image, kernel)
    sepia_image = np.clip(sepia_image, 0, 255).astype(np.uint8)
    return sepia_image

# ぼかしフィルタを適用する関数
def apply_blur_filter(image):
    blur_image = cv2.GaussianBlur(image, (15, 15), 0)
    return blur_image

# モノクロフィルタを適用する関数
def apply_monochrome_filter(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)

# エッジ検出フィルタを適用する関数
def apply_edge_detection(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    edge_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return edge_image

# 色反転フィルタを適用する関数
def apply_color_inversion(image):
    inverted_image = cv2.bitwise_not(image)
    return inverted_image

# 明るさ調整フィルタを適用する関数
def apply_brightness_adjustment(image, brightness_factor):
    adjusted_image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)
    return adjusted_image

# モザイク効果を適用する関数
def apply_mosaic_effect(image, block_size):
    h, w, _ = image.shape
    temp_image = cv2.resize(image, (w // block_size, h // block_size))
    mosaic_image = cv2.resize(temp_image, (w, h), interpolation=cv2.INTER_NEAREST)
    return mosaic_image

# 画像をクロップする関数
def apply_crop(image, crop_area):
    if crop_area == "上":
        return image[:image.shape[0] // 2, :]
    elif crop_area == "下":
        return image[image.shape[0] // 2:, :]
    elif crop_area == "左":
        return image[:, :image.shape[1] // 2]
    elif crop_area == "右":
        return image[:, image.shape[1] // 2:]

# 画像の保存とダウンロードを行う関数
def save_and_download_image(image, file_name):
    image_np = np.array(image)  # PILイメージをNumPy配列に変換
    image_pil = Image.fromarray(image_np)  # NumPy配列をPIL Imageに変換
    buffered = io.BytesIO()
    image_pil.save(buffered, format="JPEG")
    img_data = base64.b64encode(buffered.getvalue()).decode()  # Base64エンコード
    href = f'<a href="data:application/octet-stream;base64,{img_data}" download="{file_name}">ダウンロード</a>'
    st.markdown(href, unsafe_allow_html=True)

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

    # オリジナル画像を保存
    original_image_path = os.path.join("original_images", uploaded_image.name)
    image.save(original_image_path, format="JPEG")

    # サイドバーに画像処理オプションを追加
    st.sidebar.header("画像処理オプション")
    selected_filter = st.sidebar.selectbox("フィルタを選択", ["なし", "セピア", "モノクロ", "ぼかし", "エッジ検出", "色反転", "明るさ調整", "モザイク", "クロップ"])

    if selected_filter != "なし":
        st.sidebar.header("フィルタ適用")
        if selected_filter == "セピア":
            filtered_image = apply_sepia_filter(np.array(image))
        elif selected_filter == "モノクロ":
            filtered_image = apply_monochrome_filter(np.array(image))
        elif selected_filter == "ぼかし":
            filtered_image = apply_blur_filter(np.array(image))
        elif selected_filter == "エッジ検出":
            filtered_image = apply_edge_detection(np.array(image))
        elif selected_filter == "色反転":
            filtered_image = apply_color_inversion(np.array(image))
        elif selected_filter == "明るさ調整":
            brightness_factor = st.sidebar.slider("明るさ調整", 0.1, 3.0, 1.0)
            filtered_image = apply_brightness_adjustment(np.array(image), brightness_factor)
        elif selected_filter == "モザイク":
            block_size = st.sidebar.slider("モザイクのブロックサイズ", 10, 100, 20)
            filtered_image = apply_mosaic_effect(np.array(image), block_size)
        elif selected_filter == "クロップ":
            crop_area = st.sidebar.selectbox("クロップエリアを選択", ["上", "下", "左", "右"])
            filtered_image = apply_crop(np.array(image), crop_area)

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

# 新しい複数の画像処理機能の追加
st.subheader("画像の色の統計情報")
uploaded_images = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_images:
    for i, uploaded_image in enumerate(uploaded_images):
        image = Image.open(uploaded_image)

        st.image(image, caption="アップロードされた画像", use_column_width=True)

        # 画像の統計情報を表示
        st.subheader("画像統計情報")
        image_np = np.array(image)
        st.write(f"画像の幅: {image_np.shape[1]} ピクセル")
        st.write(f"画像の高さ: {image_np.shape[0]} ピクセル")
        st.write(f"画像のチャンネル数: {image_np.shape[2]}")

        # 画像の平均色を計算
        mean_color = np.mean(image_np, axis=(0, 1))
        st.write(f"画像の平均色 (RGB): {mean_color}")

        # 画像の最頻色を計算
        pixels = image_np.reshape(-1, 3)
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
        most_common_color = unique_colors[np.argmax(counts)]

        # 最頻色のRGB値を抽出
        r, g, b = most_common_color

        # 棒グラフを作成
        plt.figure(figsize=(6, 6))
        colors = [f"#{r:02x}{g:02x}{b:02x}"]
        plt.bar("Most Common Color", counts[0], color=colors)

        # グラフに値を表示
        plt.text("Most Common Color", counts[0], f"{counts[0]}", ha="center", va="bottom")

        plt.title("最頻色の割合")
        plt.xlabel("色")
        plt.ylabel("ピクセル数")
        st.pyplot(plt)

    # 画像のダウンロードオプションを追加
    if st.button("画像をダウンロード"):
        for i, uploaded_image in enumerate(uploaded_images):
            image = Image.open(uploaded_image)
            save_and_download_image(image, f"downloaded_image_{i+1}.jpg")
