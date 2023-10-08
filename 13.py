import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

# Streamlitアプリのタイトルを設定
st.title("画像処理アプリ")

# 画像処理関数の定義
def apply_sepia_filter(image):
    kernel = np.array([[0.393, 0.769, 0.189],
                      [0.349, 0.686, 0.168],
                      [0.272, 0.534, 0.131]])
    sepia_image = cv2.transform(image, kernel)
    sepia_image = np.clip(sepia_image, 0, 255).astype(np.uint8)
    return sepia_image

def apply_blur_filter(image):
    blur_image = cv2.GaussianBlur(image, (15, 15), 0)
    return blur_image

def apply_monochrome_filter(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)

def apply_edge_detection(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    edge_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return edge_image

def apply_color_inversion(image):
    inverted_image = cv2.bitwise_not(image)
    return inverted_image

def apply_brightness_adjustment(image, brightness_factor):
    adjusted_image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)
    return adjusted_image

def apply_mosaic_effect(image, block_size):
    h, w, _ = image.shape
    temp_image = cv2.resize(image, (w // block_size, h // block_size))
    mosaic_image = cv2.resize(temp_image, (w, h), interpolation=cv2.INTER_NEAREST)
    return mosaic_image

# サイドバーに画像をアップロードするセクションを追加
st.sidebar.header("画像をアップロード")
uploaded_image = st.sidebar.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])

# オリジナル画像を保存するためのディレクトリを作成
os.makedirs("original_images", exist_ok=True)

if uploaded_image is not None:
    # アップロードされた画像を表示
    image = Image.open(uploaded_image)
    st.image(image, caption="アップロードされた画像", use_column_width=True)

    # オリジナル画像を保存
    original_image_path = os.path.join("original_images", uploaded_image.name)
    image.save(original_image_path, format="JPEG")

    # サイドバーに画像処理オプションを追加
    st.sidebar.header("画像処理オプション")
    selected_filter = st.sidebar.selectbox("フィルタを選択", ["なし", "セピア", "モノクロ", "ぼかし", "エッジ検出", "色反転", "明るさ調整", "モザイク"])

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

        # フィルタ適用後の画像を表示
        st.image(filtered_image, caption=f"{selected_filter}フィルタ適用後", use_column_width=True)

    # 新しい比較機能の追加
    if st.button("オリジナル画像との比較"):
        original_image = Image.open(original_image_path)
        st.image(original_image, caption="オリジナル画像", use_column_width=True)
        st.image(filtered_image, caption="フィルタ適用後の画像", use_column_width=True)

    # 新しい画像保存機能の追加
    if st.button("画像を保存"):
        if "filtered_image" in globals():
            filtered_image.save("filtered_image.jpg", format="JPEG")
            st.success("フィルタ適用後の画像を保存しました。")

# 新しい複数の画像処理機能の追加
st.subheader("複数の画像の処理")
uploaded_images = st.file_uploader("複数の画像をアップロードしてください", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_images:
    for uploaded_image in uploaded_images:
        image = Image.open(uploaded_image)
        
        st.image(image, caption="アップロードされた画像", use_column_width=True)