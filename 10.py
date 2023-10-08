import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import io
import matplotlib.pyplot as plt

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

# ファイル形式の制限
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])

# アップロードされたファイルが画像かどうかを確認する関数
def is_image_file(file):
    return '.' in file.name and file.name.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# サイズ制限 (例: 10MBまで)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# オリジナル画像を保存するためのディレクトリを作成
os.makedirs("original_images", exist_ok=True)

# サイドバーに画像をアップロードするセクションを追加
st.sidebar.header("画像をアップロード")
uploaded_image = st.sidebar.file_uploader("画像を処理します", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # ファイル形式の検証
    if is_image_file(uploaded_image):
        # ファイルサイズの検証
        if len(uploaded_image.getvalue()) <= MAX_FILE_SIZE:
            # アップロードされた画像を表示
            image = Image.open(uploaded_image)
            st.image(image, caption="アップロードされた画像", use_column_width=True)

            # オリジナル画像を保存
            original_image_path = os.path.join("original_images", uploaded_image.name)
            image.save(original_image_path, format="JPEG")

            # サイドバーに画像処理オプションを追加
            st.sidebar.header("画像処理オプション")
            selected_filter = st.sidebar.selectbox("フィルタを選択", ["なし", "セピア", "モノクロ", "ぼかし", "エッジ検出", "色反転", "明るさ調整", "モザイク", "クロップ"])

            # 以下、以前のコードの続き（フィルタの適用など）
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
                    filtered_image = Image.fromarray(filtered_image)
                    filtered_image.save("filtered_image.jpg", format="JPEG")
                    st.success("フィルタ適用後の画像を保存しました")

        else:
            st.sidebar.warning("ファイルサイズが大きすぎます。10MB未満のファイルをアップロードしてください。")
    else:
        st.sidebar.warning("許可されていないファイル形式です。jpg、jpeg、pngファイルのみをアップロードしてください。")

# 新しい複数の画像処理機能の追加
st.subheader("画像の色の統計情報")
uploaded_images = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_images:
    for uploaded_image in uploaded_images:
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
        # 画像をバイトIOに変換してダウンロード
        image_byte_array = io.BytesIO()
        image.save(image_byte_array, format='JPEG')
        st.download_button(
            label="ダウンロード",
            data=image_byte_array.getvalue(),
            file_name="downloaded_image.jpg",
            key="download_button",
        )
