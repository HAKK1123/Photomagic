import tensorflow as tf
import numpy as np
from tensorflow import keras
from PIL import Image

# スタイル画像とコンテンツ画像のパスを指定
style_image_path = "style.jpg"  # スタイル画像のパス
content_image_path = "content.jpg"  # コンテンツ画像のパス

# 出力画像のパス
output_image_path = "output.jpg"

# モデルをロード
model = keras.applications.VGG19(include_top=False, weights="imagenet")

# レイヤー名を使用してスタイルフィーチャーマップとコンテンツフィーチャーマップを抽出
style_layers = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
content_layers = ["block5_conv2"]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

def get_model():
    vgg = keras.applications.VGG19(include_top=False, weights="imagenet")
    vgg.trainable = False

    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]

    model_outputs = style_outputs + content_outputs

    return keras.models.Model(vgg.input, model_outputs)

# スタイルとコンテンツのターゲット
style_targets = get_style_targets(style_image_path)
content_targets = get_content_targets(content_image_path)

# スタイル損失関数
def style_loss(style_targets, style_outputs):
    loss = 0
    for i in range(len(style_targets)):
        loss += tf.reduce_mean(tf.square(style_targets[i] - style_outputs[i]))
    return loss

# コンテンツ損失関数
def content_loss(content_targets, content_outputs):
    loss = 0
    for i in range(len(content_targets)):
        loss += tf.reduce_mean(tf.square(content_targets[i] - content_outputs[i]))
    return loss

# トータル変動損失
def total_variation_loss(image):
    x_deltas, y_deltas = tf.image.image_gradients(image)
    return tf.reduce_mean(x_deltas**2 + y_deltas**2)

# 最適化のパラメータ
total_variation_weight = 30
style_weight = 1e-2
content_weight = 1e4

# 最適化ループ
image = tf.Variable(content_image)

optimizer = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        style_outputs, content_outputs = extract_features(image)
        style_loss_value = style_loss(style_targets, style_outputs)
        content_loss_value = content_loss(content_targets, content_outputs)
        tv_loss_value = total_variation_loss(image)
        total_loss = style_weight * style_loss_value + content_weight * content_loss_value + total_variation_weight * tv_loss_value

    grads = tape.gradient(total_loss, image)
    optimizer.apply_gradients([(grads, image)])
    image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0))

num_iterations = 1000
for i in range(num_iterations):
    train_step(image)

# 出力画像を保存
output_img = np.array(image.read_value(), dtype=np.uint8)
output_img = Image.fromarray(output_img)
output_img.save(output_image_path)

print(f"出力画像を {output_image_path} に保存しました。")
