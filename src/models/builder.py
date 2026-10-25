"""Model layer: arsitektur VGG16 transfer learning."""
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16

import config


def build_model() -> keras.Model:
    base = VGG16(include_top=False, weights="imagenet",
                 input_shape=config.INPUT_SHAPE, name="vgg16")
    base.trainable = False                       # fase 1: dibekukan

    inputs = keras.Input(shape=config.INPUT_SHAPE)
    # Augmentasi hanya aktif saat training
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.08)(x)
    x = layers.RandomZoom(0.1)(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(config.NUM_CLASSES, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="fer_vgg16")


def compile_model(model: keras.Model, lr: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def unfreeze_last_block(model: keras.Model, block_prefix: str = "block5") -> None:
    """Fase 2: buka hanya block5 VGG16 untuk fine-tuning."""
    base = model.get_layer("vgg16")
    base.trainable = True
    for layer in base.layers:
        layer.trainable = layer.name.startswith(block_prefix)
