"""Preprocessing yang DIPAKAI BERSAMA oleh training dan inference,
supaya input model identik di kedua sisi."""
import cv2
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input

import config

__all__ = ["preprocess_input", "prepare_face"]


def prepare_face(gray_face: np.ndarray) -> np.ndarray:
    """Crop wajah grayscale -> array (H, W, 3) siap masuk VGG16."""
    face = cv2.resize(gray_face, config.IMG_SIZE, interpolation=cv2.INTER_AREA)
    face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB).astype("float32")
    return preprocess_input(face)
