"""Data layer: membaca FER2013 (struktur folder train/test per kelas)."""
import numpy as np
from tensorflow import keras

import config
from src.data.preprocessing import preprocess_input


def _make_ds(directory, subset=None, shuffle=True):
    """Return (dataset, daftar_path_file) — path dipakai untuk analisis error."""
    kwargs = dict(
        labels="inferred",
        label_mode="categorical",
        class_names=config.CLASS_NAMES,
        color_mode="rgb",                  # grayscale 48x48 -> 3 channel
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        seed=config.SEED,
    )
    if subset:
        kwargs.update(validation_split=config.VAL_SPLIT, subset=subset)
    raw = keras.utils.image_dataset_from_directory(directory, **kwargs)
    paths = list(raw.file_paths)
    ds = raw.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=-1)
    return ds.prefetch(-1), paths


def load_datasets():
    """Dataset untuk training: (train, validation)."""
    train_ds, _ = _make_ds(config.TRAIN_DIR, subset="training")
    val_ds, _ = _make_ds(config.TRAIN_DIR, subset="validation")
    return train_ds, val_ds


def load_test_dataset():
    """Dataset test (tanpa shuffle, urutan = urutan file_paths)."""
    return _make_ds(config.TEST_DIR, shuffle=False)


def compute_class_weights() -> dict:
    """FER2013 tidak seimbang (kelas 'disgust' sangat sedikit)."""
    counts = np.array([
        len(list((config.TRAIN_DIR / c).glob("*"))) for c in config.CLASS_NAMES
    ], dtype="float64")
    weights = counts.sum() / (config.NUM_CLASSES * counts)
    return {i: float(w) for i, w in enumerate(weights)}
