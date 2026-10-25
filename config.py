"""Konfigurasi terpusat. Semua path, hyperparameter, dan label ada di sini."""
import os
from pathlib import Path

# ---------- Path ----------
BASE_DIR = Path(__file__).resolve().parent
# Lokasi dataset: bisa diatur lewat env var FER_DATA_DIR (dipakai di Colab),
# atau otomatis: data/train, lalu data/fer2013/train.
_env_data = os.environ.get("FER_DATA_DIR")
if _env_data:
    DATA_DIR = Path(_env_data)
elif (BASE_DIR / "data" / "train").exists():
    DATA_DIR = BASE_DIR / "data"
else:
    DATA_DIR = BASE_DIR / "data" / "fer2013"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
MODEL_PATH = BASE_DIR / "models" / "fer_vgg16.keras"

# ---------- Data & model ----------
IMG_SIZE = (96, 96)          # VGG16 butuh minimal 32x32; 96 lebih baik dari 48 asli
INPUT_SHAPE = (*IMG_SIZE, 3)
BATCH_SIZE = 64
VAL_SPLIT = 0.15
SEED = 42

# Urutan harus alfabet (sama dengan nama folder dataset)
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_CLASSES = len(CLASS_NAMES)

# ---------- Training ----------
EPOCHS_HEAD = 15             # fase 1: base VGG16 dibekukan
EPOCHS_FINE = 15             # fase 2: fine-tune block5
LR_HEAD = 1e-3
LR_FINE = 1e-5

# ---------- Tampilan ----------
LABELS_ID = {
    "angry": "Marah", "disgust": "Jijik", "fear": "Takut", "happy": "Senang",
    "neutral": "Netral", "sad": "Sedih", "surprise": "Terkejut",
}
EMOJI = {
    "angry": "😠", "disgust": "🤢", "fear": "😨", "happy": "😄",
    "neutral": "😐", "sad": "😢", "surprise": "😲",
}
# Warna kotak (BGR) per ekspresi
COLORS_BGR = {
    "angry": (60, 60, 230), "disgust": (60, 160, 60), "fear": (180, 80, 160),
    "happy": (60, 200, 255), "neutral": (200, 200, 200), "sad": (230, 130, 50),
    "surprise": (0, 165, 255),
}

# ---------- Evaluasi ----------
REPORTS_DIR = BASE_DIR / "reports"
