"""Presentation layer: UI Streamlit. Jalankan:  streamlit run app.py"""
import json

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

import config
from src.detection.face_detector import FaceDetector
from src.evaluation.evaluate import NAMES_ID
from src.inference.predictor import ExpressionPredictor

st.set_page_config(page_title="Klasifikasi Ekspresi Wajah", page_icon="😊", layout="wide")


@st.cache_resource(show_spinner="Memuat model...")
def get_predictor() -> ExpressionPredictor:
    return ExpressionPredictor(config.MODEL_PATH, FaceDetector())


def read_image(file) -> np.ndarray:
    img = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ------------------------------------------------------------------ tab 1
def prediction_tab(min_neighbors: int):
    if not config.MODEL_PATH.exists():
        st.error("Model belum ditemukan. Latih dulu dengan "
                 "`python -m src.training.train`.")
        return

    file = st.file_uploader("Upload foto wajah", type=["jpg", "jpeg", "png"])
    if file is None:
        st.info("Upload foto, wajah akan dideteksi otomatis tanpa perlu crop.")
        return

    image = read_image(file)
    annotated, results = get_predictor().predict_image(image, min_neighbors)

    left, right = st.columns([3, 2])
    with left:
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                 caption="Hasil deteksi", use_container_width=True)
    with right:
        if not results:
            st.warning("Tidak ada wajah terdeteksi. Coba foto yang lebih jelas "
                       "atau turunkan nilai Min neighbors.")
            return
        st.subheader(f"{len(results)} wajah terdeteksi")
        for i, r in enumerate(results, 1):
            with st.container(border=True):
                c1, c2 = st.columns([1, 2])
                c1.image(r.face_rgb, use_container_width=True)
                c2.markdown(f"### {config.EMOJI[r.label]} {config.LABELS_ID[r.label]}")
                c2.metric(f"Wajah {i} — keyakinan", f"{r.confidence:.1%}")
                probs = pd.Series(
                    {config.LABELS_ID[k]: v for k, v in r.probabilities.items()})
                st.bar_chart(probs, height=180)


# ------------------------------------------------------------------ tab 2
def evaluation_tab():
    summary_path = config.REPORTS_DIR / "summary.json"
    if not summary_path.exists():
        st.info("Belum ada laporan evaluasi. Jalankan "
                "`python -m src.evaluation.evaluate` (atau selesaikan training).")
        return

    s = json.loads(summary_path.read_text())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{s['accuracy']:.1%}")
    c2.metric("Macro F1", f"{s['macro_f1']:.3f}")
    c3.metric("Weighted F1", f"{s['weighted_f1']:.3f}")
    c4.metric("Jumlah data test", f"{s['n_test']:,}")

    st.subheader("Metrik per kelas")
    table_path = config.REPORTS_DIR / "per_class_metrics.csv"
    if table_path.exists():
        df = pd.read_csv(table_path, index_col=0)
        left, right = st.columns([3, 2])
        left.dataframe(df.round(3), use_container_width=True)
        right.bar_chart(df.loc[NAMES_ID, "f1-score"], height=300)

    st.subheader("Confusion matrix")
    cm_img = config.REPORTS_DIR / "confusion_matrix.png"
    if cm_img.exists():
        st.image(str(cm_img), use_container_width=True)

    st.subheader("Ekspresi yang paling sering tertukar")
    pairs_path = config.REPORTS_DIR / "confused_pairs.csv"
    if pairs_path.exists():
        st.dataframe(pd.read_csv(pairs_path).head(10), use_container_width=True,
                     hide_index=True)

    st.subheader("Kurva training")
    hist_img = config.REPORTS_DIR / "training_history.png"
    if hist_img.exists():
        st.image(str(hist_img), use_container_width=True)

    st.subheader("Tantangan FER2013: label noise & kualitas rendah")
    err_img = config.REPORTS_DIR / "confident_errors.png"
    if err_img.exists():
        st.image(str(err_img), use_container_width=True)
    st.caption("Banyak error berkeyakinan tinggi sebenarnya berasal dari label "
               "yang ambigu atau keliru pada dataset, bukan murni kesalahan model. "
               "Gambar FER2013 hanya 48×48 piksel grayscale.")


# ------------------------------------------------------------------ main
def main():
    st.title("😊 Klasifikasi Ekspresi Wajah")
    st.caption("7 ekspresi dasar • VGG16 Transfer Learning • FER2013 • Haar Cascade")

    with st.sidebar:
        st.header("Pengaturan deteksi")
        min_neighbors = st.slider(
            "Min neighbors", 3, 10, 5,
            help="Lebih tinggi = lebih ketat (lebih sedikit salah deteksi).")
        st.markdown("**Kelas:** " + ", ".join(
            f"{config.EMOJI[c]} {config.LABELS_ID[c]}" for c in config.CLASS_NAMES))

    tab_pred, tab_eval = st.tabs(["🔍 Prediksi", "📊 Evaluasi Model"])
    with tab_pred:
        prediction_tab(min_neighbors)
    with tab_eval:
        evaluation_tab()


main()
