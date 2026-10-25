"""Presentation layer: UI Streamlit. Jalankan:  streamlit run app.py"""
import json
import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

import config
from src.detection.face_detector import FaceDetector
from src.evaluation.evaluate import NAMES_ID
from src.inference.predictor import ExpressionPredictor

st.set_page_config(page_title="Klasifikasi Ekspresi Wajah", page_icon="😊", layout="wide",
                   initial_sidebar_state="expanded")


@st.cache_resource(show_spinner="Memuat model...")
def get_predictor() -> ExpressionPredictor:
    return ExpressionPredictor(config.MODEL_PATH, FaceDetector())


def read_image(file) -> np.ndarray:
    img = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _hex(label: str) -> str:
    if label == "neutral":
        return "#8A90A0"  # abu-abu bawaan terlalu pucat di tema terang
    b, g, r = config.COLORS_BGR[label]
    return f"#{r:02x}{g:02x}{b:02x}"


PRESETS = {"Longgar": 3, "Seimbang": 5, "Ketat": 8}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Plus Jakarta Sans', system-ui, sans-serif; }
.stApp { background: #E9EAEE; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 1100px; }

/* Sidebar */
[data-testid="stSidebar"] { background: #E9EAEE; border-right: none; }
[data-testid="stSidebar"] > div:first-child { padding: .6rem .4rem; }
.brand { display: flex; align-items: center; gap: .7rem; background: #F4F5F8; border-radius: 16px;
         padding: .8rem .9rem; box-shadow: 0 1px 2px rgba(20,24,40,.06); }
.brand .logo { width: 36px; height: 36px; border-radius: 10px; background: #5B5BF0;
               display: grid; place-items: center; font-size: 1.2rem; }
.brand b { display: block; font-size: .98rem; font-weight: 800; letter-spacing: -.01em; }
.brand span { font-size: .75rem; color: #6B7184; }
.side-label { font-size: .78rem; color: #7A8094; font-weight: 600; margin: 1.1rem .3rem .4rem; }

.st-key-nav [role="radiogroup"] { gap: .25rem; }
.st-key-nav label { width: 100%; padding: .6rem .8rem; border-radius: 12px; cursor: pointer;
                    transition: background .15s; }
.st-key-nav label > div:first-child { display: none; }
.st-key-nav label:hover { background: rgba(255,255,255,.6); }
.st-key-nav label:has(input:checked) { background: #fff; box-shadow: 0 1px 3px rgba(20,24,40,.10); }
.st-key-nav label p { font-weight: 600; color: #3A3F52; }
.st-key-nav label:has(input:checked) p { color: #5B5BF0; }

.side-card { background: #F4F5F8; border-radius: 16px; padding: .9rem 1rem;
             box-shadow: 0 1px 2px rgba(20,24,40,.06); font-size: .85rem; color: #4A5064; line-height: 1.6; }
.legend { display: flex; flex-wrap: wrap; gap: .4rem .5rem; }
.legend span { display: inline-flex; align-items: center; gap: .35rem; font-size: .8rem; color: #3A3F52; }
.dot { width: 9px; height: 9px; border-radius: 50%; }

/* Main */
.page-title { font-size: 1.7rem; font-weight: 800; letter-spacing: -.02em; margin: 0; color: #1B1D26; }
.page-sub { color: #6B7184; margin: .3rem 0 1.4rem; font-size: .98rem; }

[data-testid="stFileUploader"] section { border: 2px dashed #C3C7D3; border-radius: 20px; background: #F4F5F8;
    padding: 2.4rem 1.4rem; box-shadow: 0 6px 24px rgba(20,24,40,.06); transition: border-color .2s; }
[data-testid="stFileUploader"] section:hover { border-color: #5B5BF0; }
[data-testid="stFileUploader"] button { border-radius: 10px; font-weight: 600; }

.pill { display: inline-block; background: #fff; border-radius: 10px; padding: .3rem .8rem; font-size: .8rem;
        font-weight: 600; color: #3A3F52; box-shadow: 0 1px 3px rgba(20,24,40,.10); margin-bottom: .6rem; }
[data-testid="stImage"] img { border-radius: 16px; }
[data-testid="stCameraInput"] > div { border-radius: 20px; overflow: hidden; }
[data-testid="stCameraInput"] button { border-radius: 10px; font-weight: 600; }

[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 18px !important; border: none !important;
    background: #F4F5F8; box-shadow: 0 1px 3px rgba(20,24,40,.08); }
[data-testid="stMetric"] { background: #F4F5F8; border-radius: 16px; padding: 1rem 1.2rem;
    box-shadow: 0 1px 3px rgba(20,24,40,.08); }
[data-testid="stMetricLabel"] p { color: #6B7184; font-weight: 600; }
[data-testid="stMetricValue"] { font-weight: 800; letter-spacing: -.02em; }
h2, h3 { letter-spacing: -.02em; font-weight: 700; }

.verdict { display: flex; align-items: center; gap: .8rem; }
.emo { width: 52px; height: 52px; border-radius: 14px; display: grid; place-items: center; font-size: 1.9rem; }
.verdict .name { font-size: 1.35rem; font-weight: 800; letter-spacing: -.02em; line-height: 1.1; color: #1B1D26; }
.verdict .conf { color: #6B7184; font-size: .85rem; }
.bars { display: flex; flex-direction: column; gap: .45rem; margin-top: .9rem; }
.bar { display: grid; grid-template-columns: 4.6rem 1fr 2.8rem; align-items: center; gap: .6rem;
       font-size: .82rem; color: #7A8094; }
.bar.top { color: #1B1D26; font-weight: 700; }
.track { height: 7px; background: #DDE0E8; border-radius: 99px; overflow: hidden; }
.fill { height: 100%; border-radius: 99px; }
.bar .pct { text-align: right; font-variant-numeric: tabular-nums; }
.count { color: #6B7184; font-weight: 600; margin: 1.4rem 0 .6rem; }
.count b { color: #1B1D26; font-size: 1.1rem; }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def render_bars(probs: dict, top: str) -> str:
    rows = []
    for k, v in sorted(probs.items(), key=lambda kv: -kv[1]):
        color = _hex(k) if k == top else "#AEB3C2"
        rows.append(
            f'<div class="bar{" top" if k == top else ""}"><span>{config.LABELS_ID[k]}</span>'
            f'<div class="track"><div class="fill" style="width:{v*100:.1f}%;background:{color}"></div></div>'
            f'<span class="pct">{v:.0%}</span></div>')
    return f'<div class="bars">{"".join(rows)}</div>'


def page_header(title: str, sub: str):
    st.markdown(f'<h1 class="page-title">{title}</h1><p class="page-sub">{sub}</p>',
                unsafe_allow_html=True)


# ------------------------------------------------------------------ halaman prediksi
def prediction_tab(min_neighbors: int):
    page_header("Prediksi ekspresi",
                "Unggah foto, wajah dideteksi otomatis lalu diklasifikasi ke 7 ekspresi.")
    if not config.MODEL_PATH.exists():
        st.error("Model belum ditemukan. Latih dulu dengan "
                 "`python -m src.training.train`.")
        return

    tab_up, tab_cam = st.tabs(["Unggah foto", "Ambil dari kamera"])
    with tab_up:
        uploaded = st.file_uploader("Unggah foto wajah", type=["jpg", "jpeg", "png"],
                                    label_visibility="collapsed")
    with tab_cam:
        snapshot = st.camera_input("Ambil foto wajah", label_visibility="collapsed")
    file = snapshot or uploaded
    if file is None:
        return

    image = read_image(file)
    with st.spinner("Menganalisis ekspresi..."):
        annotated, results = get_predictor().predict_image(image, min_neighbors)

    st.write("")
    a, b = st.columns(2, gap="medium")
    with a:
        st.markdown('<span class="pill">Foto asli</span>', unsafe_allow_html=True)
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), width="stretch")
    with b:
        st.markdown('<span class="pill">Hasil deteksi</span>', unsafe_allow_html=True)
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width="stretch")

    if not results:
        st.warning("Tidak ada wajah terdeteksi. Coba foto yang lebih jelas "
                   "atau ubah pengaturan deteksi ke Longgar.")
        return

    st.markdown(f'<div class="count"><b>{len(results)}</b> wajah terdeteksi</div>',
                unsafe_allow_html=True)
    cols = st.columns(min(len(results), 2), gap="medium")
    for i, r in enumerate(results, 1):
        with cols[(i - 1) % len(cols)].container(border=True):
            c1, c2 = st.columns([1, 2], gap="medium")
            c1.image(r.face_rgb, width="stretch")
            c2.markdown(
                f'<div class="verdict"><div class="emo" style="background:{_hex(r.label)}33">'
                f'{config.EMOJI[r.label]}</div><div>'
                f'<div class="name">{config.LABELS_ID[r.label]}</div>'
                f'<div class="conf">Wajah {i} &bull; yakin {r.confidence:.1%}</div></div></div>',
                unsafe_allow_html=True)
            st.markdown(render_bars(r.probabilities, r.label), unsafe_allow_html=True)


# ------------------------------------------------------------------ halaman kamera langsung
def live_tab(min_neighbors: int):
    page_header("Kamera langsung",
                "Ekspresi dikenali terus-menerus dari kamera laptop.")
    if not config.MODEL_PATH.exists():
        st.error("Model belum ditemukan. Latih dulu dengan "
                 "`python -m src.training.train`.")
        return

    on = st.toggle("Nyalakan kamera")
    left, right = st.columns([3, 2], gap="large")
    frame_slot = left.empty()
    info_slot = right.empty()
    if not on:
        frame_slot.info("Nyalakan kamera untuk mulai. Matikan lagi untuk berhenti.")
        return

    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(0, backend)
    if not cap.isOpened():
        st.error("Kamera tidak bisa dibuka. Tutup aplikasi lain yang memakai kamera, "
                 "lalu coba lagi.")
        return

    predictor = get_predictor()
    try:
        while True:  # berhenti otomatis saat toggle dimatikan (Streamlit rerun)
            ok, frame = cap.read()
            if not ok:
                st.error("Gagal membaca gambar dari kamera.")
                break
            frame = cv2.flip(frame, 1)  # cermin, supaya terasa natural
            annotated, results = predictor.predict_image(frame, min_neighbors)
            frame_slot.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width="stretch")
            if results:
                r = results[0]  # wajah terbesar
                info_slot.markdown(
                    f'<div class="side-card"><div class="verdict">'
                    f'<div class="emo" style="background:{_hex(r.label)}33">{config.EMOJI[r.label]}</div>'
                    f'<div><div class="name">{config.LABELS_ID[r.label]}</div>'
                    f'<div class="conf">yakin {r.confidence:.0%}</div></div></div>'
                    f'{render_bars(r.probabilities, r.label)}</div>',
                    unsafe_allow_html=True)
            else:
                info_slot.markdown('<div class="side-card">Wajah belum terdeteksi. '
                                   'Hadapkan wajah ke kamera.</div>', unsafe_allow_html=True)
    finally:
        cap.release()


# ------------------------------------------------------------------ tab 2
def evaluation_tab():
    summary_path = config.REPORTS_DIR / "summary.json"
    if not summary_path.exists():
        st.info("Belum ada laporan evaluasi. Jalankan "
                "`python -m src.evaluation.evaluate` (atau selesaikan training).")
        return

    s = json.loads(summary_path.read_text())
    st.write("")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{s['accuracy']:.1%}")
    c2.metric("Macro F1", f"{s['macro_f1']:.3f}")
    c3.metric("Weighted F1", f"{s['weighted_f1']:.3f}")
    c4.metric("Data uji", f"{s['n_test']:,}")

    st.subheader("Metrik per kelas")
    table_path = config.REPORTS_DIR / "per_class_metrics.csv"
    if table_path.exists():
        df = pd.read_csv(table_path, index_col=0)
        left, right = st.columns([3, 2], gap="large")
        left.dataframe(df.round(3), width="stretch")
        right.bar_chart(df.loc[NAMES_ID, "f1-score"], height=300, color="#5B5BF0")

    st.subheader("Confusion matrix")
    cm_img = config.REPORTS_DIR / "confusion_matrix.png"
    if cm_img.exists():
        st.image(str(cm_img), width="stretch")

    st.subheader("Ekspresi yang paling sering tertukar")
    pairs_path = config.REPORTS_DIR / "confused_pairs.csv"
    if pairs_path.exists():
        st.dataframe(pd.read_csv(pairs_path).head(10), width="stretch",
                     hide_index=True)

    st.subheader("Kurva training")
    hist_img = config.REPORTS_DIR / "training_history.png"
    if hist_img.exists():
        st.image(str(hist_img), width="stretch")

    st.subheader("Tantangan FER2013: label noise & kualitas rendah")
    err_img = config.REPORTS_DIR / "confident_errors.png"
    if err_img.exists():
        st.image(str(err_img), width="stretch")
    st.caption("Banyak error berkeyakinan tinggi sebenarnya berasal dari label "
               "yang ambigu atau keliru pada dataset, bukan murni kesalahan model. "
               "Gambar FER2013 hanya 48×48 piksel grayscale.")


# ------------------------------------------------------------------ main
def main():
    inject_css()

    with st.sidebar:
        st.markdown('<div class="brand"><div class="logo">😊</div><div><b>Klasifikasi Ekspresi</b>'
                    '<span>VGG16 &bull; FER2013</span></div></div>', unsafe_allow_html=True)

        st.markdown('<div class="side-label">Menu</div>', unsafe_allow_html=True)
        with st.container(key="nav"):
            page = st.radio("Menu", ["Prediksi", "Kamera langsung", "Evaluasi model"],
                            label_visibility="collapsed")

        st.markdown('<div class="side-label">Ketelitian deteksi wajah</div>', unsafe_allow_html=True)
        preset = st.select_slider(
            "Ketelitian", options=list(PRESETS), value="Seimbang",
            label_visibility="collapsed",
            help="Ketat = lebih sedikit salah deteksi, tapi wajah kecil bisa terlewat.")

        st.markdown('<div class="side-label">Ekspresi yang dikenali</div>', unsafe_allow_html=True)
        st.markdown('<div class="side-card"><div class="legend">' + "".join(
            f'<span><i class="dot" style="background:{_hex(c)}"></i>{config.LABELS_ID[c]}</span>'
            for c in config.CLASS_NAMES) + '</div></div>', unsafe_allow_html=True)

    if page == "Prediksi":
        prediction_tab(PRESETS[preset])
    elif page == "Kamera langsung":
        live_tab(PRESETS[preset])
    else:
        page_header("Evaluasi model", "Performa model pada data uji FER2013.")
        evaluation_tab()


main()