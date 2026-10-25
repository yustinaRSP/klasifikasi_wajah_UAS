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

st.set_page_config(page_title="Klasifikasi Ekspresi Wajah", page_icon="😊", layout="wide",
                   initial_sidebar_state="expanded")


@st.cache_resource(show_spinner="Memuat model...")
def get_predictor() -> ExpressionPredictor:
    return ExpressionPredictor(config.MODEL_PATH, FaceDetector())


def read_image(file) -> np.ndarray:
    img = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _hex(label: str) -> str:
    b, g, r = config.COLORS_BGR[label]
    return f"#{r:02x}{g:02x}{b:02x}"


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Plus Jakarta Sans', system-ui, sans-serif; }
.stApp {
    background:
        radial-gradient(900px 420px at 85% -10%, rgba(255,181,71,.10), transparent 60%),
        radial-gradient(700px 380px at 0% 0%, rgba(124,140,255,.08), transparent 60%),
        #12141C;
}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1180px; }

/* Hero */
.hero { padding: 1.2rem 0 1.6rem; }
.hero h1 { font-size: clamp(2rem, 4vw, 3.2rem); font-weight: 800; letter-spacing: -.03em;
           line-height: 1.08; margin: 0 0 .7rem; color: #F4F5FA; }
.hero p { color: #9AA1B5; font-size: 1.05rem; max-width: 52ch; margin: 0 0 1rem; line-height: 1.6; }
.chips { display: flex; flex-wrap: wrap; gap: .5rem; }
.chip { font-size: .8rem; font-weight: 600; color: #C9CEDD; padding: .35rem .8rem;
        border: 1px solid #2B3040; border-radius: 999px; background: #181B26; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: .4rem; border-bottom: 1px solid #262B3A; }
.stTabs [data-baseweb="tab"] { height: 46px; padding: 0 1.1rem; font-weight: 600; color: #9AA1B5;
                               border-radius: 10px 10px 0 0; }
.stTabs [aria-selected="true"] { color: #FFB547 !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color: #FFB547 !important; height: 3px; }

/* Uploader */
[data-testid="stFileUploader"] section {
    border: 2px dashed #343A4F; border-radius: 18px; background: #161923; padding: 2rem 1.4rem;
    transition: border-color .2s, background .2s; }
[data-testid="stFileUploader"] section:hover { border-color: #FFB547; background: #1A1E2A; }
[data-testid="stFileUploader"] button { border-radius: 10px; font-weight: 600; }

/* Cards */
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 18px !important;
    border-color: #272C3B !important; background: #171A24; }
[data-testid="stMetric"] { background: #171A24; border: 1px solid #272C3B; border-radius: 16px;
                           padding: 1rem 1.2rem; }
[data-testid="stMetricLabel"] p { color: #9AA1B5; font-weight: 600; }
[data-testid="stMetricValue"] { font-weight: 800; letter-spacing: -.02em; }
[data-testid="stImage"] img { border-radius: 14px; }
[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; }
h2, h3 { letter-spacing: -.02em; font-weight: 700; }

/* Sidebar */
[data-testid="stSidebar"] { background: #151823; border-right: 1px solid #232838; }
.legend { display: flex; flex-direction: column; gap: .45rem; margin-top: .4rem; }
.legend div { display: flex; align-items: center; gap: .6rem; font-size: .92rem; color: #D5D9E6; }
.dot { width: 10px; height: 10px; border-radius: 50%; flex: none; }

/* Hasil prediksi */
.summary { display: flex; align-items: baseline; gap: .6rem; margin: .2rem 0 1rem; }
.summary b { font-size: 2.4rem; font-weight: 800; color: #FFB547; letter-spacing: -.03em; }
.summary span { color: #9AA1B5; font-weight: 500; }
.verdict { display: flex; align-items: center; gap: .8rem; margin-bottom: .2rem; }
.verdict .emo { font-size: 2.6rem; line-height: 1; }
.verdict .name { font-size: 1.7rem; font-weight: 800; letter-spacing: -.02em; line-height: 1.1; }
.verdict .conf { color: #9AA1B5; font-size: .9rem; font-weight: 500; }
.bars { display: flex; flex-direction: column; gap: .5rem; margin-top: .9rem; }
.bar { display: grid; grid-template-columns: 5.2rem 1fr 3.2rem; align-items: center; gap: .7rem;
       font-size: .85rem; color: #9AA1B5; }
.bar.top { color: #F4F5FA; font-weight: 700; }
.track { height: 8px; background: #232838; border-radius: 99px; overflow: hidden; }
.fill { height: 100%; border-radius: 99px; }
.bar .pct { text-align: right; font-variant-numeric: tabular-nums; }
.empty { text-align: center; color: #7F8699; padding: 1.4rem .5rem 0; font-size: .92rem; }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def render_bars(probs: dict, top: str) -> str:
    rows = []
    for k, v in sorted(probs.items(), key=lambda kv: -kv[1]):
        color = _hex(k) if k == top else "#4A5168"
        rows.append(
            f'<div class="bar{" top" if k == top else ""}">'
            f'<span>{config.LABELS_ID[k]}</span>'
            f'<div class="track"><div class="fill" style="width:{v*100:.1f}%;background:{color}"></div></div>'
            f'<span class="pct">{v:.0%}</span></div>')
    return f'<div class="bars">{"".join(rows)}</div>'


# ------------------------------------------------------------------ tab 1
def prediction_tab(min_neighbors: int):
    if not config.MODEL_PATH.exists():
        st.error("Model belum ditemukan. Latih dulu dengan "
                 "`python -m src.training.train`.")
        return

    file = st.file_uploader("Unggah foto wajah", type=["jpg", "jpeg", "png"],
                            label_visibility="collapsed")
    if file is None:
        st.markdown('<div class="empty">Seret foto ke kotak di atas. Wajah dideteksi '
                    'otomatis, tidak perlu di-crop.</div>', unsafe_allow_html=True)
        return

    image = read_image(file)
    with st.spinner("Menganalisis ekspresi..."):
        annotated, results = get_predictor().predict_image(image, min_neighbors)

    st.write("")
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
    with right:
        if not results:
            st.warning("Tidak ada wajah terdeteksi. Coba foto yang lebih jelas "
                       "atau turunkan nilai Min neighbors.")
            return
        st.markdown(f'<div class="summary"><b>{len(results)}</b>'
                    f'<span>wajah terdeteksi</span></div>', unsafe_allow_html=True)
        for i, r in enumerate(results, 1):
            with st.container(border=True):
                c1, c2 = st.columns([1, 2], gap="medium")
                c1.image(r.face_rgb, use_container_width=True)
                c2.markdown(
                    f'<div class="verdict"><span class="emo">{config.EMOJI[r.label]}</span>'
                    f'<div><div class="name" style="color:{_hex(r.label)}">'
                    f'{config.LABELS_ID[r.label]}</div>'
                    f'<div class="conf">Wajah {i} &bull; yakin {r.confidence:.1%}</div></div></div>',
                    unsafe_allow_html=True)
                st.markdown(render_bars(r.probabilities, r.label), unsafe_allow_html=True)


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
        left.dataframe(df.round(3), use_container_width=True)
        right.bar_chart(df.loc[NAMES_ID, "f1-score"], height=300, color="#FFB547")

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
    inject_css()

    st.markdown(
        '<div class="hero"><h1>Baca ekspresi wajah dari sebuah foto</h1>'
        '<p>Unggah foto, lalu model mengenali satu dari tujuh ekspresi dasar '
        'beserta tingkat keyakinannya.</p>'
        '<div class="chips"><span class="chip">VGG16 Transfer Learning</span>'
        '<span class="chip">FER2013</span><span class="chip">Haar Cascade</span>'
        '<span class="chip">7 ekspresi</span></div></div>',
        unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Pengaturan deteksi")
        min_neighbors = st.slider(
            "Min neighbors", 3, 10, 5,
            help="Lebih tinggi = lebih ketat (lebih sedikit salah deteksi).")
        st.markdown("**Ekspresi yang dikenali**")
        st.markdown(
            '<div class="legend">' + "".join(
                f'<div><span class="dot" style="background:{_hex(c)}"></span>'
                f'{config.EMOJI[c]} {config.LABELS_ID[c]}</div>'
                for c in config.CLASS_NAMES) + '</div>',
            unsafe_allow_html=True)

    tab_pred, tab_eval = st.tabs(["Prediksi", "Evaluasi model"])
    with tab_pred:
        prediction_tab(min_neighbors)
    with tab_eval:
        evaluation_tab()


main()