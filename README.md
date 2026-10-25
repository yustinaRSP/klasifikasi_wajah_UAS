# Klasifikasi Ekspresi Wajah (FER2013 + VGG16 + Streamlit)

## Arsitektur
```
app.py                      Presentation layer  (Streamlit UI)
config.py                   Konfigurasi terpusat
src/
  inference/predictor.py    Service layer       (detektor + model, gambar -> hasil)
  detection/face_detector.py  Detection layer   (OpenCV Haar Cascade)
  models/builder.py         Model layer         (VGG16 transfer learning)
  data/preprocessing.py     Preprocessing bersama (training = inference)
  data/loader.py            Data layer          (FER2013 + class weight)
  training/train.py         Pipeline training 2 fase
  evaluation/evaluate.py    Evaluation layer (accuracy, F1, confusion matrix, error analysis)
models/                     Model hasil training (.keras)
reports/                    Output evaluasi (csv, json, png)
data/fer2013/{train,test}/  Dataset Kaggle (folder per kelas)
```
Alur inferensi: upload -> Haar Cascade -> crop wajah -> `prepare_face` -> VGG16 -> label + probabilitas.

## Cara pakai
```bash
pip install -r requirements.txt
# ekstrak dataset Kaggle msambare/fer2013 ke data/fer2013/ (isi: train/, test/)
python -m src.training.train     # latih & simpan models/fer_vgg16.keras
streamlit run app.py
```

## Evaluasi
`python -m src.evaluation.evaluate` menghasilkan di `reports/`:
`summary.json` (accuracy, macro/weighted F1), `per_class_metrics.csv`, `confusion_matrix.png/.csv`,
`confused_pairs.csv` (ekspresi yang paling sering tertukar), `confident_errors.png` (indikasi label noise),
dan `training_history.png` (dibuat saat training). Semuanya tampil di tab *Evaluasi Model* pada aplikasi.

## Catatan
Gambar FER2013 sudah berupa crop wajah 48x48, jadi Haar Cascade tidak dipakai saat training/evaluasi,
hanya pada foto yang diupload pengguna (di dataset yang sangat kecil Haar justru sering gagal mendeteksi).

## Alur kerja: Colab (training) → VS Code (app) → GitHub (deploy)
1. Zip folder proyek **tanpa** `.venv` dan `data` → `fer_app.zip`; siapkan `fer2013.zip` dari Kaggle.
2. Upload keduanya ke Google Drive (`MyDrive/fer_project/`), buka `colab_train.ipynb` di Colab (GPU T4), jalankan semua sel.
3. Unduh `hasil_training.zip`, ekstrak, tempatkan `models/fer_vgg16.keras` dan `reports/` ke proyek lokal.
4. Tes lokal: `streamlit run app.py`.
5. Upload ke GitHub (dataset dan `.venv` otomatis diabaikan oleh `.gitignore`), lalu deploy.
=======
# klasifikasi_wajah_UAS

