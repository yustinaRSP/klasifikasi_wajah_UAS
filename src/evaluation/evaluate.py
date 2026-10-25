"""Evaluation layer: accuracy, F1 per kelas, confusion matrix, analisis
ekspresi yang tertukar, dan error dengan keyakinan tinggi (indikasi label noise).

Jalankan mandiri:  python -m src.evaluation.evaluate
"""
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from tensorflow import keras

import config
from src.data.loader import load_test_dataset

NAMES_ID = [config.LABELS_ID[c] for c in config.CLASS_NAMES]


# ---------- Prediksi ----------
def predict_all(model, test_ds):
    y_true = np.concatenate([np.argmax(y, axis=1) for _, y in test_ds])
    probs = model.predict(test_ds, verbose=0)
    return y_true, probs


# ---------- Tabel ----------
def per_class_table(y_true, y_pred) -> pd.DataFrame:
    rep = classification_report(
        y_true, y_pred, labels=range(config.NUM_CLASSES),
        target_names=NAMES_ID, output_dict=True, zero_division=0)
    rep.pop("accuracy", None)
    return pd.DataFrame(rep).T[["precision", "recall", "f1-score", "support"]]


def confused_pairs(cm: np.ndarray) -> pd.DataFrame:
    rows = []
    for i in range(config.NUM_CLASSES):
        for j in range(config.NUM_CLASSES):
            if i != j and cm[i, j] > 0:
                rows.append({
                    "Ekspresi asli": NAMES_ID[i],
                    "Diprediksi sebagai": NAMES_ID[j],
                    "Jumlah": int(cm[i, j]),
                    "% dari kelas asli": round(100 * cm[i, j] / cm[i].sum(), 1),
                })
    return (pd.DataFrame(rows).sort_values("Jumlah", ascending=False)
            .reset_index(drop=True))


# ---------- Plot ----------
def plot_confusion(cm: np.ndarray, out) -> None:
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    panels = ((axes[0], cm, "Confusion matrix (jumlah)", "d"),
              (axes[1], cm_norm, "Dinormalisasi per kelas asli (= recall)", ".2f"))
    for ax, data, title, fmt in panels:
        ax.imshow(data, cmap="Blues")
        ax.set_title(title)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Asli")
        ax.set_xticks(range(config.NUM_CLASSES))
        ax.set_xticklabels(NAMES_ID, rotation=45, ha="right")
        ax.set_yticks(range(config.NUM_CLASSES))
        ax.set_yticklabels(NAMES_ID)
        thresh = data.max() / 2
        for i in range(config.NUM_CLASSES):
            for j in range(config.NUM_CLASSES):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center",
                        color="white" if data[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_confident_errors(paths, y_true, probs, out, n=24) -> None:
    """Gambar yang salah diprediksi dengan keyakinan tertinggi.
    Banyak di antaranya kemungkinan label noise atau gambar ambigu."""
    y_pred = probs.argmax(1)
    conf = probs.max(1)
    wrong = np.where(y_true != y_pred)[0]
    if len(wrong) == 0:
        return
    top = wrong[np.argsort(-conf[wrong])][:n]
    cols = 6
    rows = math.ceil(len(top) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.6))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, k in zip(axes, top):
        ax.imshow(Image.open(paths[k]).convert("L"), cmap="gray")
        ax.set_title(f"Asli: {NAMES_ID[y_true[k]]}\n"
                     f"Pred: {NAMES_ID[y_pred[k]]} ({conf[k]:.0%})", fontsize=8)
    fig.suptitle("Error dengan keyakinan tertinggi (indikasi label noise)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_history(history: dict, split_epoch: int, out_dir=config.REPORTS_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, title in ((axes[0], "accuracy", "Accuracy"), (axes[1], "loss", "Loss")):
        ax.plot(history[key], label="train")
        ax.plot(history[f"val_{key}"], label="validasi")
        ax.axvline(split_epoch - 0.5, color="gray", ls="--", label="mulai fine-tuning")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "training_history.png", dpi=150)
    plt.close(fig)


# ---------- Orkestrasi ----------
def evaluate_model(model, test_ds, file_paths, out_dir=config.REPORTS_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true, probs = predict_all(model, test_ds)
    y_pred = probs.argmax(1)
    cm = confusion_matrix(y_true, y_pred, labels=range(config.NUM_CLASSES))

    table = per_class_table(y_true, y_pred)
    pairs = confused_pairs(cm)
    summary = {
        "n_test": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }

    table.to_csv(out_dir / "per_class_metrics.csv")
    pairs.to_csv(out_dir / "confused_pairs.csv", index=False)
    pd.DataFrame(cm, index=NAMES_ID, columns=NAMES_ID).to_csv(out_dir / "confusion_matrix.csv")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    plot_confusion(cm, out_dir / "confusion_matrix.png")
    plot_confident_errors(file_paths, y_true, probs, out_dir / "confident_errors.png")

    print("\n=== Evaluasi pada test set ===")
    print(f"Accuracy   : {summary['accuracy']:.4f}")
    print(f"Macro F1   : {summary['macro_f1']:.4f}")
    print(f"Weighted F1: {summary['weighted_f1']:.4f}\n")
    print(table.round(3).to_string())
    print("\n5 ekspresi yang paling sering tertukar:")
    print(pairs.head(5).to_string(index=False))
    print(f"\nLaporan tersimpan di: {out_dir}")
    return summary


def main():
    model = keras.models.load_model(config.MODEL_PATH)
    test_ds, paths = load_test_dataset()
    evaluate_model(model, test_ds, paths)


if __name__ == "__main__":
    main()
