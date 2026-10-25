"""Service layer: menggabungkan detektor wajah + model klasifikasi."""
from dataclasses import dataclass

import cv2
import numpy as np
from tensorflow import keras

import config
from src.data.preprocessing import prepare_face
from src.detection.face_detector import Box, FaceDetector


@dataclass
class FaceResult:
    box: Box
    label: str                    # nama kelas (inggris, sesuai dataset)
    confidence: float
    probabilities: dict           # {kelas: probabilitas}
    face_rgb: np.ndarray          # crop wajah untuk ditampilkan


class ExpressionPredictor:
    def __init__(self, model_path, detector: FaceDetector):
        self.detector = detector
        self.model = keras.models.load_model(model_path, compile=False)

    def predict_image(self, image_bgr: np.ndarray, min_neighbors: int = 5):
        """Return (gambar_beranotasi_bgr, list[FaceResult])."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        boxes = self.detector.detect(gray, min_neighbors)
        annotated = image_bgr.copy()
        results: list[FaceResult] = []
        if not boxes:
            return annotated, results

        batch = np.stack([prepare_face(gray[y:y + h, x:x + w])
                          for x, y, w, h in boxes])
        probs = self.model.predict(batch, verbose=0)

        for (x, y, w, h), p in zip(boxes, probs):
            idx = int(np.argmax(p))
            label = config.CLASS_NAMES[idx]
            results.append(FaceResult(
                box=(x, y, w, h), label=label, confidence=float(p[idx]),
                probabilities={c: float(v) for c, v in zip(config.CLASS_NAMES, p)},
                face_rgb=cv2.cvtColor(image_bgr[y:y + h, x:x + w], cv2.COLOR_BGR2RGB),
            ))
            self._draw(annotated, results[-1])
        return annotated, results

    @staticmethod
    def _draw(img: np.ndarray, r: FaceResult) -> None:
        x, y, w, h = r.box
        scale = max(img.shape[1] / 900, 0.6)
        thick = max(int(scale * 2), 1)
        color = config.COLORS_BGR[r.label]
        text = f"{config.LABELS_ID[r.label]} {r.confidence:.0%}"
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thick)
        ty = y - 8 if y > 30 else y + h + int(25 * scale)
        cv2.putText(img, text, (x, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    scale * 0.7, color, thick, cv2.LINE_AA)
