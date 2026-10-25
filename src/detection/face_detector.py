"""Detection layer: deteksi wajah otomatis dengan Haar Cascade."""
import cv2
import numpy as np

Box = tuple[int, int, int, int]  # x, y, w, h


class FaceDetector:
    def __init__(self, scale_factor: float = 1.1, min_size: tuple = (48, 48)):
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(path)
        if self._cascade.empty():
            raise RuntimeError(f"Gagal memuat Haar Cascade: {path}")
        self.scale_factor = scale_factor
        self.min_size = min_size

    def detect(self, gray: np.ndarray, min_neighbors: int = 5) -> list[Box]:
        """Terima gambar grayscale, kembalikan daftar kotak (terbesar dulu)."""
        eq = cv2.equalizeHist(gray)
        faces = self._cascade.detectMultiScale(
            eq, scaleFactor=self.scale_factor,
            minNeighbors=min_neighbors, minSize=self.min_size)
        boxes = [tuple(int(v) for v in f) for f in faces]
        return sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
