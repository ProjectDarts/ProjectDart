import cv2
import numpy as np

class AbsDiffDetector:
    """
    AbsDiff-basierte Darterkennung im Full-Frame.
    Liefert Tip-Kandidaten (x,y) im Full-Frame.
    """

    def __init__(self):
        self.ref_gray = None

        # Tuning (Startwerte)
        self.min_area = 120
        self.max_area = 30000
        self.min_length = 18

        self.blur = (5, 5)

        # Tip-Glättung / Doppelkonturen
        self.merge_tip_dist = 25

    def set_reference(self, frame_bgr):
        self.ref_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    def detect(self, frame_bgr, board_center_full=None):
        if self.ref_gray is None:
            return []

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(self.ref_gray, gray)

        # Noise
        diff = cv2.GaussianBlur(diff, self.blur, 0)

        # OTSU
        _, thr = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morph close gegen Fragmentierung
        kernel = np.ones((3, 3), np.uint8)
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = gray.shape[:2]
        if board_center_full is None:
            board_center_full = np.array([w / 2, h / 2], dtype=np.float32)
        else:
            board_center_full = np.array(board_center_full, dtype=np.float32)

        raw = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue
            if len(cnt) < 5:
                continue

            pts = cnt.reshape(-1, 2).astype(np.float32)
            mean, eigenvecs = cv2.PCACompute(pts, mean=None)
            center = mean[0]
            axis = eigenvecs[0]  # Hauptachse

            # Richtung zum Board-Zentrum wählen (damit Tip eher "ins Board" zeigt)
            if np.dot(axis, (board_center_full - center)) < 0:
                axis = -axis

            proj = np.dot(pts - center, axis)
            p_min = pts[np.argmin(proj)]
            p_max = pts[np.argmax(proj)]
            length = float(np.linalg.norm(p_max - p_min))
            if length < self.min_length:
                continue

            # Tip = die "besten" Punkte entlang der Achse (Richtung board_center)
            idxs = np.argsort(proj)[-3:]
            tip = np.mean(pts[idxs], axis=0)

            # Confidence: lang + schlank + Fläche (Flight produziert oft breite Formen -> schlechter)
            width = max(area / max(length, 1e-6), 1.0)
            slenderness = length / width
            confidence = float(length * slenderness * np.log(area + 1))

            raw.append({
                "tip": (float(tip[0]), float(tip[1])),
                "confidence": confidence,
                "area": float(area),
                "contour": cnt
            })

        # Doppelkonturen mergen (Tip-Nähe)
        merged = []
        for obj in sorted(raw, key=lambda o: o["confidence"], reverse=True):
            keep = True
            for m in merged:
                if np.linalg.norm(np.array(obj["tip"]) - np.array(m["tip"])) < self.merge_tip_dist:
                    keep = False
                    break
            if keep:
                merged.append(obj)

        return merged
