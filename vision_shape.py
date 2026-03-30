import cv2
import numpy as np


class ShapeDetector:
    """
    Shape-/Kontur-basierte Darterkennung im Boardspace (warped 600x600).

    Ziel:
    - neue längliche Objekte erkennen
    - Flight/Mitte häufiger verwerfen
    - Tip im Boardspace zurückgeben

    Rückgabeformat pro Objekt:
    {
        "tip_board": (x, y),
        "confidence": float,
        "contour": cnt,
        "extra": {
            "area": ...,
            "length": ...,
            "width": ...,
            "elongation": ...,
            "solidity": ...
        }
    }
    """

    def __init__(self, board_mask, freeze_mean=20, freeze_max=70):
        self.board_mask = board_mask
        self.reference_gray = None

        self.FREEZE_MEAN = float(freeze_mean)
        self.FREEZE_MAX = float(freeze_max)

        # Tuning
        self.min_area = 300
        self.max_area = 16000
        self.min_length = 16.0
        self.min_elongation = 3.5
        self.max_width = 40.0
        self.merge_dist = 20.0

    def set_reference(self, frame_bgr):
        self.reference_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    def detect(self, warped_frame_bgr, gray=None):
        if self.reference_gray is None:
            return [], warped_frame_bgr

        if gray is None:
            gray = cv2.cvtColor(warped_frame_bgr, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(self.reference_gray, gray)
        h, w = diff.shape[:2]
        board_center = np.array([w / 2.0, h / 2.0], dtype=np.float32)

        # Globales Wackeln ausblenden
        mean_val = cv2.mean(diff)[0]
        _, max_val, _, _ = cv2.minMaxLoc(diff)
        if mean_val > self.FREEZE_MEAN and max_val < self.FREEZE_MAX:
            return [], warped_frame_bgr

        # Vorverarbeitung
        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        _, thr = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel3 = np.ones((3, 3), np.uint8)
        kernel5 = np.ones((5, 5), np.uint8)

        thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel3, iterations=1)
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel5, iterations=1)

        thr = cv2.bitwise_and(thr, self.board_mask)

        white_ratio = cv2.countNonZero(thr) / float(w * h)
        if white_ratio < 0.0002:
            return [], warped_frame_bgr

        contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        debug_img = warped_frame_bgr.copy()
        raw_objects = []

        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < self.min_area or area > self.max_area:
                continue
            if len(cnt) < 5:
                continue

            pts = cnt.reshape(-1, 2).astype(np.float32)

            # minAreaRect -> Länge/Breite
            rect = cv2.minAreaRect(cnt)
            rw, rh = rect[1]
            length = float(max(rw, rh))
            width = float(max(1.0, min(rw, rh)))

            if length < self.min_length:
                continue
            if width > self.max_width:
                continue

            elongation = length / width
            if elongation < self.min_elongation:
                continue

            # Solidity
            hull = cv2.convexHull(cnt)
            hull_area = float(cv2.contourArea(hull))
            if hull_area <= 1.0:
                continue
            solidity = area / hull_area

            # PCA-Hauptachse
            mean, eigenvecs = cv2.PCACompute(pts, mean=None)
            center = mean[0]
            axis = eigenvecs[0]

            # Achse Richtung Boardzentrum orientieren
            if np.dot(axis, (board_center - center)) < 0:
                axis = -axis

            projections = np.dot(pts - center, axis)

            # Nur Punkte "vorne" in Richtung Zentrum
            forward_pts = pts[projections > 0]

            if len(forward_pts) >= 3:
                # aus den 5 board-nächsten forward-Punkten mitteln
                d = np.linalg.norm(forward_pts - board_center, axis=1)
                idx = np.argsort(d)[: min(5, len(forward_pts))]
                tip = np.mean(forward_pts[idx], axis=0)
            else:
                # Fallback: board-nächste Konturpunkte allgemein
                d = np.linalg.norm(pts - board_center, axis=1)
                idx = np.argsort(d)[: min(5, len(pts))]
                tip = np.mean(pts[idx], axis=0)

            tip_dist = float(np.linalg.norm(board_center - tip))
            center_dist = float(np.linalg.norm(board_center - center))

            # Wenn "Spitze" deutlich weiter außen als der Schwerpunkt liegt,
            # ist es oft Flight/Heck -> verwerfen
            if tip_dist > center_dist + 14:
                continue

            # Confidence
            center_bonus = 1.0 / (1.0 + tip_dist / 280.0)
            shape_bonus = min(elongation / 6.0, 2.0)
            solidity_bonus = min(max(solidity, 0.3), 1.2)

            confidence = area * shape_bonus * solidity_bonus * (1.0 + center_bonus)

            raw_objects.append({
                "tip_board": (float(tip[0]), float(tip[1])),
                "confidence": float(confidence),
                "contour": cnt,
                "extra": {
                    "area": area,
                    "length": length,
                    "width": width,
                    "elongation": float(elongation),
                    "solidity": float(solidity)
                }
            })

        # Nahe Kandidaten zusammenfassen
        raw_objects.sort(key=lambda o: o["confidence"], reverse=True)
        merged = []
        for obj in raw_objects:
            keep = True
            for m in merged:
                if np.linalg.norm(np.array(obj["tip_board"]) - np.array(m["tip_board"])) < self.merge_dist:
                    keep = False
                    break
            if keep:
                merged.append(obj)

        for obj in merged[:10]:
            tx, ty = obj["tip_board"]
            cv2.circle(debug_img, (int(tx), int(ty)), 5, (255, 0, 255), -1)

        return merged, debug_img
