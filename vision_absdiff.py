import cv2
import numpy as np


class AbsDiffDetector:
    """
    AbsDiff-basierte Darterkennung im WARP/Boardspace (z.B. 600x600):
    - arbeitet auf warped_frame (BGR) + optional gray
    - nutzt reference_frame (gray)
    - liefert Kandidaten: tip_board=(x,y), confidence, contour
    """

    def __init__(self, board_mask, freeze_mean=20, freeze_max=70):
        self.board_mask = board_mask
        self.reference_gray = None
        self.FREEZE_MEAN = float(freeze_mean)
        self.FREEZE_MAX = float(freeze_max)

        # Tuning
        self.blur = (5, 5)
        self.min_area = 220
        self.max_area = 14000
        self.white_ratio_min = 0.00015
        self.tip_merge_dist = 22

    def set_reference(self, warped_frame_bgr):
        self.reference_gray = cv2.cvtColor(warped_frame_bgr, cv2.COLOR_BGR2GRAY)

    def detect(self, warped_frame_bgr, gray=None):
        if self.reference_gray is None:
            return [], warped_frame_bgr

        if gray is None:
            gray = cv2.cvtColor(warped_frame_bgr, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape[:2]
        board_center = np.array([w / 2.0, h / 2.0], dtype=np.float32)

        diff = cv2.absdiff(self.reference_gray, gray)

        # Smart Motion Freeze (nur falls globales Wackeln, ohne starke lokale Kanten)
        mean_val = float(cv2.mean(diff)[0])
        _, max_val, _, _ = cv2.minMaxLoc(diff)
        if mean_val > self.FREEZE_MEAN and max_val < self.FREEZE_MAX:
            return [], warped_frame_bgr

        diff = cv2.GaussianBlur(diff, self.blur, 0)

        # OTSU Threshold
        _, thr = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morph Close
        kernel = np.ones((3, 3), np.uint8)
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Micro-noise Filter
        white_ratio = cv2.countNonZero(thr) / float(w * h)
        if white_ratio < self.white_ratio_min:
            return [], warped_frame_bgr

        # Board mask
        thr = cv2.bitwise_and(thr, self.board_mask)

        contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw = []
        debug = warped_frame_bgr.copy()

        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < self.min_area or area > self.max_area:
                continue

            if len(cnt) < 8:
                continue

            pts = cnt.reshape(-1, 2).astype(np.float32)

            # Tip-Strategie:
            # 1) finde die Punkte, die dem Board-Zentrum am nächsten sind
            dists = np.linalg.norm(pts - board_center, axis=1)
            k = max(3, int(0.06 * len(pts)))  # 6% der Punkte (mind. 3)
            idxs = np.argsort(dists)[:k]
            tip = pts[idxs].mean(axis=0)

            # Axis/Length für Confidence
            mean, eigenvecs = cv2.PCACompute(pts, mean=None)
            axis = eigenvecs[0]
            proj = np.dot(pts - mean[0], axis)
            p1 = pts[np.argmin(proj)]
            p2 = pts[np.argmax(proj)]
            length = float(np.linalg.norm(p2 - p1))
            if length < 12:
                continue

            width = max(1.0, area / max(length, 1e-6))
            slenderness = length / width

            # Confidence: bevorzugt schlank & lang & genügend Fläche
            confidence = (length * (slenderness ** 1.15)) * np.log(area + 1.0)

            raw.append({
                "tip_board": (float(tip[0]), float(tip[1])),
                "confidence": float(confidence),
                "contour": cnt,
                "extra": {
                    "area": float(area),
                    "length": float(length),
                    "slenderness": float(slenderness)
                }
            })

        # Merge (Doppelkonturen)
        raw.sort(key=lambda o: o["confidence"], reverse=True)
        merged = []
        for obj in raw:
            keep = True
            for m in merged:
                if np.linalg.norm(np.array(obj["tip_board"]) - np.array(m["tip_board"])) < self.tip_merge_dist:
                    keep = False
                    if obj["confidence"] > m["confidence"]:
                        m.update(obj)
                    break
            if keep:
                merged.append(obj)

        # Debug zeichnen
        for obj in merged:
            cx, cy = obj["tip_board"]
            cv2.circle(debug, (int(cx), int(cy)), 6, (0, 0, 255), -1)

        return merged, debug
