import cv2
import numpy as np


class AbsDiffDetector:
    """
    AbsDiff Detector im Boardspace (warped 600x600):
    - reference_frame = gray reference (clean/last state)
    - detect(warped_bgr, gray_optional) -> list of objects:
        {
          "tip_board": (x,y),
          "confidence": float,
          "contour": cnt,
          "extra": {...}
        }
    """

    def __init__(self, board_mask, freeze_mean=20, freeze_max=70):
        self.board_mask = board_mask
        self.reference_frame = None  # gray
        self.FREEZE_MEAN = float(freeze_mean)
        self.FREEZE_MAX = float(freeze_max)

        # Tuning
        self.min_area = 220
        self.max_area = 18000
        self.min_length = 18
        self.merge_dist = 22

    def set_reference(self, frame_bgr):
        self.reference_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    def detect(self, warped_frame_bgr, gray=None):
        if self.reference_frame is None:
            return [], warped_frame_bgr

        if gray is None:
            gray = cv2.cvtColor(warped_frame_bgr, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(self.reference_frame, gray)
        h, w = diff.shape[:2]
        board_center = np.array([w / 2.0, h / 2.0], dtype=np.float32)

        # --- SMART MOTION FREEZE (nur "globales Wackeln" ohne harte lokale Kante) ---
        mean_val = cv2.mean(diff)[0]
        _, max_val, _, _ = cv2.minMaxLoc(diff)
        if mean_val > self.FREEZE_MEAN and max_val < self.FREEZE_MAX:
            return [], warped_frame_bgr

        # Noise / Threshold
        diff_blur = cv2.GaussianBlur(diff, (5, 5), 0)
        _, thr = cv2.threshold(diff_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morph close
        kernel = np.ones((3, 3), np.uint8)
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Micro-noise reject
        white_ratio = cv2.countNonZero(thr) / float(w * h)
        if white_ratio < 0.0002:
            return [], warped_frame_bgr

        # Apply board mask (Score-relevanter Bereich)
        thr = cv2.bitwise_and(thr, self.board_mask)

        contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw = []
        dbg = warped_frame_bgr.copy()

        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < self.min_area or area > self.max_area:
                continue

            if len(cnt) < 5:
                continue

            pts = cnt.reshape(-1, 2).astype(np.float32)

            mean, eigenvecs = cv2.PCACompute(pts, mean=None)
            center = mean[0]  # (x,y)
            axis = eigenvecs[0]

            # axis so drehen, dass er Richtung Boardzentrum zeigt
            if np.dot(axis, (board_center - center)) < 0:
                axis = -axis

            proj = np.dot(pts - center, axis)  # scalar per point
            i_min = int(np.argmin(proj))
            i_max = int(np.argmax(proj))

            p1 = pts[i_min]
            p2 = pts[i_max]
            length = float(np.linalg.norm(p2 - p1))
            if length < self.min_length:
                continue

            # "Tip" = Extrempunkt in Richtung Boardzentrum (max projection entlang axis)
            # statt nur 1 Punkt: Mittelwert der Top-3 (stabiler, weniger Flight-Mitte)
            idxs = np.argsort(proj)[-3:]
            tip = np.mean(pts[idxs], axis=0)

            # Zusatzheuristik gegen "Flight/Mitte":
            # Wenn der Tip weiter vom Zentrum weg ist als das Kontur-Zentrum, dann ist es verdächtig -> skip/penalty
            tip_dist = float(np.linalg.norm(board_center - tip))
            center_dist = float(np.linalg.norm(board_center - center))
            if tip_dist > center_dist + 12:
                # In der Praxis: Flight/Heck ist oft weiter außen als der Kontur-Schwerpunkt
                continue

            width = area / max(length, 1.0)
            width = max(width, 1.0)
            slenderness = length / width

            confidence = length * slenderness * np.log(area + 1.0)

            raw.append({
                "tip_board": (float(tip[0]), float(tip[1])),
                "confidence": float(confidence),
                "contour": cnt,
                "extra": {
                    "area": area,
                    "length": length
                }
            })

        # Merge nahe Tips (Doppelkonturen)
        raw.sort(key=lambda o: o["confidence"], reverse=True)
        merged = []
        for o in raw:
            keep = True
            for m in merged:
                if np.linalg.norm(np.array(o["tip_board"]) - np.array(m["tip_board"])) < self.merge_dist:
                    keep = False
                    break
            if keep:
                merged.append(o)

        # Debug zeichnen (optional — vision.py nutzt VisionDebugger, aber hier ok)
        for o in merged[:10]:
            tx, ty = o["tip_board"]
            cv2.circle(dbg, (int(tx), int(ty)), 5, (0, 0, 255), -1)

        return merged, dbg
