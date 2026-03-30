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

        # Weniger empfindlich / robuster
        self.min_area = 350
        self.max_area = 12000
        self.min_length = 28
        self.merge_dist = 22

        # Neuer fester Mindest-Threshold gegen Phantom-Diffs
        self.min_diff_threshold = 18

        # Leichte Erosion der Boardmaske, damit Randflackern weniger zählt
        kernel_mask = np.ones((9, 9), np.uint8)
        self.inner_board_mask = cv2.erode(self.board_mask, kernel_mask, iterations=1)

    def _prepare_gray(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        return gray

    def set_reference(self, frame_bgr):
        self.reference_frame = self._prepare_gray(frame_bgr)

    def detect(self, warped_frame_bgr, gray=None):
        if self.reference_frame is None:
            return [], warped_frame_bgr

        if gray is None:
            gray = self._prepare_gray(warped_frame_bgr)
        else:
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

        diff = cv2.absdiff(self.reference_frame, gray)

        # Nur relevanten Bereich betrachten
        diff_masked = cv2.bitwise_and(diff, self.inner_board_mask)

        h, w = diff_masked.shape[:2]
        board_center = np.array([w / 2.0, h / 2.0], dtype=np.float32)

        # Nur Werte innerhalb der Maske für Freeze auswerten
        mask_pixels = diff_masked[self.inner_board_mask > 0]
        if mask_pixels.size == 0:
            return [], warped_frame_bgr

        mean_val = float(np.mean(mask_pixels))
        max_val = float(np.max(mask_pixels))

        # Globales leichtes Flackern ignorieren
        if mean_val > self.FREEZE_MEAN and max_val < self.FREEZE_MAX:
            return [], warped_frame_bgr

        # Nochmals leicht glätten
        diff_blur = cv2.GaussianBlur(diff_masked, (5, 5), 0)

        # Otsu verwenden, aber nie unter festen Mindestwert gehen
        otsu_thr, _ = cv2.threshold(
            diff_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        final_thr = max(self.min_diff_threshold, int(otsu_thr))

        _, thr = cv2.threshold(diff_blur, final_thr, 255, cv2.THRESH_BINARY)

        # Micro-noise reject jetzt NACH Maskierung
        white_ratio = cv2.countNonZero(thr) / float(cv2.countNonZero(self.inner_board_mask))
        if white_ratio < 0.00015:
            return [], warped_frame_bgr

        # Zu viele weiße Pixel = meistens kein Dart, sondern großes Flackern / Störung
        if white_ratio > 0.08:
            return [], warped_frame_bgr

        # Morphology: erst kleine Flecken entfernen, dann leichte Lücken schließen
        kernel_open = np.ones((3, 3), np.uint8)
        kernel_close = np.ones((5, 5), np.uint8)

        thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel_open, iterations=1)
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        # Sicherheitshalber nochmals maskieren
        thr = cv2.bitwise_and(thr, self.inner_board_mask)

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
            center = mean[0]
            axis = eigenvecs[0]

            # Achse Richtung Boardzentrum drehen
            if np.dot(axis, (board_center - center)) < 0:
                axis = -axis

            proj = np.dot(pts - center, axis)

            i_min = int(np.argmin(proj))
            i_max = int(np.argmax(proj))

            p1 = pts[i_min]
            p2 = pts[i_max]
            length = float(np.linalg.norm(p2 - p1))
            if length < self.min_length:
                continue

            idxs = np.argsort(proj)[-3:]
            tip = np.mean(pts[idxs], axis=0)

            tip_dist = float(np.linalg.norm(board_center - tip))
            center_dist = float(np.linalg.norm(board_center - center))
            if tip_dist > center_dist + 12:
                continue

            width = area / max(length, 1.0)
            width = max(width, 1.0)
            slenderness = length / width

            # neue zusätzliche Formfilter gegen Blob-Phantome
            if width > 32:
                continue

            if slenderness < 2.2:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = max(bw, bh) / max(1, min(bw, bh))
            if aspect < 1.4:
                continue

            confidence = length * slenderness * np.log(area + 1.0)

            raw.append({
                "tip_board": (float(tip[0]), float(tip[1])),
                "confidence": float(confidence),
                "contour": cnt,
                "extra": {
                    "area": area,
                    "length": length,
                    "width": width,
                    "slenderness": slenderness,
                    "aspect": aspect,
                    "threshold": final_thr,
                    "white_ratio": white_ratio,
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

        # Debug
        for o in merged[:10]:
            tx, ty = o["tip_board"]
            cv2.circle(dbg, (int(tx), int(ty)), 5, (0, 0, 255), -1)

        return merged, dbg