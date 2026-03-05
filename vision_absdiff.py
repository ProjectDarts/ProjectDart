import cv2
import numpy as np

class AbsDiffDetector:
    """
    Erkennt Dart-Konturen über AbsDiff zur Referenz und bestimmt die Spitze robust:
    Spitze = Punkt auf Convex Hull mit minimaler Distanz zum Board-Zentrum.
    Dadurch wird Flight/Tail (weiter weg vom Zentrum) sehr zuverlässig ausgeschlossen.
    """

    def __init__(self, board_mask, extended_mask=None):
        self.board_mask = board_mask
        self.extended_mask = extended_mask if extended_mask is not None else board_mask
        self.reference_gray = None

        # Tuning
        self.min_area = 220
        self.max_area = 25000
        self.min_len = 20
        self.noise_mean_gate = 1.2  # wenn Diff quasi nix -> skip
        self.white_ratio_gate = 0.00010  # wenn fast nix weiß -> skip

    def set_reference(self, frame_bgr):
        self.reference_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    def detect(self, frame_bgr, gray=None, center=None):
        """
        Returns: (objects, debug_img)
        object keys:
          tip: (x,y) float
          confidence: float
          contour: cnt
          method: "absdiff"
          tip_in_board: bool
        """
        if self.reference_gray is None:
            return [], frame_bgr

        if gray is None:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape[:2]
        if center is None:
            center = (w / 2.0, h / 2.0)
        cx, cy = center

        diff = cv2.absdiff(self.reference_gray, gray)

        # Noise gate: wenn nahezu keine Änderung, sofort raus
        if float(np.mean(diff)) < self.noise_mean_gate:
            return [], frame_bgr

        # Glätten
        diff_blur = cv2.GaussianBlur(diff, (5, 5), 0)

        # OTSU Threshold
        _, thr = cv2.threshold(diff_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morph close gegen Löcher
        kernel = np.ones((3, 3), np.uint8)
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Mikro-noise gate (vor Mask!)
        white_ratio = cv2.countNonZero(thr) / float(w * h)
        if white_ratio < self.white_ratio_gate:
            return [], frame_bgr

        # Nur extended Bereich berücksichtigen (Flight darf drin sein)
        thr = cv2.bitwise_and(thr, self.extended_mask)

        contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        debug = frame_bgr.copy()
        objects = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue
            if len(cnt) < 5:
                continue

            # Hull für stabilere Spitze
            hull = cv2.convexHull(cnt)
            hull_pts = hull.reshape(-1, 2).astype(np.float32)

            # Spitze = Hull-Punkt mit minimaler Distanz zum Zentrum
            dists = np.linalg.norm(hull_pts - np.array([[cx, cy]], dtype=np.float32), axis=1)
            tip_idx = int(np.argmin(dists))
            tip = tuple(hull_pts[tip_idx])

            # Tail = Hull-Punkt mit maximaler Distanz zum Zentrum (für Länge)
            tail_idx = int(np.argmax(dists))
            tail = tuple(hull_pts[tail_idx])

            length = float(np.linalg.norm(np.array(tip) - np.array(tail)))
            if length < self.min_len:
                continue

            # Breite grob über area/length
            width = max(area / max(length, 1.0), 1.0)

            # Confidence: lang + schlank + größer = besser
            slenderness = length / width
            confidence = (length * slenderness) * np.log(area + 1.0)

            # Bonus wenn Tip näher am Boardzentrum liegt (typisch echte Spitze)
            tip_dist = float(np.linalg.norm(np.array(tip) - np.array([cx, cy])))
            confidence *= 1.0 / (1.0 + (tip_dist / 200.0))

            tip_in_board = False
            tx, ty = int(round(tip[0])), int(round(tip[1]))
            if 0 <= tx < w and 0 <= ty < h:
                tip_in_board = (self.board_mask[ty, tx] > 0)

            objects.append({
                "tip": tip,
                "confidence": float(confidence),
                "area": float(area),
                "contour": cnt,
                "method": "absdiff",
                "tip_in_board": bool(tip_in_board),
            })

            # Debug
            cv2.drawContours(debug, [cnt], -1, (0, 255, 0), 1)
            cv2.circle(debug, (int(tip[0]), int(tip[1])), 6, (0, 0, 255), -1)
            cv2.circle(debug, (int(tail[0]), int(tail[1])), 6, (255, 0, 0), 1)

        return objects, debug
