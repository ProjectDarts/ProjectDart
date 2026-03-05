import cv2
import numpy as np

class VectorDetector:
    """
    Vektor/Line-basierte Darterkennung im Full-Frame:
    - Canny edges
    - HoughLinesP
    - Tip = Linien-Endpunkt in Richtung Boardzentrum
    Liefert Kandidaten im Full-Frame: tip=(x,y), confidence
    """

    def __init__(self):
        # Tuning (Startwerte)
        self.canny1 = 60
        self.canny2 = 160
        self.blur = (5, 5)

        self.hough_threshold = 40
        self.min_line_len = 35
        self.max_line_gap = 6

        # Filter
        self.max_candidates = 8
        self.tip_merge_dist = 18

    def detect(self, frame_bgr, board_center_full=None, board_roi_mask_full=None):
        """
        board_center_full: (x,y) im Full-Frame
        board_roi_mask_full: optional uint8 Maske im Full-Frame (255 im Board/nahe Board), um Linien zu gewichten.
        """
        h, w = frame_bgr.shape[:2]

        if board_center_full is None:
            board_center_full = (w / 2.0, h / 2.0)
        bc = np.array(board_center_full, dtype=np.float32)

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, self.blur, 0)

        edges = cv2.Canny(gray, self.canny1, self.canny2)

        # optional ROI: nicht hart croppen, nur edges im ROI verstärken / außerhalb schwächen
        if board_roi_mask_full is not None:
            # Nur leicht: außerhalb ROI edges dämpfen
            edges = cv2.bitwise_and(edges, board_roi_mask_full)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_len,
            maxLineGap=self.max_line_gap
        )

        if lines is None:
            return []

        raw = []
        for l in lines[:200]:
            x1, y1, x2, y2 = l[0]
            p1 = np.array([x1, y1], dtype=np.float32)
            p2 = np.array([x2, y2], dtype=np.float32)

            length = float(np.linalg.norm(p2 - p1))
            if length < self.min_line_len:
                continue

            # Tip-Endpunkt wählen: der Endpunkt "zeigt" zum Boardzentrum
            # Wir nehmen den Endpunkt, der näher am Boardzentrum ist (Tip steckt Richtung Board)
            d1 = float(np.linalg.norm(bc - p1))
            d2 = float(np.linalg.norm(bc - p2))
            tip = p1 if d1 < d2 else p2

            # Confidence: längere Linien + Nähe zum Zentrum (Darts sind meist nahe am Board)
            # (Das ist bewusst simpel – später kann man das verbessern)
            dist_to_center = float(np.linalg.norm(bc - tip))
            center_bonus = 1.0 / (1.0 + (dist_to_center / 300.0))  # skaliert
            confidence = length * (1.5 + 2.5 * center_bonus)

            raw.append({
                "tip": (float(tip[0]), float(tip[1])),
                "confidence": float(confidence),
                "line": (int(x1), int(y1), int(x2), int(y2)),
                "length": length
            })

        # Sortieren und Tips mergen (Doppellinien)
        raw.sort(key=lambda o: o["confidence"], reverse=True)

        merged = []
        for obj in raw:
            keep = True
            for m in merged:
                if np.linalg.norm(np.array(obj["tip"]) - np.array(m["tip"])) < self.tip_merge_dist:
                    keep = False
                    break
            if keep:
                merged.append(obj)
            if len(merged) >= self.max_candidates:
                break

        return merged
