import cv2
import numpy as np


class VectorDetector:
    """
    Vektor/Line-basierte Darterkennung im Full-Frame:
    - Canny edges
    - HoughLinesP
    - Tip = Linien-Endpunkt Richtung Boardzentrum (näher am Zentrum)
    Liefert Kandidaten im Full-Frame: tip=(x,y), confidence, line=(x1,y1,x2,y2)

    Wichtige Schutzmechanismen:
    - harte ROI-Maskierung
    - Linien durch/nahe Boardzentrum werden verworfen
    - typische Spider-Linien werden reduziert
    """

    def __init__(self):
        self.canny1 = 70
        self.canny2 = 180
        self.blur = (5, 5)

        self.hough_threshold = 65
        self.min_line_len = 55
        self.max_line_gap = 2

        self.max_candidates = 3
        self.tip_merge_dist = 22

        # Neue Filter gegen Spider-/Boardlinien
        self.min_tip_center_dist = 28       # Tip nicht exakt im Zentrum
        self.max_tip_center_dist = 235      # Tip nicht weit außerhalb des Boards
        self.min_mid_center_dist = 70       # Linienmittelpunkt nicht zu zentrumsnah
        self.min_line_center_dist = 28      # Linie darf Zentrum nicht zu nah schneiden

    def _point_line_distance(self, p, a, b):
        """
        Abstand Punkt p zu Liniensegment a-b.
        """
        p = np.array(p, dtype=np.float32)
        a = np.array(a, dtype=np.float32)
        b = np.array(b, dtype=np.float32)

        ab = b - a
        ab_len2 = float(np.dot(ab, ab))
        if ab_len2 < 1e-6:
            return float(np.linalg.norm(p - a))

        t = float(np.dot(p - a, ab) / ab_len2)
        t = max(0.0, min(1.0, t))
        proj = a + t * ab
        return float(np.linalg.norm(p - proj))

    def _prepare_roi_mask(self, board_roi_mask_full):
        """
        ROI etwas nach innen ziehen, damit Rand und Boardstruktur weniger reinspielen.
        """
        if board_roi_mask_full is None:
            return None

        kernel = np.ones((9, 9), np.uint8)
        roi = cv2.erode(board_roi_mask_full, kernel, iterations=1)
        return roi

    def detect(self, frame_bgr, board_center_full=None, board_roi_mask_full=None):
        h, w = frame_bgr.shape[:2]

        if board_center_full is None:
            board_center_full = (w / 2.0, h / 2.0)
        bc = np.array(board_center_full, dtype=np.float32)

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, self.blur, 0)

        edges = cv2.Canny(gray, self.canny1, self.canny2)

        roi_mask = self._prepare_roi_mask(board_roi_mask_full)
        if roi_mask is not None:
            edges = cv2.bitwise_and(edges, roi_mask)

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
        for l in lines[:250]:
            x1, y1, x2, y2 = l[0]
            p1 = np.array([x1, y1], dtype=np.float32)
            p2 = np.array([x2, y2], dtype=np.float32)

            length = float(np.linalg.norm(p2 - p1))
            if length < self.min_line_len:
                continue

            # Tip-Endpunkt = näher am Boardzentrum
            d1 = float(np.linalg.norm(bc - p1))
            d2 = float(np.linalg.norm(bc - p2))
            tip = p1 if d1 < d2 else p2
            tail = p2 if d1 < d2 else p1

            tip_dist = float(np.linalg.norm(bc - tip))
            tail_dist = float(np.linalg.norm(bc - tail))
            mid = (p1 + p2) * 0.5
            mid_dist = float(np.linalg.norm(bc - mid))

            # 1) Tip nicht exakt im Zentrum und nicht zu weit außen
            if tip_dist < self.min_tip_center_dist:
                continue
            if tip_dist > self.max_tip_center_dist:
                continue

            # 2) Mittelpunkt der Linie nicht zu nah am Zentrum
            #    Spider-Linien sitzen oft sehr zentrumsnah
            if mid_dist < self.min_mid_center_dist:
                continue

            # 3) Linie darf das Boardzentrum nicht schneiden / fast schneiden
            dist_line_to_center = self._point_line_distance(bc, p1, p2)
            if dist_line_to_center < self.min_line_center_dist:
                continue

            # 4) Der "Tail" sollte typischerweise weiter außen liegen als die Spitze
            if tail_dist <= tip_dist + 20:
                continue

            # 5) Zusätzliche Stabilisierung über ROI:
            #    beide Endpunkte sollten möglichst innerhalb / nahe ROI sein
            if roi_mask is not None:
                x1c = int(np.clip(x1, 0, w - 1))
                y1c = int(np.clip(y1, 0, h - 1))
                x2c = int(np.clip(x2, 0, w - 1))
                y2c = int(np.clip(y2, 0, h - 1))

                if roi_mask[y1c, x1c] == 0 and roi_mask[y2c, x2c] == 0:
                    continue

            # Confidence:
            # Länge gut, aber Zentrumsschnitt schlecht.
            center_bonus = 1.0 / (1.0 + (tip_dist / 260.0))
            anti_spider_bonus = min(1.5, dist_line_to_center / 35.0)
            tail_bonus = min(1.4, tail_dist / max(tip_dist, 1.0))

            confidence = length * (1.0 + 1.8 * center_bonus) * anti_spider_bonus * tail_bonus

            raw.append({
                "tip": (float(tip[0]), float(tip[1])),
                "confidence": float(confidence),
                "line": (int(x1), int(y1), int(x2), int(y2)),
                "length": float(length),
                "extra": {
                    "tip_dist": float(tip_dist),
                    "tail_dist": float(tail_dist),
                    "mid_dist": float(mid_dist),
                    "line_center_dist": float(dist_line_to_center),
                }
            })

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