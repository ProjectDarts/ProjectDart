import cv2
import numpy as np


class VectorDetector:
    """
    vision_vector.py (Boardspace 600x600)

    Ziel:
    - Line/Vektor-basierte Darterkennung im *gewarpten* Boardspace (typisch 600x600)
    - Canny -> HoughLinesP -> Kandidaten
    - Tip = Linien-Endpunkt, der Richtung Boardzentrum zeigt
    - Liefert kompatible Objekte für vision.py:

      [{
        "tip_board": (x, y),
        "confidence": float,
        "contour": None,
        "extra": {
            "line_warp": (x1, y1, x2, y2),
            "length": float,
            "angle_deg": float,
        }
      }, ...]
    """

    def __init__(self, board_mask: np.ndarray):
        self.board_mask = board_mask

        # --- Preprocessing ---
        self.blur = (5, 5)
        self.canny1 = 60
        self.canny2 = 160

        # --- Hough ---
        self.hough_threshold = 35
        self.min_line_len = 28
        self.max_line_gap = 6

        # --- Candidate filtering / merging ---
        self.max_candidates = 8
        self.tip_merge_dist = 18

        # --- Optional: score weighting ---
        self.center_pull = 300.0  # Skalierung für center bonus

        # Reference (optional, falls du später mal diff-basierte Edge-Stabilisierung willst)
        self.reference_gray = None

    def set_reference(self, warped_frame_bgr):
        """Optional: Speichere eine Referenz. Aktuell nicht zwingend genutzt."""
        self.reference_gray = cv2.cvtColor(warped_frame_bgr, cv2.COLOR_BGR2GRAY)

    def detect(self, warped_frame_bgr, gray=None):
        """
        warped_frame_bgr: Boardspace-Frame (z.B. 600x600)
        gray: optional vor-konvertiert, spart CPU
        """
        h, w = warped_frame_bgr.shape[:2]
        board_center = np.array([w / 2.0, h / 2.0], dtype=np.float32)

        if gray is None:
            gray = cv2.cvtColor(warped_frame_bgr, cv2.COLOR_BGR2GRAY)

        g = cv2.GaussianBlur(gray, self.blur, 0)
        edges = cv2.Canny(g, self.canny1, self.canny2)

        # Kein harter Crop: nur Maske anwenden (dämpft „außerhalb Board“)
        if self.board_mask is not None:
            edges = cv2.bitwise_and(edges, self.board_mask)

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

            # Tip: Endpunkt wählen, der näher am Boardzentrum ist
            d1 = float(np.linalg.norm(board_center - p1))
            d2 = float(np.linalg.norm(board_center - p2))
            tip = p1 if d1 < d2 else p2

            # sehr grobe Plausibilität: Tip sollte „im Boardmaskenbereich“ liegen
            tx, ty = int(round(tip[0])), int(round(tip[1]))
            if tx < 0 or tx >= w or ty < 0 or ty >= h:
                continue
            if self.board_mask is not None and self.board_mask[ty, tx] == 0:
                # Tip außerhalb Board-ROI -> kann trotzdem Dart sein, aber meist Noise
                # Wenn du Missed erkennen willst, ist das später in vision.py Score-Logik.
                # Hier lassen wir es zu, aber bestrafen confidence.
                tip_outside_penalty = 0.35
            else:
                tip_outside_penalty = 1.0

            # Angle (nur für Debug)
            ang = float(np.degrees(np.arctan2((p2 - p1)[1], (p2 - p1)[0])))

            # Confidence: Länge + Bonus wenn Tip näher am Zentrum
            dist_to_center = float(np.linalg.norm(board_center - tip))
            center_bonus = 1.0 / (1.0 + (dist_to_center / self.center_pull))
            confidence = length * (1.3 + 2.7 * center_bonus) * tip_outside_penalty

            raw.append({
                "tip_board": (float(tip[0]), float(tip[1])),
                "confidence": float(confidence),
                "contour": None,
                "extra": {
                    "line_warp": (int(x1), int(y1), int(x2), int(y2)),
                    "length": float(length),
                    "angle_deg": ang,
                    "dist_to_center": dist_to_center
                }
            })

        if not raw:
            return []

        # Sortieren
        raw.sort(key=lambda o: o["confidence"], reverse=True)

        # Tips mergen (Doppellinien)
        merged = []
        for obj in raw:
            keep = True
            for m in merged:
                if np.linalg.norm(np.array(obj["tip_board"]) - np.array(m["tip_board"])) < self.tip_merge_dist:
                    # behalte den besseren
                    if obj["confidence"] > m["confidence"]:
                        m.update(obj)
                    keep = False
                    break
            if keep:
                merged.append(obj)
            if len(merged) >= self.max_candidates:
                break

        return merged
