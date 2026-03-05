import cv2
import numpy as np

class AbsDiffTipDetector:
    """
    Detects a dart tip in FULL FRAME using absdiff to a reference frame.
    Returns candidates with tip (x,y) in FULL FRAME coordinates.
    """

    def __init__(self):
        self.reference_gray = None

        # Tuning
        self.min_area = 120
        self.max_area = 25000
        self.min_length = 18
        self.blur = (5, 5)

    def set_reference(self, frame_bgr):
        self.reference_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    def detect(self, frame_bgr):
        if self.reference_gray is None:
            return [], frame_bgr

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(self.reference_gray, gray)

        # Slight blur to reduce sensor noise
        diff_blur = cv2.GaussianBlur(diff, self.blur, 0)

        # OTSU threshold
        _, thr = cv2.threshold(diff_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morph close helps connect fragmented dart shapes
        kernel = np.ones((3, 3), np.uint8)
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = gray.shape[:2]
        board_center_full = np.array([w // 2, h // 2], dtype=np.float32)  # will be replaced by projected center externally (optional)

        debug = frame_bgr.copy()
        objects = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue
            if len(cnt) < 5:
                continue

            pts = cnt.reshape(-1, 2).astype(np.float32)

            mean, eigenvecs = cv2.PCACompute(pts, mean=None)
            center = mean[0]
            axis = eigenvecs[0]  # principal axis direction

            # Two directions. Choose direction that points "towards board center"
            # If dot(axis, (board_center - center)) < 0 -> invert
            if np.dot(axis, board_center_full - center) < 0:
                axis = -axis

            proj = np.dot(pts - center, axis)
            p_min = pts[np.argmin(proj)]
            p_max = pts[np.argmax(proj)]
            length = float(np.linalg.norm(p_max - p_min))
            if length < self.min_length:
                continue

            # Tip is the farthest point along axis towards board
            idxs = np.argsort(proj)[-3:]
            tip = np.mean(pts[idxs], axis=0)

            # Slenderness confidence
            width = max(area / max(length, 1e-6), 1.0)
            slenderness = length / width
            confidence = length * slenderness * np.log(area + 1)

            objects.append({
                "tip": (float(tip[0]), float(tip[1])),
                "area": float(area),
                "length": length,
                "confidence": float(confidence),
                "contour": cnt
            })

        # Draw debug (optional)
        for obj in objects:
            cv2.drawContours(debug, [obj["contour"]], 0, (0, 255, 0), 1)
            tx, ty = int(obj["tip"][0]), int(obj["tip"][1])
            cv2.circle(debug, (tx, ty), 6, (0, 0, 255), -1)

        # Sort descending by confidence
        objects.sort(key=lambda o: o["confidence"], reverse=True)

        return objects, debug, thr
