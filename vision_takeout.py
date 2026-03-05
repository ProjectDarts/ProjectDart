import cv2
import numpy as np

class TakeoutDetector:
    """
    Idee:
    - clean_board_gray = Referenz ohne Pfeile
    - "empty" heißt: im extended ROI ist sehr wenig Diff
    - wenn wir vorher einen Dart hatten und jetzt empty -> Takeout
    """

    def __init__(self, board_mask, extended_mask=None):
        self.board_mask = board_mask
        self.extended_mask = extended_mask if extended_mask is not None else board_mask
        self.clean_board_gray = None

        # Tuning
        self.thresh = 35
        self.empty_pixels_threshold = 350  # je nach Licht ggf. 200..800

    def set_clean_board(self, frame_bgr):
        self.clean_board_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    def is_empty(self, frame_bgr, gray=None):
        if self.clean_board_gray is None:
            return False, frame_bgr

        if gray is None:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(self.clean_board_gray, gray)
        diff = cv2.GaussianBlur(diff, (5, 5), 0)

        _, thr = cv2.threshold(diff, self.thresh, 255, cv2.THRESH_BINARY)
        thr = cv2.bitwise_and(thr, self.extended_mask)

        cnt = cv2.countNonZero(thr)
        debug = frame_bgr.copy()
        # Optional: Debug-Overlay
        cv2.putText(debug, f"EMPTY_PIX={cnt}", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        return (cnt < self.empty_pixels_threshold), debug
