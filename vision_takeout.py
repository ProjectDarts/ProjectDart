import cv2
import numpy as np


class TakeoutDetector:
    """
    Takeout-Erkennung im Boardspace (warped):
    - clean_board = graues Referenzbild "Board ohne Pfeile"
    - check_takeout(warped_bgr, last_hit_contours) -> (is_empty, debug_img)
      is_empty=True  => Board ist sauber/leer (Takeout passiert)
      is_empty=False => es steckt noch was / Bewegung im Boardbereich
    """

    def __init__(self, board_mask):
        self.board_mask = board_mask
        self.clean_board = None  # gray

        # Tuning
        self.thr = 40
        self.min_nonzero = 1000
        self.min_cnt_area = 300

    def set_clean_board(self, frame_bgr):
        self.clean_board = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    def check_takeout(self, warped_frame_bgr, last_hit_contours):
        if self.clean_board is None:
            return False, warped_frame_bgr

        gray = cv2.cvtColor(warped_frame_bgr, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(self.clean_board, gray)
        diff = cv2.GaussianBlur(diff, (5, 5), 0)

        _, thr = cv2.threshold(diff, self.thr, 255, cv2.THRESH_BINARY)
        thr = cv2.bitwise_and(thr, self.board_mask)

        debug_img = warped_frame_bgr.copy()

        # Default: leer (true) – und wir widerlegen es bei Befund
        is_empty = True

        if last_hit_contours:
            contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) > self.min_cnt_area:
                    is_empty = False
                    break

            if not is_empty:
                cv2.drawContours(debug_img, contours, -1, (0, 0, 255), 1)

        else:
            # Wenn "eigentlich leer", sollte im Threshold kaum was sein
            if cv2.countNonZero(thr) > self.min_nonzero:
                is_empty = False

        return is_empty, debug_img
