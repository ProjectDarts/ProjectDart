import cv2
import numpy as np


class TakeoutDetector:
    """
    Erkennt, ob Darts entfernt wurden (Board wieder "leer"):
    - Referenz: clean_board (warped gray)
    - Vergleich: absdiff + threshold
    - Optional: wenn last_hit_contours vorhanden, reicht es, im Board generell Diff zu sehen
      (kann später per ROI um Konturen herum verfeinert werden)
    """

    def __init__(self, board_mask):
        self.board_mask = board_mask
        self.clean_board = None  # gray warped

        # Tuning
        self.thr_val = 40
        self.min_area = 300
        self.global_nonzero_limit = 1200

    def set_clean_board(self, warped_bgr):
        self.clean_board = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)

    def check_takeout(self, warped_bgr, last_hit_contours):
        if self.clean_board is None:
            return False, warped_bgr

        gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(self.clean_board, gray)
        diff = cv2.GaussianBlur(diff, (5, 5), 0)

        _, thr = cv2.threshold(diff, self.thr_val, 255, cv2.THRESH_BINARY)
        thr = cv2.bitwise_and(thr, self.board_mask)

        debug = warped_bgr.copy()

        contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Standard: "leer" = keine ausreichende Veränderung mehr sichtbar
        still_something = False

        if last_hit_contours:
            # Sobald irgendein signifikanter Bereich noch Differenz hat, gilt: noch nicht leer
            for cnt in contours:
                if cv2.contourArea(cnt) > self.min_area:
                    still_something = True
                    break
        else:
            # Wenn eigentlich keine Darts stecken sollten, aber trotzdem viel Diff: nicht leer
            if cv2.countNonZero(thr) > self.global_nonzero_limit:
                still_something = True

        if still_something:
            cv2.drawContours(debug, contours, -1, (0, 0, 255), 1)

        takeout_detected = not still_something
        return takeout_detected, debug
