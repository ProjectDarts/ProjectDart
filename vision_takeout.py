import cv2
import numpy as np

class TakeoutDetector:
    """
    Erkennt, ob das Board (im Board-Bereich) wieder 'clean' ist.
    - speichert clean_board_gray_full
    - prüft Differenz und warpt sie ins Board-Space, um nur dort zu zählen
    """

    def __init__(self, board_mask_600):
        self.board_mask_600 = board_mask_600
        self.clean_gray_full = None

        # Startwerte (tunen!)
        self.thresh = 35
        self.max_nonzero = 900  # unterhalb => leer

    def set_clean_board(self, frame_bgr):
        self.clean_gray_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    def check_takeout(self, frame_bgr, H_cam_to_board):
        if self.clean_gray_full is None or H_cam_to_board is None:
            return False

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(self.clean_gray_full, gray)
        diff = cv2.GaussianBlur(diff, (5, 5), 0)

        _, thr = cv2.threshold(diff, self.thresh, 255, cv2.THRESH_BINARY)

        # Nur Board-Bereich prüfen: thr -> Boardspace -> masken
        thr_board = cv2.warpPerspective(thr, H_cam_to_board, (600, 600))
        thr_board = cv2.bitwise_and(thr_board, self.board_mask_600)

        return cv2.countNonZero(thr_board) < self.max_nonzero
