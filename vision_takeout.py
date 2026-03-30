import cv2
import numpy as np


class TakeoutDetector:
    """
    Takeout-Erkennung im Boardspace (warped):
    - clean_board = graues Referenzbild "Board ohne Pfeile"
    - check_takeout(warped_bgr, last_hit_contours) -> (is_empty, debug_img)

      is_empty=True  => Board ist stabil leer
      is_empty=False => Es steckt vermutlich noch etwas im Board
    """

    def __init__(self, board_mask):
        self.board_mask = board_mask
        self.clean_board = None  # gray

        # Robuster gegen Rauschen
        self.thr = 22
        self.min_nonzero = 220
        self.min_cnt_area = 120

        kernel = np.ones((5, 5), np.uint8)
        self.inner_board_mask = cv2.erode(self.board_mask, kernel, iterations=1)

    def _prepare_gray(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        return gray

    def set_clean_board(self, frame_bgr):
        self.clean_board = self._prepare_gray(frame_bgr)

    def check_takeout(self, warped_frame_bgr, last_hit_contours):
        if self.clean_board is None:
            return False, warped_frame_bgr

        gray = self._prepare_gray(warped_frame_bgr)

        diff = cv2.absdiff(self.clean_board, gray)
        diff = cv2.bitwise_and(diff, self.inner_board_mask)

        _, thr = cv2.threshold(diff, self.thr, 255, cv2.THRESH_BINARY)

        kernel_open = np.ones((3, 3), np.uint8)
        kernel_close = np.ones((5, 5), np.uint8)

        thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel_open, iterations=1)
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        thr = cv2.bitwise_and(thr, self.inner_board_mask)

        nonzero = cv2.countNonZero(thr)
        contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        large_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.min_cnt_area:
                large_contours.append(cnt)

        debug_img = warped_frame_bgr.copy()
        if large_contours:
            cv2.drawContours(debug_img, large_contours, -1, (0, 0, 255), 1)

        # Wenn zuvor ein echter Treffer existierte, sollen wir vorsichtiger sein:
        # Schon moderate Restkonturen bedeuten dann eher "noch nicht leer".
        if last_hit_contours:
            is_empty = (nonzero < self.min_nonzero) and (len(large_contours) == 0)
        else:
            # Ohne bekannte letzte Dart-Kontur etwas lockerer.
            is_empty = (nonzero < (self.min_nonzero * 0.75)) and (len(large_contours) == 0)

        return is_empty, debug_img