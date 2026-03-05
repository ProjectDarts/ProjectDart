import cv2
import numpy as np

class VisionDebugger:
    def __init__(self, warp_size=800):
        self.warp_size = int(warp_size)
        self.windows_ready = False

    def _ensure_windows(self, cam_ids):
        if self.windows_ready:
            return
        for cam_id in cam_ids:
            cv2.namedWindow(f"VISION DEBUG Cam {cam_id}", cv2.WINDOW_NORMAL)
        self.windows_ready = True

    def show(self, cam_id, frame_bgr, H_cam_to_board, tip_full=None, tip_board=None):
        self._ensure_windows([cam_id])

        view = frame_bgr.copy()

        # Warp nur für Debug-Ansicht
        if H_cam_to_board is not None:
            warp = cv2.warpPerspective(frame_bgr, H_cam_to_board, (self.warp_size, self.warp_size))
        else:
            warp = None

        # Overlays im Full-Frame
        if tip_full is not None:
            x, y = int(tip_full[0]), int(tip_full[1])
            cv2.circle(view, (x, y), 10, (0, 0, 255), 3)
            cv2.putText(view, "TIP(full)", (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # Overlays im Warp
        if warp is not None and tip_board is not None:
            # tip_board kommt in 600x600; auf warp_size skalieren
            sx = self.warp_size / 600.0
            xw = int(tip_board[0] * sx)
            yw = int(tip_board[1] * sx)
            cv2.circle(warp, (xw, yw), 10, (0, 0, 255), 3)
            cv2.putText(warp, "TIP(board)", (xw+10, yw-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # Anzeige: Wir zeigen lieber Warp (leicht fürs Auge). Falls warp None, show full.
        out = warp if warp is not None else view
        cv2.imshow(f"VISION DEBUG Cam {cam_id}", out)
        cv2.waitKey(1)

    def close(self):
        try:
            cv2.destroyAllWindows()
        except:
            pass
