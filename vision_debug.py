import cv2
import numpy as np
import os
import sys
import configparser


def get_external_path(filename):
    if getattr(sys, 'frozen', False):
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, filename)


class VisionDebugger:
    """
    Debugger für Entwicklung:
    - Full-Frame Overlay (Tip + optional Vector-Linie)
    - Warp/Board Overlay (Tip in Boardspace)
    Ein/Aus über vision_debug.ini

    Methoden-Farben:
    - abs   = rot
    - vec   = gelb
    - shape = magenta
    - fusion / kombiniert = cyan
    """

    def __init__(self, warp_size=800):
        self.enabled, self.warp_size, self.show_full, self.show_warp = self._read_ini(warp_size)
        self._windows_created = set()

    def _read_ini(self, fallback_warp_size):
        ini_path = get_external_path("vision_debug.ini")
        cfg = configparser.ConfigParser()

        enabled = True
        warp_size = fallback_warp_size
        show_full = True
        show_warp = True

        if os.path.exists(ini_path):
            cfg.read(ini_path, encoding="utf-8")
            enabled = int(cfg.get("vision", "debugging", fallback="1").strip()) == 1
            warp_size = int(cfg.get("vision", "warp_size", fallback=str(fallback_warp_size)).strip())
            show_full = int(cfg.get("vision", "show_full", fallback="1").strip()) == 1
            show_warp = int(cfg.get("vision", "show_warp", fallback="1").strip()) == 1

        return enabled, warp_size, show_full, show_warp

    def _ensure_windows(self, cam_id):
        if not self.enabled:
            return

        if self.show_full:
            name = f"Cam {cam_id} - FULL"
            if name not in self._windows_created:
                cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                self._windows_created.add(name)

        if self.show_warp:
            name = f"Cam {cam_id} - WARP"
            if name not in self._windows_created:
                cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                self._windows_created.add(name)

    def _get_method_color(self, method):
        """
        OpenCV nutzt BGR.
        """
        if method is None:
            return (255, 255, 255)  # weiß

        m = str(method).lower().strip()

        # kombinierte Methoden / Fusion
        if "+" in m or "fusion" in m:
            return (255, 255, 0)  # cyan

        if m == "abs":
            return (0, 0, 255)  # rot
        if m == "vec":
            return (0, 255, 255)  # gelb
        if m == "shape":
            return (255, 0, 255)  # magenta

        return (255, 255, 255)  # fallback weiß

    def _draw_tip_marker(self, img, x, y, color, inner_radius=8, outer_radius=16):
        cv2.circle(img, (int(x), int(y)), inner_radius, color, -1)
        cv2.circle(img, (int(x), int(y)), outer_radius, color, 2)

    def show(self, cam_id, frame_bgr, H_cam_to_board,
             tip_full=None, tip_board=None, line_full=None,
             method=None, conf=None, extra_lines=None):
        """
        cam_id: int
        frame_bgr: Full frame (z.B. 1920x1080)
        H_cam_to_board: Homography (Full -> 600x600 Boardspace)
        tip_full: (x,y) im Full Frame
        tip_board: (bx,by) im Boardspace (600x600)
        line_full: (x1,y1,x2,y2) optional (z.B. aus Vector)
        method: optional string ("abs", "vec", "shape", "abs+shape", ...)
        conf: optional float
        extra_lines: optional list[(x1,y1,x2,y2)] für mehrere Linien
        """
        if not self.enabled:
            return

        self._ensure_windows(cam_id)
        method_color = self._get_method_color(method)

        # ---------- FULL FRAME ----------
        if self.show_full:
            full = frame_bgr.copy()

            # optionale Hauptlinie (typisch Vector)
            if line_full is not None:
                x1, y1, x2, y2 = line_full
                cv2.line(full, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

            # optionale Zusatzlinien
            if extra_lines:
                for l in extra_lines:
                    if l is None:
                        continue
                    x1, y1, x2, y2 = l
                    cv2.line(full, (int(x1), int(y1)), (int(x2), int(y2)), (255, 200, 0), 1)

            # Tip markieren
            if tip_full is not None:
                tx, ty = tip_full
                self._draw_tip_marker(full, tx, ty, method_color, inner_radius=8, outer_radius=16)

            # HUD
            hud = f"Cam {cam_id}"
            if method:
                hud += f" | {method}"
            if conf is not None:
                hud += f" | conf={conf:.1f}"

            cv2.putText(
                full,
                hud,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (255, 255, 255),
                2
            )

            if tip_full is not None:
                cv2.putText(
                    full,
                    f"tip_full=({tip_full[0]:.1f},{tip_full[1]:.1f})",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (220, 220, 220),
                    2
                )

            cv2.imshow(f"Cam {cam_id} - FULL", full)

        # ---------- WARP / BOARD SPACE ----------
        if self.show_warp and H_cam_to_board is not None:
            warp = cv2.warpPerspective(frame_bgr, H_cam_to_board, (self.warp_size, self.warp_size))

            # Boardcenter markieren
            c = self.warp_size // 2
            cv2.circle(warp, (c, c), 4, (255, 255, 255), -1)

            # Tip in Boardspace (600) -> auf warp_size skalieren
            if tip_board is not None:
                bx, by = tip_board
                sx = int((bx / 600.0) * self.warp_size)
                sy = int((by / 600.0) * self.warp_size)

                self._draw_tip_marker(warp, sx, sy, method_color, inner_radius=6, outer_radius=14)

                cv2.putText(
                    warp,
                    f"tip_board=({bx:.1f},{by:.1f})",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2
                )

            # HUD im Warp
            warp_hud = f"Cam {cam_id}"
            if method:
                warp_hud += f" | {method}"
            if conf is not None:
                warp_hud += f" | conf={conf:.1f}"

            cv2.putText(
                warp,
                warp_hud,
                (20, self.warp_size - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                method_color,
                2
            )

            cv2.imshow(f"Cam {cam_id} - WARP", warp)

        cv2.waitKey(1)

    def close(self):
        if not self.enabled:
            return
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        self._windows_created.clear()
