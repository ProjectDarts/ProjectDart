import cv2
import numpy as np
import os
import sys
import configparser


def get_external_path(filename):
    if getattr(sys, "frozen", False):
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

    Unterstützt jetzt auch Texte wie:
    - "abs | Single 7"
    - "abs | Double 19"
    - "fusion | Triple 20"
    """

    def __init__(self, warp_size=800):
        (
            self.enabled,
            self.warp_size,
            self.show_full,
            self.show_warp,
            self.full_window_w,
            self.full_window_h,
            self.warp_window_w,
            self.warp_window_h,
            self.window_topmost,
        ) = self._read_ini(warp_size)

        self._windows_created = set()

    def _read_ini(self, fallback_warp_size):
        ini_path = get_external_path("vision_debug.ini")
        cfg = configparser.ConfigParser()

        enabled = True
        warp_size = fallback_warp_size
        show_full = True
        show_warp = True

        full_window_w = 1280
        full_window_h = 720
        warp_window_w = 1200
        warp_window_h = 1200

        window_topmost = False

        if os.path.exists(ini_path):
            cfg.read(ini_path, encoding="utf-8")
            enabled = int(cfg.get("vision", "debugging", fallback="1").strip()) == 1
            warp_size = int(cfg.get("vision", "warp_size", fallback=str(fallback_warp_size)).strip())
            show_full = int(cfg.get("vision", "show_full", fallback="1").strip()) == 1
            show_warp = int(cfg.get("vision", "show_warp", fallback="1").strip()) == 1

            full_window_w = int(cfg.get("vision", "full_window_w", fallback="1280").strip())
            full_window_h = int(cfg.get("vision", "full_window_h", fallback="720").strip())
            warp_window_w = int(cfg.get("vision", "warp_window_w", fallback="1200").strip())
            warp_window_h = int(cfg.get("vision", "warp_window_h", fallback="1200").strip())

            window_topmost = int(cfg.get("vision", "window_topmost", fallback="0").strip()) == 1

        return (
            enabled,
            warp_size,
            show_full,
            show_warp,
            full_window_w,
            full_window_h,
            warp_window_w,
            warp_window_h,
            window_topmost,
        )

    def _ensure_windows(self, cam_id):
        if not self.enabled:
            return

        if self.show_full:
            name = f"Cam {cam_id} - FULL"
            if name not in self._windows_created:
                cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(name, self.full_window_w, self.full_window_h)
                if self.window_topmost:
                    try:
                        cv2.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 1)
                    except Exception:
                        pass
                self._windows_created.add(name)

        if self.show_warp:
            name = f"Cam {cam_id} - WARP"
            if name not in self._windows_created:
                cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(name, self.warp_window_w, self.warp_window_h)
                if self.window_topmost:
                    try:
                        cv2.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 1)
                    except Exception:
                        pass
                self._windows_created.add(name)

    def _normalize_method_key(self, method):
        if method is None:
            return None

        m = str(method).lower().strip()

        # Nur den linken Teil vor "|" als Methoden-Key werten
        # z.B. "abs | double 19" -> "abs"
        if "|" in m:
            m = m.split("|", 1)[0].strip()

        return m

    def _extract_field_label(self, method):
        if method is None:
            return None

        m = str(method).strip()
        if "|" in m:
            return m.split("|", 1)[1].strip()

        return None

    def _get_method_color(self, method):
        key = self._normalize_method_key(method)

        if key is None:
            return (255, 255, 255)

        if "+" in key or "fusion" in key or "combined" in key or "kombi" in key:
            return (255, 255, 0)  # cyan
        if key == "abs":
            return (0, 0, 255)    # rot
        if key == "vec":
            return (0, 255, 255)  # gelb
        if key == "shape":
            return (255, 0, 255)  # magenta

        return (255, 255, 255)

    def _draw_tip_marker(self, img, x, y, color, inner_radius=8, outer_radius=16):
        cv2.circle(img, (int(x), int(y)), inner_radius, color, -1)
        cv2.circle(img, (int(x), int(y)), outer_radius, color, 2)

    def _fit_to_canvas(self, img, canvas_w, canvas_h, pad=20):
        """
        Bild proportional in feste Zielgröße einpassen und mittig platzieren.
        """
        h, w = img.shape[:2]

        usable_w = max(1, canvas_w - 2 * pad)
        usable_h = max(1, canvas_h - 2 * pad)

        scale = min(usable_w / w, usable_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        x_off = (canvas_w - new_w) // 2
        y_off = (canvas_h - new_h) // 2

        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        return canvas

    def show(
        self,
        cam_id,
        frame_bgr,
        H_cam_to_board,
        tip_full=None,
        tip_board=None,
        line_full=None,
        method=None,
        conf=None,
        extra_lines=None,
    ):
        if not self.enabled:
            return

        self._ensure_windows(cam_id)
        method_color = self._get_method_color(method)
        field_label = self._extract_field_label(method)

        # ---------- FULL FRAME ----------
        if self.show_full:
            full = frame_bgr.copy()

            if line_full is not None:
                x1, y1, x2, y2 = line_full
                cv2.line(full, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

            if extra_lines:
                for l in extra_lines:
                    if l is None:
                        continue
                    x1, y1, x2, y2 = l
                    cv2.line(full, (int(x1), int(y1)), (int(x2), int(y2)), (255, 200, 0), 1)

            if tip_full is not None:
                tx, ty = tip_full
                self._draw_tip_marker(full, tx, ty, method_color, inner_radius=8, outer_radius=16)

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
                1.0,
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

            if field_label:
                cv2.putText(
                    full,
                    f"Field: {field_label}",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    method_color,
                    2
                )

            full_display = self._fit_to_canvas(
                full,
                self.full_window_w,
                self.full_window_h,
                pad=20
            )

            cv2.imshow(f"Cam {cam_id} - FULL", full_display)

        # ---------- WARP / BOARD SPACE ----------
        if self.show_warp and H_cam_to_board is not None:
            warp = cv2.warpPerspective(frame_bgr, H_cam_to_board, (self.warp_size, self.warp_size))

            c = self.warp_size // 2
            cv2.circle(warp, (c, c), 4, (255, 255, 255), -1)

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

            # Große Feldanzeige oben
            if field_label:
                cv2.putText(
                    warp,
                    field_label,
                    (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    method_color,
                    3
                )

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

            warp_display = self._fit_to_canvas(
                warp,
                self.warp_window_w,
                self.warp_window_h,
                pad=40
            )

            cv2.imshow(f"Cam {cam_id} - WARP", warp_display)

        cv2.waitKey(1)

    def close(self):
        if not self.enabled:
            return
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        self._windows_created.clear()