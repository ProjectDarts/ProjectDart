import cv2
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# =========================================================
# KONSTANTEN
# =========================================================
WINDOW_NAME = "Kalibrierung - 1080p"

FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

MAX_POINTS = 4
ZOOM_FACTOR = 3
ZOOM_SIZE = 110

CAMERA_WARMUP_SECONDS = 1.5
DISCARD_INITIAL_FRAMES = 10

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Farben (BGR)
COLOR_WHITE = (245, 245, 245)
COLOR_LIGHT = (220, 220, 220)
COLOR_MUTED = (170, 170, 170)
COLOR_YELLOW = (80, 220, 255)
COLOR_GREEN = (80, 220, 120)
COLOR_RED = (60, 80, 255)
COLOR_BLUE = (255, 170, 70)
COLOR_DARK = (25, 25, 25)
COLOR_PANEL = (35, 35, 35)
COLOR_PANEL_2 = (45, 45, 45)
COLOR_BORDER = (90, 90, 90)
COLOR_CROSSHAIR = (0, 0, 255)

PANEL_ALPHA = 0.72
PANEL_RADIUS = 14  # optisch, wird hier über normale Rechtecke angenähert

STATUS_MESSAGE_DURATION = 2.0


# =========================================================
# PFAD-LOGIK
# =========================================================
def get_config_path(filename: str) -> str:
    """
    Liefert einen stabilen Pfad für Konfigurationsdateien:
    - Python-Skript: Verzeichnis der .py-Datei
    - PyInstaller-EXE: Verzeichnis der EXE
    """
    if getattr(sys, "frozen", False):
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(base_path, filename)


# =========================================================
# DATENMODELLE
# =========================================================
@dataclass
class CalibrationState:
    points: List[List[int]] = field(default_factory=list)
    mouse_x: int = FRAME_WIDTH // 2
    mouse_y: int = FRAME_HEIGHT // 2
    is_dragging: bool = False

    def reset(self) -> None:
        self.points.clear()
        self.mouse_x = FRAME_WIDTH // 2
        self.mouse_y = FRAME_HEIGHT // 2
        self.is_dragging = False

    @property
    def point_count(self) -> int:
        return len(self.points)

    @property
    def is_complete(self) -> bool:
        return len(self.points) == MAX_POINTS

    def remove_last_point(self) -> bool:
        if not self.points:
            return False
        self.points.pop()
        return True


@dataclass
class StatusMessage:
    text: str = ""
    color: Tuple[int, int, int] = COLOR_LIGHT
    until: float = 0.0

    def set(self, text: str, color: Tuple[int, int, int], duration: float = STATUS_MESSAGE_DURATION) -> None:
        self.text = text
        self.color = color
        self.until = time.time() + duration

    def clear(self) -> None:
        self.text = ""
        self.until = 0.0

    @property
    def is_active(self) -> bool:
        return bool(self.text) and time.time() < self.until


# =========================================================
# KALIBRIERER
# =========================================================
class Calibrator:
    POINTS_DESC = [
        "Oben (20/1)",
        "Rechts (6/10)",
        "Unten (3/19)",
        "Links (11/14)",
    ]

    def __init__(self, cam_ids: Optional[List[int]] = None) -> None:
        self.cam_ids = cam_ids if cam_ids is not None else [0, 1, 2]
        self.current_cam_idx = 0
        self.cap: Optional[cv2.VideoCapture] = None
        self.state = CalibrationState()
        self.status = StatusMessage()

    # =====================================================
    # KAMERA-VERWALTUNG
    # =====================================================
    def release_camera(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def open_camera(self, cam_id: int) -> bool:
        self.release_camera()

        print(f"[INFO] Initialisiere Kamera ID: {cam_id}...")

        cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"[ERROR] Kamera ID {cam_id} konnte nicht geöffnet werden.")
            return False

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        time.sleep(CAMERA_WARMUP_SECONDS)

        for _ in range(DISCARD_INITIAL_FRAMES):
            cap.read()

        self.cap = cap
        self.state.reset()
        self.status.set(f"Kamera {cam_id} bereit", COLOR_GREEN)
        return True

    def setup_current_camera(self) -> bool:
        while self.current_cam_idx < len(self.cam_ids):
            cam_id = self.cam_ids[self.current_cam_idx]
            if self.open_camera(cam_id):
                return True
            self.current_cam_idx += 1

        print("[INFO] Alle Kameras kalibriert!")
        return False

    def move_to_next_camera(self) -> bool:
        self.current_cam_idx += 1
        return self.setup_current_camera()

    def read_frame(self) -> Optional[np.ndarray]:
        if self.cap is None:
            return None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None

        return frame

    # =====================================================
    # PERSISTENZ
    # =====================================================
    def save_current_points(self) -> bool:
        cam_id = self.cam_ids[self.current_cam_idx]
        filename = get_config_path(f"cam{cam_id}_config.json")

        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump({"points": self.state.points}, f, indent=2)

            print(f"[INFO] Erfolg: {filename} gespeichert.")
            self.status.set(f"Gespeichert: cam{cam_id}_config.json", COLOR_GREEN)
            return True

        except Exception as e:
            print(f"[ERROR] Speichern fehlgeschlagen: {e}")
            self.status.set("Speichern fehlgeschlagen", COLOR_RED)
            return False

    # =====================================================
    # STATE / EINGABE
    # =====================================================
    def get_current_point_description(self) -> str:
        if self.state.is_complete:
            return "Fertig"
        return self.POINTS_DESC[self.state.point_count]

    def on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        self.state.mouse_x = x
        self.state.mouse_y = y

        if event == cv2.EVENT_LBUTTONDOWN and not self.state.is_complete:
            self.state.is_dragging = True

        elif event == cv2.EVENT_LBUTTONUP and self.state.is_dragging:
            self.state.is_dragging = False

            if not self.state.is_complete:
                self.state.points.append([x, y])
                idx = self.state.point_count - 1
                print(
                    f"Punkt {idx + 1} "
                    f"({self.POINTS_DESC[idx]}) "
                    f"gesetzt: {x}, {y}"
                )
                self.status.set(
                    f"Punkt {idx + 1}/4 gesetzt: {self.POINTS_DESC[idx]}",
                    COLOR_GREEN,
                    duration=1.8,
                )

    # =====================================================
    # UI HELFER
    # =====================================================
    @staticmethod
    def overlay_rect(
        img: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: Tuple[int, int, int],
        alpha: float = PANEL_ALPHA,
        border_color: Optional[Tuple[int, int, int]] = None,
        border_thickness: int = 1,
    ) -> None:
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        if border_color is not None and border_thickness > 0:
            cv2.rectangle(img, (x1, y1), (x2, y2), border_color, border_thickness)

    @staticmethod
    def draw_text(
        img: np.ndarray,
        text: str,
        pos: Tuple[int, int],
        scale: float,
        color: Tuple[int, int, int],
        thickness: int = 1,
    ) -> None:
        cv2.putText(img, text, pos, FONT, scale, color, thickness, cv2.LINE_AA)

    @staticmethod
    def draw_crosshair(img: np.ndarray, x: int, y: int, size: int = 18, color: Tuple[int, int, int] = COLOR_CROSSHAIR) -> None:
        cv2.line(img, (x - size, y), (x + size, y), color, 1, cv2.LINE_AA)
        cv2.line(img, (x, y - size), (x, y + size), color, 1, cv2.LINE_AA)
        cv2.circle(img, (x, y), 4, color, 1, cv2.LINE_AA)

    def draw_progress_bar(self, img: np.ndarray, x: int, y: int, w: int, h: int) -> None:
        self.overlay_rect(img, x, y, x + w, y + h, COLOR_PANEL_2, alpha=0.85, border_color=COLOR_BORDER)

        progress = self.state.point_count / MAX_POINTS
        fill_w = int((w - 4) * progress)

        if fill_w > 0:
            cv2.rectangle(img, (x + 2, y + 2), (x + 2 + fill_w, y + h - 2), COLOR_GREEN, -1)

        self.draw_text(
            img,
            f"{self.state.point_count} / {MAX_POINTS} Punkte",
            (x + 10, y + h - 10),
            0.65,
            COLOR_WHITE,
            2,
        )

    # =====================================================
    # RENDERING
    # =====================================================
    def draw_main_info_panel(self, display_frame: np.ndarray, cam_id: int) -> None:
        x1, y1, x2, y2 = 20, 20, 640, 235
        self.overlay_rect(display_frame, x1, y1, x2, y2, COLOR_PANEL, alpha=PANEL_ALPHA, border_color=COLOR_BORDER)

        self.draw_text(display_frame, f"Kamera {cam_id}", (40, 58), 1.0, COLOR_WHITE, 2)
        self.draw_text(display_frame, f"Aufloesung: {FRAME_WIDTH}x{FRAME_HEIGHT}", (40, 90), 0.65, COLOR_MUTED, 1)

        if self.state.is_complete:
            target_text = "Alle 4 Punkte gesetzt - bereit zum Speichern"
            target_color = COLOR_GREEN
        else:
            target_text = f"Naechster Punkt: {self.get_current_point_description()}"
            target_color = COLOR_YELLOW

        self.draw_text(display_frame, target_text, (40, 132), 0.8, target_color, 2)

        self.draw_progress_bar(display_frame, 40, 152, 300, 26)

        self.draw_text(display_frame, "Punktreihenfolge:", (370, 172), 0.65, COLOR_LIGHT, 1)

        for i, desc in enumerate(self.POINTS_DESC):
            yy = 198 + i * 28
            done = i < self.state.point_count
            is_current = i == self.state.point_count and not self.state.is_complete

            marker = "✓" if done else ">" if is_current else "o"
            color = COLOR_GREEN if done else COLOR_YELLOW if is_current else COLOR_MUTED

            self.draw_text(display_frame, f"{marker} {i+1}. {desc}", (370, yy), 0.62, color, 2 if done or is_current else 1)

    def draw_help_panel(self, display_frame: np.ndarray) -> None:
        x1, y1, x2, y2 = 20, FRAME_HEIGHT - 145, 760, FRAME_HEIGHT - 20
        self.overlay_rect(display_frame, x1, y1, x2, y2, COLOR_PANEL, alpha=PANEL_ALPHA, border_color=COLOR_BORDER)

        line1 = "Linksklick = Punkt setzen    Backspace / X = letzter Punkt loeschen    R = Reset"
        line2 = "Leertaste = Speichern    N = Kamera ueberspringen    Q / ESC = Abbrechen"

        self.draw_text(display_frame, line1, (40, FRAME_HEIGHT - 92), 0.68, COLOR_WHITE, 1)
        self.draw_text(display_frame, line2, (40, FRAME_HEIGHT - 52), 0.68, COLOR_WHITE, 1)

    def draw_status_bar(self, display_frame: np.ndarray) -> None:
        x1, y1, x2, y2 = FRAME_WIDTH - 520, FRAME_HEIGHT - 80, FRAME_WIDTH - 20, FRAME_HEIGHT - 20
        self.overlay_rect(display_frame, x1, y1, x2, y2, COLOR_PANEL_2, alpha=0.82, border_color=COLOR_BORDER)

        if self.status.is_active:
            text = self.status.text
            color = self.status.color
        else:
            if self.state.is_complete:
                text = "Bereit zum Speichern"
                color = COLOR_GREEN
            else:
                text = "Kalibrierung aktiv"
                color = COLOR_LIGHT

        self.draw_text(display_frame, text, (x1 + 18, y1 + 40), 0.72, color, 2)

    def draw_points(self, display_frame: np.ndarray) -> None:
        for i, point in enumerate(self.state.points):
            px, py = point

            cv2.circle(display_frame, (px, py), 11, COLOR_DARK, -1, cv2.LINE_AA)
            cv2.circle(display_frame, (px, py), 8, COLOR_GREEN, -1, cv2.LINE_AA)
            cv2.circle(display_frame, (px, py), 14, COLOR_WHITE, 1, cv2.LINE_AA)

            label = str(i + 1)
            self.draw_text(display_frame, label, (px + 18, py - 14), 0.75, COLOR_GREEN, 2)

    def draw_live_crosshair(self, display_frame: np.ndarray) -> None:
        self.draw_crosshair(display_frame, self.state.mouse_x, self.state.mouse_y, size=16, color=COLOR_RED)

    def draw_zoom(self, frame: np.ndarray, display_frame: np.ndarray) -> None:
        if not (self.state.is_dragging or self.state.point_count < MAX_POINTS):
            return

        half_size = ZOOM_SIZE // 2

        x1 = max(0, self.state.mouse_x - half_size)
        y1 = max(0, self.state.mouse_y - half_size)
        x2 = min(FRAME_WIDTH, self.state.mouse_x + half_size)
        y2 = min(FRAME_HEIGHT, self.state.mouse_y + half_size)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return

        zoom_img = cv2.resize(
            roi,
            (ZOOM_SIZE * ZOOM_FACTOR, ZOOM_SIZE * ZOOM_FACTOR),
            interpolation=cv2.INTER_NEAREST,
        )

        h_z, w_z = zoom_img.shape[:2]

        outer_margin = 20
        panel_pad = 12

        panel_x1 = FRAME_WIDTH - w_z - 2 * panel_pad - outer_margin
        panel_y1 = 20
        panel_x2 = FRAME_WIDTH - outer_margin
        panel_y2 = panel_y1 + h_z + 52

        self.overlay_rect(
            display_frame,
            panel_x1,
            panel_y1,
            panel_x2,
            panel_y2,
            COLOR_PANEL,
            alpha=0.82,
            border_color=COLOR_BORDER,
        )

        self.draw_text(display_frame, "Lupe", (panel_x1 + 14, panel_y1 + 26), 0.72, COLOR_WHITE, 2)

        img_x1 = panel_x1 + panel_pad
        img_y1 = panel_y1 + 38
        img_x2 = img_x1 + w_z
        img_y2 = img_y1 + h_z

        display_frame[img_y1:img_y2, img_x1:img_x2] = zoom_img
        cv2.rectangle(display_frame, (img_x1, img_y1), (img_x2, img_y2), COLOR_WHITE, 1)

        center_x = img_x1 + w_z // 2
        center_y = img_y1 + h_z // 2
        cv2.line(display_frame, (center_x, img_y1), (center_x, img_y2), COLOR_CROSSHAIR, 1, cv2.LINE_AA)
        cv2.line(display_frame, (img_x1, center_y), (img_x2, center_y), COLOR_CROSSHAIR, 1, cv2.LINE_AA)

    def render_frame(self, frame: np.ndarray, cam_id: int) -> np.ndarray:
        display_frame = frame.copy()

        self.draw_main_info_panel(display_frame, cam_id)
        self.draw_help_panel(display_frame)
        self.draw_status_bar(display_frame)
        self.draw_points(display_frame)
        self.draw_live_crosshair(display_frame)
        self.draw_zoom(frame, display_frame)

        return display_frame

    # =====================================================
    # TASTATUR-LOGIK
    # =====================================================
    def handle_key(self, key: int) -> bool:
        if key == ord(" ") and self.state.is_complete:
            self.save_current_points()
            return self.move_to_next_camera()

        if key == ord(" ") and not self.state.is_complete:
            self.status.set("Zum Speichern muessen 4 Punkte gesetzt sein", COLOR_YELLOW)
            return True

        if key in (8, 127, ord("x"), ord("X")):
            if self.state.remove_last_point():
                self.status.set("Letzter Punkt entfernt", COLOR_YELLOW, duration=1.4)
            else:
                self.status.set("Keine Punkte zum Entfernen", COLOR_MUTED, duration=1.2)
            return True

        if key in (ord("r"), ord("R")):
            self.state.reset()
            self.status.set("Aktuelle Kamera zurueckgesetzt", COLOR_YELLOW)
            return True

        if key in (ord("n"), ord("N")):
            current_cam = self.cam_ids[self.current_cam_idx]
            print(f"[INFO] Kamera {current_cam} uebersprungen.")
            self.status.set(f"Kamera {current_cam} uebersprungen", COLOR_BLUE)
            return self.move_to_next_camera()

        if key == ord("q") or key == 27:
            print("[INFO] Kalibrierung abgebrochen.")
            return False

        return True

    # =====================================================
    # HAUPTLAUF
    # =====================================================
    def run(self) -> None:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)

        if not self.setup_current_camera():
            self.cleanup()
            return

        while self.current_cam_idx < len(self.cam_ids):
            frame = self.read_frame()

            if frame is None:
                print("[WARN] Frame ungültig, wechsle zur nächsten Kamera.")
                self.status.set("Frame ungueltig - naechste Kamera", COLOR_RED)
                if not self.move_to_next_camera():
                    break
                continue

            cam_id = self.cam_ids[self.current_cam_idx]
            display_frame = self.render_frame(frame, cam_id)

            cv2.imshow(WINDOW_NAME, display_frame)
            key = cv2.waitKey(1) & 0xFF

            if not self.handle_key(key):
                break

        self.cleanup()

    def cleanup(self) -> None:
        self.release_camera()
        cv2.destroyAllWindows()


# =========================================================
# STARTFUNKTION
# =========================================================
def start_calibration() -> None:
    print("[SYSTEM] Kalibrierung gestartet...")
    calibrator = Calibrator(cam_ids=[0, 1, 2])
    calibrator.run()


# =========================================================
# SCRIPT STARTPUNKT
# =========================================================
if __name__ == "__main__":
    start_calibration()
