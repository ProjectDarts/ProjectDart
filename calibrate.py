import cv2
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# =========================================================
# KONSTANTEN
# =========================================================
WINDOW_NAME = "Kalibrierung - Touch UI"

FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

MAX_POINTS = 4
ZOOM_FACTOR = 3
ZOOM_SIZE = 110

CAMERA_WARMUP_SECONDS = 1.5
DISCARD_INITIAL_FRAMES = 10

TOUCH_HIT_RADIUS = 28
STATUS_MESSAGE_DURATION = 2.2

# Eigene UI-Leiste unterhalb des Kamerabilds
CONTROL_BAR_HEIGHT = 150
SIDE_PANEL_WIDTH = 420
WINDOW_WIDTH = FRAME_WIDTH + SIDE_PANEL_WIDTH
WINDOW_HEIGHT = FRAME_HEIGHT + CONTROL_BAR_HEIGHT

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Farben (BGR)
COLOR_BG = (18, 18, 18)
COLOR_PANEL = (34, 34, 34)
COLOR_PANEL_2 = (44, 44, 44)
COLOR_PANEL_3 = (58, 58, 58)
COLOR_BORDER = (90, 90, 90)

COLOR_WHITE = (245, 245, 245)
COLOR_LIGHT = (220, 220, 220)
COLOR_MUTED = (160, 160, 160)

COLOR_GREEN = (80, 220, 120)
COLOR_YELLOW = (80, 220, 255)
COLOR_RED = (60, 80, 255)
COLOR_BLUE = (255, 170, 70)
COLOR_ORANGE = (0, 170, 255)

COLOR_POINT = (80, 220, 120)
COLOR_POINT_SELECTED = (0, 180, 255)
COLOR_CROSSHAIR = (0, 0, 255)

PANEL_ALPHA = 0.78


# =========================================================
# PFAD-LOGIK
# =========================================================
def get_config_path(filename: str) -> str:
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
    finger_down: bool = False

    # Touch / Bearbeitung
    press_x: int = 0
    press_y: int = 0
    preview_x: int = FRAME_WIDTH // 2
    preview_y: int = FRAME_HEIGHT // 2
    active_point_idx: Optional[int] = None
    pending_new_point: bool = False
    pending_button: Optional[str] = None

    def reset(self) -> None:
        self.points.clear()
        self.mouse_x = FRAME_WIDTH // 2
        self.mouse_y = FRAME_HEIGHT // 2
        self.finger_down = False
        self.press_x = 0
        self.press_y = 0
        self.preview_x = FRAME_WIDTH // 2
        self.preview_y = FRAME_HEIGHT // 2
        self.active_point_idx = None
        self.pending_new_point = False
        self.pending_button = None

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

    @property
    def is_active(self) -> bool:
        return bool(self.text) and time.time() < self.until


@dataclass
class Button:
    key: str
    label: str
    rect: Tuple[int, int, int, int]
    fill: Tuple[int, int, int]
    border: Tuple[int, int, int] = COLOR_BORDER

    def contains(self, x: int, y: int) -> bool:
        x1, y1, x2, y2 = self.rect
        return x1 <= x <= x2 and y1 <= y <= y2


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
        self.running = True
        self.buttons: List[Button] = []

    # =====================================================
    # KAMERA
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

        self.load_existing_points()
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
    def get_current_config_filename(self) -> str:
        cam_id = self.cam_ids[self.current_cam_idx]
        return get_config_path(f"cam{cam_id}_config.json")

    def load_existing_points(self) -> bool:
        filename = self.get_current_config_filename()

        if not os.path.exists(filename):
            return False

        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)

            points = data.get("points", [])
            valid_points = []

            for p in points[:MAX_POINTS]:
                if (
                    isinstance(p, list)
                    and len(p) == 2
                    and isinstance(p[0], (int, float))
                    and isinstance(p[1], (int, float))
                ):
                    x = int(np.clip(p[0], 0, FRAME_WIDTH - 1))
                    y = int(np.clip(p[1], 0, FRAME_HEIGHT - 1))
                    valid_points.append([x, y])

            self.state.points = valid_points

            if valid_points:
                print(f"[INFO] Vorhandene Kalibrierung geladen: {filename}")
                self.status.set("Vorhandene Punkte geladen", COLOR_BLUE)
                return True

        except Exception as e:
            print(f"[WARN] Laden fehlgeschlagen: {e}")
            self.status.set("Vorhandene Config konnte nicht geladen werden", COLOR_RED)

        return False

    def save_current_points(self) -> bool:
        filename = self.get_current_config_filename()

        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump({"points": self.state.points}, f, indent=2)

            print(f"[INFO] Erfolg: {filename} gespeichert.")
            self.status.set(f"Gespeichert: {os.path.basename(filename)}", COLOR_GREEN)
            return True

        except Exception as e:
            print(f"[ERROR] Speichern fehlgeschlagen: {e}")
            self.status.set("Speichern fehlgeschlagen", COLOR_RED)
            return False

    # =====================================================
    # HILFSFUNKTIONEN
    # =====================================================
    def get_current_cam_id(self) -> int:
        return self.cam_ids[self.current_cam_idx]

    def get_current_point_description(self) -> str:
        if self.state.is_complete:
            return "Fertig"
        return self.POINTS_DESC[self.state.point_count]

    def clamp_to_frame(self, x: int, y: int) -> Tuple[int, int]:
        x = int(np.clip(x, 0, FRAME_WIDTH - 1))
        y = int(np.clip(y, 0, FRAME_HEIGHT - 1))
        return x, y

    def distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def find_nearest_point(self, x: int, y: int, max_radius: int = TOUCH_HIT_RADIUS) -> Optional[int]:
        best_idx = None
        best_dist = 1e9

        for i, p in enumerate(self.state.points):
            d = self.distance((x, y), (p[0], p[1]))
            if d <= max_radius and d < best_dist:
                best_idx = i
                best_dist = d

        return best_idx

    # =====================================================
    # TOUCH / BUTTONS
    # =====================================================
    def build_buttons(self) -> List[Button]:
        margin = 22
        gap = 16
        y1 = FRAME_HEIGHT + 22
        y2 = WINDOW_HEIGHT - 22

        button_defs = [
            ("undo", "Undo", COLOR_PANEL_3),
            ("reset", "Reset", COLOR_PANEL_3),
            ("save", "Save", COLOR_GREEN),
            ("skip", "Skip", COLOR_BLUE),
            ("quit", "Quit", COLOR_RED),
        ]

        available_w = WINDOW_WIDTH - 2 * margin - gap * (len(button_defs) - 1)
        button_w = available_w // len(button_defs)

        buttons = []
        x = margin
        for key, label, fill in button_defs:
            rect = (x, y1, x + button_w, y2)
            buttons.append(Button(key=key, label=label, rect=rect, fill=fill))
            x += button_w + gap

        return buttons

    def handle_button_action(self, key: str) -> None:
        if key == "undo":
            if self.state.remove_last_point():
                self.status.set("Letzter Punkt entfernt", COLOR_YELLOW, 1.6)
            else:
                self.status.set("Keine Punkte zum Entfernen", COLOR_MUTED, 1.4)
            return

        if key == "reset":
            self.state.reset()
            self.status.set("Aktuelle Kamera zurückgesetzt", COLOR_YELLOW)
            return

        if key == "save":
            if not self.state.is_complete:
                self.status.set("Zum Speichern müssen 4 Punkte gesetzt sein", COLOR_YELLOW)
                return

            self.save_current_points()
            if not self.move_to_next_camera():
                self.running = False
            return

        if key == "skip":
            cam_id = self.get_current_cam_id()
            print(f"[INFO] Kamera {cam_id} übersprungen.")
            self.status.set(f"Kamera {cam_id} übersprungen", COLOR_BLUE)
            if not self.move_to_next_camera():
                self.running = False
            return

        if key == "quit":
            print("[INFO] Kalibrierung abgebrochen.")
            self.running = False
            return

    def on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        """
        Touch-/Mouse-Handling:
        - Punkt wird IMMER erst bei EVENT_LBUTTONUP gesetzt
        - Touch auf vorhandenen Punkt -> Punkt wird beim Loslassen verschoben
        - Touch auf Button -> Aktion erst beim Loslassen
        """
        self.state.mouse_x = x
        self.state.mouse_y = y

        if event == cv2.EVENT_MOUSEMOVE and self.state.finger_down:
            self.state.preview_x = x
            self.state.preview_y = y

        if event == cv2.EVENT_LBUTTONDOWN:
            self.state.finger_down = True
            self.state.press_x = x
            self.state.press_y = y
            self.state.preview_x = x
            self.state.preview_y = y
            self.state.pending_button = None
            self.state.active_point_idx = None
            self.state.pending_new_point = False

            # Prüfen: Button getroffen?
            for btn in self.buttons:
                if btn.contains(x, y):
                    self.state.pending_button = btn.key
                    return

            # Nur im Kamerabereich dürfen Punkte bearbeitet/gesetzt werden
            if 0 <= x < FRAME_WIDTH and 0 <= y < FRAME_HEIGHT:
                nearest_idx = self.find_nearest_point(x, y)
                if nearest_idx is not None:
                    self.state.active_point_idx = nearest_idx
                elif not self.state.is_complete:
                    self.state.pending_new_point = True

        elif event == cv2.EVENT_LBUTTONUP:
            release_x, release_y = x, y

            # Button-Aktion erst beim Loslassen
            if self.state.pending_button is not None:
                pressed_key = self.state.pending_button
                self.state.pending_button = None
                self.state.finger_down = False

                for btn in self.buttons:
                    if btn.key == pressed_key and btn.contains(release_x, release_y):
                        self.handle_button_action(pressed_key)
                        return
                return

            # Punkt-Interaktion nur im Kamerabereich
            if 0 <= release_x < FRAME_WIDTH and 0 <= release_y < FRAME_HEIGHT:
                rx, ry = self.clamp_to_frame(release_x, release_y)

                if self.state.active_point_idx is not None:
                    idx = self.state.active_point_idx
                    self.state.points[idx] = [rx, ry]
                    self.status.set(f"Punkt {idx + 1} verschoben", COLOR_ORANGE, 1.6)
                    print(f"[INFO] Punkt {idx + 1} verschoben auf: {rx}, {ry}")

                elif self.state.pending_new_point and not self.state.is_complete:
                    self.state.points.append([rx, ry])
                    idx = len(self.state.points) - 1
                    desc = self.POINTS_DESC[idx]
                    self.status.set(f"Punkt {idx + 1}/4 gesetzt: {desc}", COLOR_GREEN, 1.8)
                    print(f"[INFO] Punkt {idx + 1} ({desc}) gesetzt: {rx}, {ry}")

            self.state.finger_down = False
            self.state.active_point_idx = None
            self.state.pending_new_point = False

    # =====================================================
    # UI-HILFEN
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
    def draw_crosshair(
        img: np.ndarray,
        x: int,
        y: int,
        size: int = 18,
        color: Tuple[int, int, int] = COLOR_CROSSHAIR,
    ) -> None:
        cv2.line(img, (x - size, y), (x + size, y), color, 1, cv2.LINE_AA)
        cv2.line(img, (x, y - size), (x, y + size), color, 1, cv2.LINE_AA)
        cv2.circle(img, (x, y), 4, color, 1, cv2.LINE_AA)

    def draw_progress_bar(self, img: np.ndarray, x: int, y: int, w: int, h: int) -> None:
        self.overlay_rect(img, x, y, x + w, y + h, COLOR_PANEL_2, alpha=0.9, border_color=COLOR_BORDER)
        progress = self.state.point_count / MAX_POINTS
        fill_w = int((w - 4) * progress)

        if fill_w > 0:
            cv2.rectangle(img, (x + 2, y + 2), (x + 2 + fill_w, y + h - 2), COLOR_GREEN, -1)

        self.draw_text(img, f"{self.state.point_count} / {MAX_POINTS} Punkte", (x + 10, y + h - 10), 0.65, COLOR_WHITE, 2)

    # =====================================================
    # RENDERING - KAMERA
    # =====================================================
    def draw_points(self, frame: np.ndarray) -> None:
        for i, point in enumerate(self.state.points):
            px, py = point
            is_selected = self.state.active_point_idx == i and self.state.finger_down

            outer = COLOR_POINT_SELECTED if is_selected else COLOR_WHITE
            inner = COLOR_POINT_SELECTED if is_selected else COLOR_POINT

            cv2.circle(frame, (px, py), 16, outer, 2, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 10, inner, -1, cv2.LINE_AA)

            label = str(i + 1)
            self.draw_text(frame, label, (px + 18, py - 14), 0.78, inner, 2)

    def draw_polygon_preview(self, frame: np.ndarray) -> None:
        if len(self.state.points) >= 2:
            pts = np.array(self.state.points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], False, COLOR_BLUE, 2, cv2.LINE_AA)

        if len(self.state.points) == 4:
            pts = np.array(self.state.points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, COLOR_GREEN, 2, cv2.LINE_AA)

    def draw_live_touch_preview(self, frame: np.ndarray) -> None:
        if not self.state.finger_down:
            return

        x, y = self.clamp_to_frame(self.state.preview_x, self.state.preview_y)

        if self.state.active_point_idx is not None:
            cv2.circle(frame, (x, y), 20, COLOR_ORANGE, 2, cv2.LINE_AA)
            self.draw_crosshair(frame, x, y, size=20, color=COLOR_ORANGE)
        elif self.state.pending_new_point and not self.state.is_complete:
            cv2.circle(frame, (x, y), 20, COLOR_YELLOW, 2, cv2.LINE_AA)
            self.draw_crosshair(frame, x, y, size=20, color=COLOR_YELLOW)

    def draw_zoom(self, frame: np.ndarray, canvas: np.ndarray) -> None:
        if not self.state.finger_down and self.state.point_count >= MAX_POINTS:
            return

        src_x, src_y = self.clamp_to_frame(
            self.state.preview_x if self.state.finger_down else self.state.mouse_x,
            self.state.preview_y if self.state.finger_down else self.state.mouse_y,
        )

        half_size = ZOOM_SIZE // 2
        x1 = max(0, src_x - half_size)
        y1 = max(0, src_y - half_size)
        x2 = min(FRAME_WIDTH, src_x + half_size)
        y2 = min(FRAME_HEIGHT, src_y + half_size)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return

        zoom_img = cv2.resize(
            roi,
            (ZOOM_SIZE * ZOOM_FACTOR, ZOOM_SIZE * ZOOM_FACTOR),
            interpolation=cv2.INTER_NEAREST,
        )

        h_z, w_z = zoom_img.shape[:2]

        panel_x1 = FRAME_WIDTH + 24
        panel_y1 = 24
        panel_x2 = WINDOW_WIDTH - 24
        panel_y2 = panel_y1 + h_z + 54

        self.overlay_rect(canvas, panel_x1, panel_y1, panel_x2, panel_y2, COLOR_PANEL, alpha=0.9, border_color=COLOR_BORDER)
        self.draw_text(canvas, "Lupe", (panel_x1 + 14, panel_y1 + 28), 0.72, COLOR_WHITE, 2)

        img_x1 = panel_x1 + 14
        img_y1 = panel_y1 + 40
        img_x2 = img_x1 + w_z
        img_y2 = img_y1 + h_z

        canvas[img_y1:img_y2, img_x1:img_x2] = zoom_img
        cv2.rectangle(canvas, (img_x1, img_y1), (img_x2, img_y2), COLOR_WHITE, 1)

        center_x = img_x1 + w_z // 2
        center_y = img_y1 + h_z // 2
        cv2.line(canvas, (center_x, img_y1), (center_x, img_y2), COLOR_CROSSHAIR, 1, cv2.LINE_AA)
        cv2.line(canvas, (img_x1, center_y), (img_x2, center_y), COLOR_CROSSHAIR, 1, cv2.LINE_AA)

    # =====================================================
    # RENDERING - SIDE PANEL
    # =====================================================
    def draw_side_info_panel(self, canvas: np.ndarray) -> None:
        x1 = FRAME_WIDTH + 24
        y1 = 420
        x2 = WINDOW_WIDTH - 24
        y2 = 780

        self.overlay_rect(canvas, x1, y1, x2, y2, COLOR_PANEL, alpha=0.9, border_color=COLOR_BORDER)

        cam_id = self.get_current_cam_id()
        self.draw_text(canvas, f"Kamera {cam_id}", (x1 + 16, y1 + 38), 0.95, COLOR_WHITE, 2)
        self.draw_text(canvas, f"Auflösung: {FRAME_WIDTH}x{FRAME_HEIGHT}", (x1 + 16, y1 + 70), 0.62, COLOR_MUTED, 1)

        if self.state.is_complete:
            target_text = "Alle 4 Punkte gesetzt"
            target_color = COLOR_GREEN
        else:
            target_text = f"Nächster Punkt: {self.get_current_point_description()}"
            target_color = COLOR_YELLOW

        self.draw_text(canvas, target_text, (x1 + 16, y1 + 112), 0.72, target_color, 2)

        self.draw_progress_bar(canvas, x1 + 16, y1 + 136, 250, 26)

        self.draw_text(canvas, "Punktreihenfolge", (x1 + 16, y1 + 200), 0.68, COLOR_LIGHT, 1)

        for i, desc in enumerate(self.POINTS_DESC):
            yy = y1 + 238 + i * 34
            done = i < self.state.point_count
            is_current = i == self.state.point_count and not self.state.is_complete

            marker = "[OK]" if done else ">>" if is_current else "[ ]"
            color = COLOR_GREEN if done else COLOR_YELLOW if is_current else COLOR_MUTED
            self.draw_text(canvas, f"{marker} {i+1}. {desc}", (x1 + 16, yy), 0.60, color, 2 if (done or is_current) else 1)

    def draw_status_panel(self, canvas: np.ndarray) -> None:
        x1 = FRAME_WIDTH + 24
        y1 = 804
        x2 = WINDOW_WIDTH - 24
        y2 = FRAME_HEIGHT - 24

        self.overlay_rect(canvas, x1, y1, x2, y2, COLOR_PANEL, alpha=0.9, border_color=COLOR_BORDER)

        self.draw_text(canvas, "Status", (x1 + 16, y1 + 34), 0.78, COLOR_WHITE, 2)

        if self.status.is_active:
            text = self.status.text
            color = self.status.color
        else:
            if self.state.is_complete:
                text = "Bereit zum Speichern"
                color = COLOR_GREEN
            else:
                text = "Touch auf Bild: Punkt setzen oder verschieben"
                color = COLOR_LIGHT

        self.draw_text(canvas, text, (x1 + 16, y1 + 82), 0.62, color, 2)

        help_lines = [
            "Touch & Release im Bild: Punkt setzen",
            "Touch auf bestehenden Punkt: Punkt verschieben",
            "Buttons unten: komplette Bedienung ohne Tastatur",
        ]
        yy = y1 + 130
        for line in help_lines:
            self.draw_text(canvas, line, (x1 + 16, yy), 0.54, COLOR_MUTED, 1)
            yy += 30

    # =====================================================
    # RENDERING - CONTROL BAR
    # =====================================================
    def draw_buttons(self, canvas: np.ndarray) -> None:
        self.buttons = self.build_buttons()

        # Hintergrund der unteren Leiste
        self.overlay_rect(
            canvas,
            0,
            FRAME_HEIGHT,
            WINDOW_WIDTH - 1,
            WINDOW_HEIGHT - 1,
            COLOR_BG,
            alpha=1.0,
            border_color=COLOR_BORDER,
            border_thickness=1,
        )

        for btn in self.buttons:
            x1, y1, x2, y2 = btn.rect
            is_pressed = self.state.pending_button == btn.key and self.state.finger_down

            fill = btn.fill
            border = COLOR_WHITE if is_pressed else btn.border
            thickness = 3 if is_pressed else 2

            self.overlay_rect(canvas, x1, y1, x2, y2, fill, alpha=0.88, border_color=border, border_thickness=thickness)

            label_scale = 1.0 if (x2 - x1) > 220 else 0.85
            text_size, _ = cv2.getTextSize(btn.label, FONT, label_scale, 2)
            tw, th = text_size
            tx = x1 + ((x2 - x1) - tw) // 2
            ty = y1 + ((y2 - y1) + th) // 2

            text_color = COLOR_DARK if btn.key == "save" else COLOR_WHITE
            self.draw_text(canvas, btn.label, (tx, ty), label_scale, text_color, 2)

    # =====================================================
    # GESAMT-RENDERING
    # =====================================================
    def render_canvas(self, frame: np.ndarray) -> np.ndarray:
        canvas = np.full((WINDOW_HEIGHT, WINDOW_WIDTH, 3), COLOR_BG, dtype=np.uint8)

        # Kamerabild oben links
        canvas[0:FRAME_HEIGHT, 0:FRAME_WIDTH] = frame

        # Separator rechts
        cv2.line(canvas, (FRAME_WIDTH, 0), (FRAME_WIDTH, FRAME_HEIGHT), COLOR_BORDER, 1, cv2.LINE_AA)

        # Panels
        self.draw_zoom(frame, canvas)
        self.draw_side_info_panel(canvas)
        self.draw_status_panel(canvas)
        self.draw_buttons(canvas)

        return canvas

    def render_frame(self, frame: np.ndarray) -> np.ndarray:
        display_frame = frame.copy()

        self.draw_polygon_preview(display_frame)
        self.draw_points(display_frame)
        self.draw_live_touch_preview(display_frame)

        # optionales Fadenkreuz nur im Bildbereich
        mx, my = self.state.mouse_x, self.state.mouse_y
        if 0 <= mx < FRAME_WIDTH and 0 <= my < FRAME_HEIGHT:
            self.draw_crosshair(display_frame, mx, my, size=16, color=COLOR_RED)

        return self.render_canvas(display_frame)

    # =====================================================
    # HAUPTLAUF
    # =====================================================
    def run(self) -> None:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1600, 900)

        # Buttons einmal initial anlegen, damit Touch sofort funktioniert
        self.buttons = self.build_buttons()
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)

        if not self.setup_current_camera():
            self.cleanup()
            return

        while self.running and self.current_cam_idx < len(self.cam_ids):
            frame = self.read_frame()

            if frame is None:
                print("[WARN] Frame ungültig, wechsle zur nächsten Kamera.")
                self.status.set("Frame ungültig - nächste Kamera", COLOR_RED)
                if not self.move_to_next_camera():
                    break
                continue

            canvas = self.render_frame(frame)
            cv2.imshow(WINDOW_NAME, canvas)

            # nur fürs OpenCV-Fenster-Update; keine Bedienlogik mehr nötig
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # optionaler Not-Aus
                print("[INFO] Kalibrierung abgebrochen.")
                break

        self.cleanup()

    def cleanup(self) -> None:
        self.release_camera()
        cv2.destroyAllWindows()


# =========================================================
# START
# =========================================================
def start_calibration() -> None:
    print("[SYSTEM] Kalibrierung gestartet...")
    calibrator = Calibrator(cam_ids=[0, 1, 2])
    calibrator.run()


if __name__ == "__main__":
    start_calibration()
