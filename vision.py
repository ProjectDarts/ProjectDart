import cv2
import numpy as np
import json
import os
import sys
import time

from vision_absdiff import AbsDiffDetector
from vision_takeout import TakeoutDetector


def get_external_path(filename):
    if getattr(sys, 'frozen', False):
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, filename)


class CameraHandler:
    def __init__(self, cam_id, canvas_size=800, nutzungs_faktor=0.70):
        self.cam_id = cam_id
        self.canvas_size = int(canvas_size)
        self.nutzungs_faktor = float(nutzungs_faktor)

        self.config_file = get_external_path(f"cam{cam_id}_config.json")
        self.src_points = []
        self.load_config()

        self.cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        time.sleep(1.5)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # Exposure
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # manual
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -7)
        self.cap.set(cv2.CAP_PROP_GAIN, 10)
        if self.cam_id == 2:
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 100)
        else:
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)

        self.matrix = None
        self.compute_warp_matrix()

        # Motion reference
        self.reference_gray = None

    def load_config(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    data = json.load(f)
                    self.src_points = data.get("points", [])
            except Exception:
                print(f"[ERROR] Config für Cam {self.cam_id} konnte nicht geladen werden.")
                self.src_points = []

    def compute_warp_matrix(self):
        if len(self.src_points) < 4:
            return

        pts1 = np.float32(self.src_points)

        canvas = self.canvas_size
        c = canvas / 2.0
        dist = (canvas / 2.0) * self.nutzungs_faktor

        # Rotation 9°: cos=0.987, sin=0.156
        sin9, cos9 = 0.156, 0.987

        top =  (c + dist * sin9, c - dist * cos9)
        right = (c + dist * cos9, c + dist * sin9)
        bot =  (c - dist * sin9, c + dist * cos9)
        left = (c - dist * cos9, c - dist * sin9)

        pts2 = np.float32([top, right, bot, left])
        self.matrix = cv2.getPerspectiveTransform(pts1, pts2)

    def get_warped(self, frame):
        if self.matrix is None or frame is None:
            return None
        return cv2.warpPerspective(frame, self.matrix, (self.canvas_size, self.canvas_size))

    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass


class DartVisionSystem:
    def __init__(self, hit_callback):
        self.hit_callback = hit_callback

        # --- Bild & Masken ---
        self.canvas_size = 800
        self.center = (self.canvas_size / 2.0, self.canvas_size / 2.0)

        # Board-Skalierung (wie bei dir, nur canvas angepasst)
        total_radius_mm = 170.0 + 55.0
        self.nutzungs_faktor = 0.70
        self.px_per_mm = (self.canvas_size * self.nutzungs_faktor) / (total_radius_mm * 2.0)

        # Board-Maske: bis Double-Outer
        self.board_mask = np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)
        r_double_outer = int(round(170.0 * self.px_per_mm))
        cv2.circle(self.board_mask, (int(self.center[0]), int(self.center[1])), r_double_outer, 255, -1)

        # Extended Maske: größer, damit Flights / Überstand fürs Event drin sind
        self.extended_mask = np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)
        extra_mm = 120.0  # ggf. 80..140 tunen
        r_ext = int(round((170.0 + extra_mm) * self.px_per_mm))
        r_ext = min(r_ext, int(self.center[0]) - 2)
        cv2.circle(self.extended_mask, (int(self.center[0]), int(self.center[1])), r_ext, 255, -1)

        # --- Kameras ---
        self.cameras = [CameraHandler(i, canvas_size=self.canvas_size, nutzungs_faktor=self.nutzungs_faktor) for i in range(3)]

        # --- Detektoren ---
        self.detectors = [AbsDiffDetector(self.board_mask, self.extended_mask) for _ in range(3)]
        self.takeout_detectors = [TakeoutDetector(self.board_mask, self.extended_mask) for _ in range(3)]

        # Scoring-Radien
        self.radii = {
            "bull": 6.35 * self.px_per_mm,
            "single_bull": 15.9 * self.px_per_mm,
            "triple_inner": 97.0 * self.px_per_mm,
            "triple_outer": 107.0 * self.px_per_mm,
            "double_inner": 160.0 * self.px_per_mm,
            "double_outer": 170.0 * self.px_per_mm
        }

        self.WINKEL_OFFSET = 0

        # --- State ---
        self.running = True
        self.last_hit_time = 0.0
        self.cooldown_s = 0.25

        self.last_hit_coords = None  # (x,y) final
        self.has_dart_in_board = False

        # Temporal verification (2-Frame confirm)
        self.candidate = None
        self.candidate_time = 0.0
        self.confirm_dist_px = 18
        self.confirm_time_s = 0.25

        # Motion freeze (gegen Board-Schwingen)
        self.freeze_mean = 16.0
        self.freeze_max = 70.0

        # Event gate
        self.event_pixel_threshold = 450  # je nach Licht 250..900

    def reset_references(self):
        for i, cam in enumerate(self.cameras):
            cam.load_config()
            cam.compute_warp_matrix()
            if cam.matrix is None:
                continue

            for _ in range(10):
                cam.cap.read()

            ret, frame = cam.cap.read()
            if not ret:
                continue

            warped = cam.get_warped(frame)
            if warped is None:
                continue

            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            cam.reference_gray = gray.copy()

            self.detectors[i].set_reference(warped)
            self.takeout_detectors[i].set_clean_board(warped)

        self.candidate = None
        self.has_dart_in_board = False
        self.last_hit_coords = None

    def update_references(self):
        # nach Treffer: neue Referenz, damit AbsDiff nicht “dauernd Dart” sieht
        for i, cam in enumerate(self.cameras):
            for _ in range(5):
                cam.cap.read()
            ret, frame = cam.cap.read()
            if not ret:
                continue
            warped = cam.get_warped(frame)
            if warped is None:
                continue
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            cam.reference_gray = gray.copy()
            self.detectors[i].set_reference(warped)

    def is_board_moving(self, gray, cam_ref_gray):
        if cam_ref_gray is None:
            return False
        diff = cv2.absdiff(gray, cam_ref_gray)
        mean_val = cv2.mean(diff)[0]
        _, max_val, _, _ = cv2.minMaxLoc(diff)
        return (mean_val > self.freeze_mean and max_val < self.freeze_max)

    def get_score(self, x, y):
        cx, cy = self.center
        rel_x, rel_y = x - cx, y - cy
        dist = float(np.linalg.norm([rel_x, rel_y]))
        angle = (np.degrees(np.arctan2(-rel_y, rel_x)) + 360.0) % 360.0
        angle = (angle + self.WINKEL_OFFSET) % 360.0

        segments = [6, 13, 4, 18, 1, 20, 5, 12, 9, 14, 11, 8, 16, 7, 19, 3, 17, 2, 15, 10]
        val = segments[int((angle + 9) / 18) % 20]

        if dist <= self.radii["bull"]:
            return (25, 2)
        if dist <= self.radii["single_bull"]:
            return (25, 1)
        if self.radii["triple_inner"] <= dist <= self.radii["triple_outer"]:
            return (val, 3)
        if self.radii["double_inner"] <= dist <= self.radii["double_outer"]:
            return (val, 2)
        if dist <= self.radii["double_outer"]:
            return (val, 1)
        return (0, 0)

    def point_in_board(self, x, y):
        ix, iy = int(round(x)), int(round(y))
        if ix < 0 or iy < 0 or ix >= self.canvas_size or iy >= self.canvas_size:
            return False
        return self.board_mask[iy, ix] > 0

    def run(self):
        print("[VISION] System bereit (NEU)...")
        try:
            self.reset_references()

            while self.running:
                debug_frames = {}
                valid = []  # (cam_id, tip(x,y), conf, tip_in_board)

                # Takeout: leer?
                empties = 0
                moving_any = False

                for i, cam in enumerate(self.cameras):
                    ret, frame = cam.cap.read()
                    if not ret:
                        continue

                    warped = cam.get_warped(frame)
                    if warped is None:
                        continue

                    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

                    # Motion freeze pro Cam
                    if self.is_board_moving(gray, cam.reference_gray):
                        moving_any = True
                        # trotzdem Takeout nicht entscheiden, aber debug zeigen
                        empty_now, tdbg = self.takeout_detectors[i].is_empty(warped, gray)
                        debug_frames[cam.cam_id] = tdbg
                        continue

                    # Takeout check
                    empty_now, tdbg = self.takeout_detectors[i].is_empty(warped, gray)
                    if empty_now:
                        empties += 1

                    # AbsDiff candidates
                    objs, dbg = self.detectors[i].detect(warped, gray=gray, center=self.center)

                    # Event gate (optional, hilft gegen random noise):
                    # Wenn keinerlei diff im extended ROI, dann skip candidates.
                    # (Wir nutzen detector schon; hier könnte man zusätzlich thr count nutzen,
                    # aber AbsDiffDetector macht bereits gates.)
                    if objs:
                        best = max(objs, key=lambda o: o["confidence"])
                        tip = best["tip"]
                        conf = float(best["confidence"])
                        valid.append((cam.cam_id, tip, conf, bool(best["tip_in_board"])))
                        cv2.circle(tdbg, (int(tip[0]), int(tip[1])), 8, (0, 0, 255), -1)

                    debug_frames[cam.cam_id] = tdbg

                # Wenn Board bewegt sich: keine Entscheidungen, Candidate reset
                if moving_any:
                    self.candidate = None
                    cv2.waitKey(1)
                    for cid, img in debug_frames.items():
                        cv2.imshow(f"Cam {cid} Debug", img)
                    time.sleep(0.01)
                    continue

                # --- TAKEOUT LOGIK ---
                # Wenn Dart vorher da war und jetzt alle 3 "empty": Next player
                if self.last_hit_coords is not None and empties >= 3:
                    print("[VISION] Takeout: alle Cams leer -> NEXT_PLAYER")
                    self.last_hit_coords = None
                    self.has_dart_in_board = False
                    self.candidate = None
                    self.reset_references()
                    self.hit_callback("NEXT_PLAYER")

                # --- HIT / MISSED LOGIK ---
                # Wir verarbeiten nur, wenn mindestens 2 Cams etwas sehen
                if len(valid) >= 2 and (time.time() - self.last_hit_time) > self.cooldown_s:
                    # sort by confidence
                    valid.sort(key=lambda x: x[2], reverse=True)

                    top = valid[:3]
                    weights = [min(v[2], 1e6) for v in top]
                    points = [v[1] for v in top]

                    fx = float(np.average([p[0] for p in points], weights=weights))
                    fy = float(np.average([p[1] for p in points], weights=weights))

                    # Plausibility: beste 2 dürfen nicht zu weit auseinander liegen
                    d12 = float(np.linalg.norm(np.array(valid[0][1]) - np.array(valid[1][1])))
                    if d12 < 85:
                        current = (fx, fy)

                        # Temporal confirm (2 Frames)
                        if self.candidate is None:
                            self.candidate = current
                            self.candidate_time = time.time()
                        else:
                            dist_prev = float(np.linalg.norm(np.array(current) - np.array(self.candidate)))
                            dt = time.time() - self.candidate_time

                            if dist_prev < self.confirm_dist_px and dt < self.confirm_time_s:
                                # CONFIRMED
                                in_board = self.point_in_board(fx, fy)

                                if in_board:
                                    sec, mult = self.get_score(fx, fy)
                                    if (sec, mult) != (0, 0):
                                        payload = {"sector": int(sec), "is_missed": False, "x": float(fx), "y": float(fy)}
                                        print(f"[VISION] HIT sector={sec} at {fx:.1f},{fy:.1f}")
                                        self.hit_callback(payload)

                                        self.last_hit_coords = (fx, fy)
                                        self.has_dart_in_board = True
                                        self.last_hit_time = time.time()

                                        time.sleep(0.25)  # kurze Stabilisierung
                                        self.update_references()

                                else:
                                    # Throw erkannt, aber außerhalb Board -> MISSED
                                    payload = {"sector": 0, "is_missed": True, "x": float(fx), "y": float(fy)}
                                    print(f"[VISION] MISSED at {fx:.1f},{fy:.1f}")
                                    self.hit_callback(payload)

                                    self.last_hit_coords = (fx, fy)  # merken, damit Takeout überhaupt Sinn macht
                                    self.has_dart_in_board = False
                                    self.last_hit_time = time.time()

                                    time.sleep(0.15)
                                    self.update_references()

                                self.candidate = None
                            else:
                                # reset candidate
                                self.candidate = current
                                self.candidate_time = time.time()
                    else:
                        # zu weit auseinander -> kein event
                        self.candidate = None

                # Debug anzeigen
                for cid, img in debug_frames.items():
                    cv2.imshow(f"Cam {cid} Debug", img)
                cv2.waitKey(1)

        except Exception as e:
            print(f"[ERROR] Vision run crashed: {e}")

    def stop(self):
        self.running = False
        for cam in self.cameras:
            cam.release()
        cv2.destroyAllWindows()
