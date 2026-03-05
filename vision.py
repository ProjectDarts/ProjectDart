import cv2
import numpy as np
import json
import os
import sys
import time
import configparser

from vision_absdiff import AbsDiffDetector
from vision_takeout import TakeoutDetector

def get_external_path(filename):
    if getattr(sys, 'frozen', False):
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, filename)

def read_debug_ini():
    ini_path = get_external_path("vision_debug.ini")
    cfg = configparser.ConfigParser()
    debugging = 0
    warp_size = 800
    if os.path.exists(ini_path):
        cfg.read(ini_path, encoding="utf-8")
        debugging = int(cfg.get("vision", "debugging", fallback="0").strip())
        warp_size = int(cfg.get("vision", "warp_size", fallback="800").strip())
    return debugging == 1, warp_size

def load_points(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        pts = data.get("points", None)
        if not pts or len(pts) != 4:
            return None
        return np.float32(pts)
    except:
        return None

class CameraHandler:
    def __init__(self, cam_id):
        self.cam_id = cam_id
        self.config_file = get_external_path(f"cam{cam_id}_config.json")
        self.src_points = load_points(self.config_file)

        self.cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        time.sleep(1.2)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # deine Exposure-Fixes
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -7)
        self.cap.set(cv2.CAP_PROP_GAIN, 10)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 100 if cam_id == 2 else 150)

        self.H = None      # cam -> board(600)
        self.Hinv = None
        self.board_center_full = None

        self.compute_homography()

    def compute_homography(self):
        if self.src_points is None:
            self.H = None
            self.Hinv = None
            self.board_center_full = None
            return

        # Board-space (600x600) Referenzpunkte (wie bisher, 9° Rotation)
        canvas = 600
        c = canvas / 2
        f = 0.70
        r = (canvas / 2) * f
        cos9, sin9 = 0.987, 0.156

        dst = np.float32([
            [c + r * sin9, c - r * cos9],  # top
            [c + r * cos9, c + r * sin9],  # right
            [c - r * sin9, c + r * cos9],  # bottom
            [c - r * cos9, c - r * sin9],  # left
        ])

        self.H = cv2.getPerspectiveTransform(self.src_points, dst)
        self.Hinv = np.linalg.inv(self.H)

        # Boardzentrum in Full-Frame schätzen (für Tip-Richtung)
        pt = np.array([[[300.0, 300.0]]], dtype=np.float32)
        center_full = cv2.perspectiveTransform(pt, self.Hinv)[0][0]
        self.board_center_full = (float(center_full[0]), float(center_full[1]))

    def read(self):
        return self.cap.read()

class DartVisionSystem:
    def __init__(self, hit_callback):
        self.hit_callback = hit_callback
        self.running = True

        self.debug_enabled, warp_size = read_debug_ini()
        self.debugger = None
        if self.debug_enabled:
            from vision_debug import VisionDebugger
            self.debugger = VisionDebugger(warp_size=warp_size)

        self.cameras = [CameraHandler(i) for i in range(3)]

        # Board mask + radii (board space)
        self.canvas = 600
        self.center = np.array([300.0, 300.0], dtype=np.float32)

        total_radius_mm = 170.0 + 55.0
        self.px_per_mm = (self.canvas * 0.70) / (total_radius_mm * 2)

        self.radii = {
            "bull": 6.35 * self.px_per_mm,
            "single_bull": 15.9 * self.px_per_mm,
            "triple_inner": 97.0 * self.px_per_mm,
            "triple_outer": 107.0 * self.px_per_mm,
            "double_inner": 160.0 * self.px_per_mm,
            "double_outer": 170.0 * self.px_per_mm
        }

        self.board_mask = np.zeros((600, 600), dtype=np.uint8)
        cv2.circle(self.board_mask, (300, 300), int(self.radii["double_outer"]), 255, -1)

        # Modules
        self.absdet = [AbsDiffDetector() for _ in range(3)]
        self.takeout = [TakeoutDetector(self.board_mask) for _ in range(3)]

        # State
        self.last_hit_time = 0.0
        self.last_hit_board = None  # (bx,by) für Duplicate-Filter
        self.hit_candidate = None
        self.hit_candidate_time = 0.0

        self.WINKEL_OFFSET = 0

        self.set_references()

    def set_references(self):
        for i, cam in enumerate(self.cameras):
            # buffer leeren
            for _ in range(5):
                cam.cap.read()
            ret, frame = cam.read()
            if not ret or frame is None:
                continue
            self.absdet[i].set_reference(frame)
            self.takeout[i].set_clean_board(frame)

    def full_to_board(self, cam, x, y):
        if cam.H is None:
            return None
        pt = np.array([[[float(x), float(y)]]], dtype=np.float32)
        b = cv2.perspectiveTransform(pt, cam.H)[0][0]
        return float(b[0]), float(b[1])

    def is_missed(self, bx, by):
        d = float(np.linalg.norm(np.array([bx, by], dtype=np.float32) - self.center))
        return d > float(self.radii["double_outer"])

    def get_score(self, bx, by):
        rel = np.array([bx, by], dtype=np.float32) - self.center
        dist = float(np.linalg.norm(rel))
        angle = (np.degrees(np.arctan2(-rel[1], rel[0])) + 360) % 360
        angle = (angle + self.WINKEL_OFFSET) % 360
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

    def fuse_points(self, board_points_conf):
        """
        board_points_conf: list of (bx,by,conf, cam_id, tip_full)
        -> weighted average of top 3
        """
        if len(board_points_conf) < 2:
            return None

        board_points_conf.sort(key=lambda t: t[2], reverse=True)
        top = board_points_conf[:3]

        weights = np.array([max(1.0, min(t[2], 1e6)) for t in top], dtype=np.float32)
        xs = np.array([t[0] for t in top], dtype=np.float32)
        ys = np.array([t[1] for t in top], dtype=np.float32)

        bx = float(np.average(xs, weights=weights))
        by = float(np.average(ys, weights=weights))
        return bx, by

    def run(self):
        while self.running:
            board_points_conf = []
            takeout_votes = 0

            # 1) pro Cam lesen + AbsDiff Tip finden + optional Debug anzeigen
            for i, cam in enumerate(self.cameras):
                ret, frame = cam.read()
                if not ret or frame is None or cam.H is None:
                    continue

                tips = self.absdet[i].detect(frame, board_center_full=cam.board_center_full)
                if tips:
                    best = tips[0]
                    tip_full = best["tip"]

                    tip_board = self.full_to_board(cam, tip_full[0], tip_full[1])
                    if tip_board is not None:
                        bx, by = tip_board
                        board_points_conf.append((bx, by, best["confidence"], cam.cam_id, tip_full))

                    if self.debugger:
                        self.debugger.show(cam.cam_id, frame, cam.H, tip_full=tip_full, tip_board=tip_board)
                else:
                    if self.debugger:
                        self.debugger.show(cam.cam_id, frame, cam.H, tip_full=None, tip_board=None)

                # Takeout nur sinnvoll, wenn vorher ein Dart erkannt wurde
                if self.last_hit_board is not None:
                    if self.takeout[i].check_takeout(frame, cam.H):
                        takeout_votes += 1

            # 2) Takeout-Event (2/3 Kameras reichen)
            if self.last_hit_board is not None and takeout_votes >= 2:
                self.last_hit_board = None
                self.hit_candidate = None
                self.set_references()
                self.hit_callback("NEXT_PLAYER")
                time.sleep(0.05)
                continue

            # 3) Hit-Fusion + Temporal Confirm + Duplicate Filter
            fused = self.fuse_points(board_points_conf)
            if fused is None:
                time.sleep(0.005)
                continue

            bx, by = fused
            now = time.time()

            # Temporal verification
            if self.hit_candidate is None:
                self.hit_candidate = (bx, by)
                self.hit_candidate_time = now
                continue

            dist_prev = float(np.linalg.norm(np.array([bx, by]) - np.array(self.hit_candidate)))
            if dist_prev > 18 or (now - self.hit_candidate_time) > 0.25:
                self.hit_candidate = (bx, by)
                self.hit_candidate_time = now
                continue

            # Debounce
            if now - self.last_hit_time < 0.25:
                continue

            # Duplicate protection
            if self.last_hit_board is not None:
                old_dist = float(np.linalg.norm(np.array([bx, by]) - np.array(self.last_hit_board)))
                if old_dist < 18:
                    continue

            # 4) Missed vs Score an main.py
            if self.is_missed(bx, by):
                payload = {"is_missed": True, "sector": 0, "board_x": bx, "board_y": by}
            else:
                sector, mult = self.get_score(bx, by)
                payload = {"is_missed": False, "sector": sector, "multiplier": mult, "board_x": bx, "board_y": by}

            self.hit_callback(payload)

            # 5) State update + references
            self.last_hit_board = (bx, by)
            self.last_hit_time = now
            self.hit_candidate = None

            # kurze Settle-time, dann reference aktualisieren
            time.sleep(0.25)
            self.set_references()

    def stop(self):
        self.running = False
        for cam in self.cameras:
            try:
                cam.cap.release()
            except:
                pass
        if self.debugger:
            self.debugger.close()
