import cv2
import numpy as np
import json
import os
import sys
import time

from vision_absdiff import AbsDiffDetector
from vision_takeout import TakeoutDetector
from vision_vector import VectorDetector
from vision_debug import VisionDebugger


def get_external_path(filename: str) -> str:
    if getattr(sys, 'frozen', False):
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, filename)


def pt_transform(H: np.ndarray, pt):
    p = np.array([[pt]], dtype=np.float32)
    out = cv2.perspectiveTransform(p, H)[0][0]
    return (float(out[0]), float(out[1]))


def line_transform(H: np.ndarray, line):
    x1, y1, x2, y2 = line
    p1 = pt_transform(H, (x1, y1))
    p2 = pt_transform(H, (x2, y2))
    return (p1[0], p1[1], p2[0], p2[1])


class CameraHandler:
    def __init__(self, cam_id: int):
        self.cam_id = cam_id
        self.config_file = get_external_path(f"cam{cam_id}_config.json")
        self.src_points = []
        self.H = None
        self.invH = None

        self.load_config()
        self.compute_homography()

        self.cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        time.sleep(1.5)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -7)
        self.cap.set(cv2.CAP_PROP_GAIN, 10)
        if self.cam_id == 2:
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 100)
        else:
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)

        self.reference_gray = None

    def load_config(self):
        self.src_points = []
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    data = json.load(f)
                pts = data.get("points", [])
                if isinstance(pts, list) and len(pts) == 4:
                    self.src_points = pts
            except:
                print(f"[ERROR] Config für Cam {self.cam_id} konnte nicht geladen werden.")

    def compute_homography(self):
        if len(self.src_points) < 4:
            self.H = None
            self.invH = None
            return

        pts1 = np.float32(self.src_points)

        canvas_size = 600
        target_center = canvas_size / 2
        nutzungs_faktor = 0.70
        dist_double_px = (canvas_size / 2) * nutzungs_faktor

        top_x = target_center + dist_double_px * 0.156
        top_y = target_center - dist_double_px * 0.987
        right_x = target_center + dist_double_px * 0.987
        right_y = target_center + dist_double_px * 0.156
        bot_x = target_center - dist_double_px * 0.156
        bot_y = target_center + dist_double_px * 0.987
        left_x = target_center - dist_double_px * 0.987
        left_y = target_center - dist_double_px * 0.156

        pts2 = np.float32([[top_x, top_y], [right_x, right_y], [bot_x, bot_y], [left_x, left_y]])

        self.H = cv2.getPerspectiveTransform(pts1, pts2)
        try:
            self.invH = np.linalg.inv(self.H)
        except:
            self.invH = None

    def warp_to_board(self, frame_bgr, size=600):
        if self.H is None:
            return None
        return cv2.warpPerspective(frame_bgr, self.H, (size, size))


class DartVisionSystem:
    def __init__(self, hit_callback):
        self.hit_callback = hit_callback
        self.cameras = [CameraHandler(i) for i in range(3)]

        # Board mask + scale
        self.board_mask = np.zeros((600, 600), dtype=np.uint8)
        total_radius_mm = 170.0 + 55.0
        self.px_per_mm_calc = (600 * 0.70) / (total_radius_mm * 2)
        cv2.circle(self.board_mask, (300, 300), int(225 * self.px_per_mm_calc), 255, -1)

        self.radii = {
            "bull": 6.35 * self.px_per_mm_calc,
            "single_bull": 15.9 * self.px_per_mm_calc,
            "triple_inner": 97.0 * self.px_per_mm_calc,
            "triple_outer": 107.0 * self.px_per_mm_calc,
            "double_inner": 160.0 * self.px_per_mm_calc,
            "double_outer": 170.0 * self.px_per_mm_calc
        }

        self.FREEZE_MEAN = 20
        self.FREEZE_MAX = 70

        self.abs_detectors = [AbsDiffDetector(self.board_mask, self.FREEZE_MEAN, self.FREEZE_MAX) for _ in range(3)]
        self.vec_detectors = [VectorDetector() for _ in range(3)]
        self.takeout_detectors = [TakeoutDetector(self.board_mask) for _ in range(3)]

        self.debugger = VisionDebugger(warp_size=800)

        self.running = True
        self.last_hit_time = 0.0
        self.last_hit_board = None
        self.last_hit_contours = {}

        self.hit_candidate = None
        self.hit_candidate_time = 0.0

        self.WINKEL_OFFSET = 0

    def run(self):
        print("[VISION] System bereit...")
        try:
            self.reset_references()

            while self.running:
                cam_hits = []
                all_cameras_empty = True
                board_is_moving = False

                for idx, cam in enumerate(self.cameras):
                    ret, frame = cam.cap.read()
                    if not ret or frame is None or cam.H is None:
                        continue

                    warped = cam.warp_to_board(frame, 600)
                    if warped is None:
                        continue

                    gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

                    # Motion freeze (Boardspace)
                    if cam.reference_gray is not None:
                        diff_motion = cv2.absdiff(gray_warped, cam.reference_gray)
                        mean_val = float(cv2.mean(diff_motion)[0])
                        _, max_val, _, _ = cv2.minMaxLoc(diff_motion)
                        if mean_val > self.FREEZE_MEAN and max_val < self.FREEZE_MAX:
                            board_is_moving = True

                    # Takeout
                    takeout_detected, _take_dbg = self.takeout_detectors[idx].check_takeout(warped, self.last_hit_contours)
                    if not takeout_detected:
                        all_cameras_empty = False

                    # AbsDiff Kandidaten (Boardspace)
                    abs_objs, _abs_dbg = self.abs_detectors[idx].detect(warped, gray_warped)

                    # Vector Kandidaten (Fullframe) -> map to Board
                    h_full, w_full = frame.shape[:2]
                    board_center_full = None
                    if cam.invH is not None:
                        board_center_full = pt_transform(cam.invH, (300.0, 300.0))

                    # ROI Maske in Fullframe (nur zur Gewichtung, kein harter Crop)
                    roi_mask_full = None
                    if cam.invH is not None:
                        roi_mask_full = cv2.warpPerspective(self.board_mask, cam.invH, (w_full, h_full))

                    vec_objs_full = self.vec_detectors[idx].detect(
                        frame_bgr=frame,
                        board_center_full=board_center_full,
                        board_roi_mask_full=roi_mask_full
                    )

                    vec_objs = []
                    if cam.H is not None:
                        for o in vec_objs_full:
                            tip_full = o["tip"]
                            tip_board = pt_transform(cam.H, tip_full)
                            vec_objs.append({
                                "tip_board": tip_board,
                                "confidence": float(o["confidence"]),
                                "contour": None,
                                "extra": {
                                    "line_full": o.get("line", None),
                                    "length": float(o.get("length", 0.0))
                                }
                            })

                    best = self._select_best(abs_objs, vec_objs)

                    # Debug: show best per cam (wenn enabled)
                    if best is not None:
                        tip_board = best["tip_board"]
                        tip_full = None
                        if cam.invH is not None:
                            tip_full = pt_transform(cam.invH, tip_board)

                        line_full = None
                        if best["src"] == "vec":
                            line_full = best.get("extra", {}).get("line_full", None)

                        self.debugger.show(
                            cam_id=cam.cam_id,
                            frame_bgr=frame,
                            H_cam_to_board=cam.H,
                            tip_full=tip_full,
                            tip_board=tip_board,
                            line_full=line_full,
                            method=best["src"],
                            conf=float(best["confidence"])
                        )

                    # Wenn Board wackelt: keine Hits sammeln (aber Takeout läuft)
                    if board_is_moving:
                        continue

                    # Kandidaten sammeln
                    if best is not None and (time.time() - self.last_hit_time > 0.25):
                        if best.get("contour", None) is not None:
                            self.last_hit_contours[cam.cam_id] = best["contour"]

                        cam_hits.append({
                            "cam_id": cam.cam_id,
                            "tip_board": best["tip_board"],
                            "confidence": float(best["confidence"]),
                            "src": best["src"]
                        })

                # Bei Bewegung Candidate reset
                if board_is_moving:
                    self.hit_candidate = None
                    time.sleep(0.01)
                    continue

                # Takeout Event
                if all_cameras_empty and self.last_hit_board is not None:
                    print("[INFO] Alle Darts entfernt.")
                    self.last_hit_board = None
                    self.last_hit_contours = {}
                    self.hit_candidate = None
                    self.reset_references()
                    self.hit_callback("NEXT_PLAYER")

                # Hit Fusion
                if len(cam_hits) >= 2:
                    fused = self._fuse_hits(cam_hits)
                    if fused is not None:
                        final_x, final_y, dist_ok = fused
                        if dist_ok:
                            # Temporal verification
                            current_point = (final_x, final_y)
                            if self.hit_candidate is None:
                                self.hit_candidate = current_point
                                self.hit_candidate_time = time.time()
                            else:
                                dist_prev = float(np.linalg.norm(np.array(current_point) - np.array(self.hit_candidate)))
                                if dist_prev < 18 and (time.time() - self.hit_candidate_time) < 0.25:
                                    self.hit_candidate = None
                                    self._emit_score(final_x, final_y)
                                else:
                                    self.hit_candidate = current_point
                                    self.hit_candidate_time = time.time()

                time.sleep(0.001)

        except Exception as e:
            print(f"[VISION ERROR] {e}")

    def stop(self):
        self.running = False
        for cam in self.cameras:
            try:
                cam.cap.release()
            except:
                pass
        try:
            self.debugger.close()
        except:
            pass
        try:
            cv2.destroyAllWindows()
        except:
            pass

    def reset_references(self):
        for idx, cam in enumerate(self.cameras):
            cam.load_config()
            cam.compute_homography()
            if cam.H is None:
                continue

            for _ in range(10):
                cam.cap.read()

            ret, frame = cam.cap.read()
            if not ret or frame is None:
                continue

            warped = cam.warp_to_board(frame, 600)
            if warped is None:
                continue

            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

            self.abs_detectors[idx].set_reference(warped)
            self.takeout_detectors[idx].set_clean_board(warped)
            cam.reference_gray = gray

    def update_references(self):
        for idx, cam in enumerate(self.cameras):
            if cam.H is None:
                continue

            for _ in range(5):
                cam.cap.read()

            ret, frame = cam.cap.read()
            if not ret or frame is None:
                continue

            warped = cam.warp_to_board(frame, 600)
            if warped is None:
                continue

            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            self.abs_detectors[idx].set_reference(warped)
            cam.reference_gray = gray

    def _select_best(self, abs_objs, vec_objs):
        best = None
        best_score = -1.0

        # Abs Kandidaten
        for o in abs_objs:
            c = float(o.get("confidence", 0.0))
            if c > best_score:
                best_score = c
                best = {
                    "src": "abs",
                    "tip_board": o["tip_board"],
                    "confidence": c,
                    "contour": o.get("contour", None),
                    "extra": o.get("extra", {})
                }

        # Vector Kandidaten (Bias: nur wenn klar besser)
        for o in vec_objs:
            c = float(o.get("confidence", 0.0))
            if c > best_score * 1.15:
                best_score = c
                best = {
                    "src": "vec",
                    "tip_board": o["tip_board"],
                    "confidence": c,
                    "contour": None,
                    "extra": o.get("extra", {})
                }

        return best

    def _fuse_hits(self, cam_hits):
        cam_hits = sorted(cam_hits, key=lambda d: d["confidence"], reverse=True)

        p1 = np.array(cam_hits[0]["tip_board"], dtype=np.float32)
        p2 = np.array(cam_hits[1]["tip_board"], dtype=np.float32)
        dist = float(np.linalg.norm(p1 - p2))
        dist_ok = dist < 80

        top = cam_hits[:3]
        weights = np.array([min(h["confidence"], 1e6) for h in top], dtype=np.float32)
        pts = np.array([h["tip_board"] for h in top], dtype=np.float32)

        if float(np.sum(weights)) <= 1e-6:
            return None

        final = np.average(pts, axis=0, weights=weights)
        return (int(final[0]), int(final[1]), dist_ok)

    def _emit_score(self, bx, by):
        if self.last_hit_board is not None:
            od = float(np.linalg.norm(np.array((bx, by)) - np.array(self.last_hit_board)))
            if od < 18:
                print("[DEBUG] Punkt zu nah am alten - verworfen")
                return

        score_dict = self._score_from_board(bx, by)

        self.last_hit_board = (bx, by)
        self.last_hit_time = time.time()

        self.hit_callback(score_dict)

        time.sleep(0.35)
        self.update_references()

    def _score_from_board(self, x, y):
        rel_x, rel_y = x - 300, y - 300
        dist = float(np.linalg.norm([rel_x, rel_y]))

        if dist > float(self.radii["double_outer"]):
            return {
                "sector": 0,
                "is_missed": True,
                "board_x": float(x),
                "board_y": float(y)
            }

        angle = (np.degrees(np.arctan2(-rel_y, rel_x)) + 360) % 360
        angle = (angle + self.WINKEL_OFFSET) % 360

        segments = [6, 13, 4, 18, 1, 20, 5, 12, 9, 14, 11, 8, 16, 7, 19, 3, 17, 2, 15, 10]
        val = segments[int((angle + 9) / 18) % 20]

        ring = "single"
        if dist <= float(self.radii["bull"]):
            ring = "bull"
        elif dist <= float(self.radii["single_bull"]):
            ring = "single_bull"
        elif float(self.radii["triple_inner"]) <= dist <= float(self.radii["triple_outer"]):
            ring = "triple"
        elif float(self.radii["double_inner"]) <= dist <= float(self.radii["double_outer"]):
            ring = "double"

        return {
            "sector": int(val),
            "is_missed": False,
            "ring": ring,
            "board_x": float(x),
            "board_y": float(y)
        }
