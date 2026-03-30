import cv2
import numpy as np
import json
import os
import sys
import time

from vision_absdiff import AbsDiffDetector
from vision_takeout import TakeoutDetector
from vision_vector import VectorDetector
from vision_shape import ShapeDetector
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
            except Exception:
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

        pts2 = np.float32([
            [top_x, top_y],
            [right_x, right_y],
            [bot_x, bot_y],
            [left_x, left_y]
        ])

        self.H = cv2.getPerspectiveTransform(pts1, pts2)
        try:
            self.invH = np.linalg.inv(self.H)
        except Exception:
            self.invH = None

    def warp_to_board(self, frame_bgr, size=600):
        if self.H is None:
            return None
        return cv2.warpPerspective(frame_bgr, self.H, (size, size))


class DartVisionSystem:
    def __init__(self, hit_callback):
        self.hit_callback = hit_callback
        self.cameras = [CameraHandler(i) for i in range(3)]

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

        self.abs_detectors = [
            AbsDiffDetector(self.board_mask, self.FREEZE_MEAN, self.FREEZE_MAX)
            for _ in range(3)
        ]
        self.shape_detectors = [
            ShapeDetector(self.board_mask, self.FREEZE_MEAN, self.FREEZE_MAX)
            for _ in range(3)
        ]
        self.vec_detectors = [
            VectorDetector()
            for _ in range(3)
        ]
        self.takeout_detectors = [
            TakeoutDetector(self.board_mask)
            for _ in range(3)
        ]

        self.debugger = VisionDebugger(warp_size=800)

        self.running = True
        self.last_hit_time = 0.0
        self.last_hit_board = None
        self.last_hit_contours = {}

        self.hit_candidate = None
        self.hit_candidate_time = 0.0

        self.WINKEL_OFFSET = 0

        self.takeout_empty_frames = 0
        self.takeout_empty_frames_required = 10
        self.board_empty_state = True

        self.hit_blocked_until = time.time() + 2.0

        # --- Neu: Konsolen-Debug ---
        self.console_debug = True

    def _dbg(self, msg):
        if self.console_debug:
            print(msg)

    def _block_hits(self, seconds=1.0):
        self.hit_blocked_until = max(self.hit_blocked_until, time.time() + seconds)
        self.hit_candidate = None
        self._dbg(f"[HIT BLOCK] Treffer gesperrt für {seconds:.2f}s bis {self.hit_blocked_until:.3f}")

    def _hits_allowed(self):
        return time.time() >= self.hit_blocked_until

    def run(self):
        print("[VISION] System bereit...")
        try:
            self.reset_references()

            while self.running:
                cam_hits = []
                board_is_moving = False

                tracking_active = self.last_hit_board is not None
                all_cameras_empty = tracking_active

                for idx, cam in enumerate(self.cameras):
                    ret, frame = cam.cap.read()
                    if not ret or frame is None or cam.H is None:
                        continue

                    warped = cam.warp_to_board(frame, 600)
                    if warped is None:
                        continue

                    gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

                    cam_is_moving = False
                    if cam.reference_gray is not None:
                        diff_motion = cv2.absdiff(gray_warped, cam.reference_gray)
                        mean_val = float(cv2.mean(diff_motion)[0])
                        _, max_val, _, _ = cv2.minMaxLoc(diff_motion)

                        if mean_val > self.FREEZE_MEAN and max_val < self.FREEZE_MAX:
                            cam_is_moving = True
                            board_is_moving = True
                            self._dbg(
                                f"[CAM {cam.cam_id}] BOARD_MOVING mean={mean_val:.2f} max={max_val:.2f}"
                            )

                    if tracking_active:
                        takeout_detected, _take_dbg = self.takeout_detectors[idx].check_takeout(
                            warped, self.last_hit_contours
                        )
                        self._dbg(
                            f"[CAM {cam.cam_id}] TAKEOUT empty={takeout_detected}"
                        )
                        if not takeout_detected:
                            all_cameras_empty = False

                    abs_objs, _abs_dbg = self.abs_detectors[idx].detect(warped, gray_warped)
                    shape_objs, _shape_dbg = self.shape_detectors[idx].detect(warped, gray_warped)

                    self._dbg(
                        f"[CAM {cam.cam_id}] COUNTS abs={len(abs_objs)} shape={len(shape_objs)}"
                    )

                    board_center_full = None
                    if cam.invH is not None:
                        board_center_full = pt_transform(cam.invH, (300.0, 300.0))

                    roi_mask_full = None
                    if cam.invH is not None:
                        full_h, full_w = frame.shape[:2]
                        roi_mask_full = cv2.warpPerspective(self.board_mask, cam.invH, (full_w, full_h))

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
                                "confidence": float(o.get("confidence", 0.0)),
                                "contour": None,
                                "extra": {
                                    "line_full": o.get("line", None),
                                    "length": float(o.get("length", 0.0))
                                }
                            })

                    self._dbg(
                        f"[CAM {cam.cam_id}] COUNTS vec={len(vec_objs)}"
                    )

                    best = self._select_best(abs_objs, shape_objs, vec_objs)

                    if best is not None:
                        bx, by = best["tip_board"]
                        self._dbg(
                            f"[CAM {cam.cam_id}] BEST src={best['src']} "
                            f"tip_board=({bx:.1f},{by:.1f}) "
                            f"conf={float(best['confidence']):.1f}"
                        )

                        tip_board = best["tip_board"]
                        tip_full = None
                        if cam.invH is not None:
                            tip_full = pt_transform(cam.invH, tip_board)

                        line_full = None
                        if "vec" in best["src"]:
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

                    if cam_is_moving:
                        continue

                    if self._hits_allowed() and best is not None and (time.time() - self.last_hit_time > 0.35):
                        if best.get("contour", None) is not None:
                            self.last_hit_contours[cam.cam_id] = best["contour"]

                        self._dbg(
                            f"[CAM {cam.cam_id} -> HIT_CAND] "
                            f"src={best['src']} "
                            f"tip_board=({best['tip_board'][0]:.1f},{best['tip_board'][1]:.1f}) "
                            f"conf={float(best['confidence']):.1f}"
                        )

                        cam_hits.append({
                            "cam_id": cam.cam_id,
                            "tip_board": best["tip_board"],
                            "confidence": float(best["confidence"]),
                            "src": best["src"]
                        })
                    elif best is not None and not self._hits_allowed():
                        self._dbg(
                            f"[CAM {cam.cam_id}] BEST IGNORIERT (HIT BLOCK aktiv)"
                        )

                if board_is_moving:
                    self.hit_candidate = None
                    self.takeout_empty_frames = 0
                    time.sleep(0.01)
                    continue

                if tracking_active:
                    if all_cameras_empty:
                        self.takeout_empty_frames += 1
                        self._dbg(
                            f"[TAKEOUT] empty_frames={self.takeout_empty_frames}/{self.takeout_empty_frames_required}"
                        )
                    else:
                        if self.takeout_empty_frames > 0:
                            self._dbg("[TAKEOUT] Reset empty_frames -> 0")
                        self.takeout_empty_frames = 0
                        self.board_empty_state = False

                    if (
                        self.takeout_empty_frames >= self.takeout_empty_frames_required
                        and not self.board_empty_state
                    ):
                        print("[INFO] Alle Darts entfernt.")
                        self.board_empty_state = True
                        self.takeout_empty_frames = 0

                        self.last_hit_board = None
                        self.last_hit_contours = {}
                        self.hit_candidate = None

                        self.reset_references()
                        self._block_hits(1.5)
                        self.hit_callback("NEXT_PLAYER")
                else:
                    self.takeout_empty_frames = 0
                    self.board_empty_state = True

                if self._hits_allowed() and len(cam_hits) >= 2:
                    fused = self._fuse_hits(cam_hits)
                    if fused is not None:
                        final_x, final_y, dist_ok = fused
                        if dist_ok:
                            current_point = (final_x, final_y)

                            if self.hit_candidate is None:
                                self.hit_candidate = current_point
                                self.hit_candidate_time = time.time()
                                self._dbg(
                                    f"[HIT CAND START] point=({final_x},{final_y})"
                                )
                            else:
                                dist_prev = float(
                                    np.linalg.norm(np.array(current_point) - np.array(self.hit_candidate))
                                )
                                self._dbg(
                                    f"[HIT CAND CHECK] prev=({self.hit_candidate[0]:.1f},{self.hit_candidate[1]:.1f}) "
                                    f"curr=({final_x},{final_y}) dist={dist_prev:.1f} "
                                    f"dt={(time.time() - self.hit_candidate_time):.3f}s"
                                )
                                if dist_prev < 18 and (time.time() - self.hit_candidate_time) < 0.25:
                                    self.hit_candidate = None
                                    self._emit_score(final_x, final_y)
                                else:
                                    self.hit_candidate = current_point
                                    self.hit_candidate_time = time.time()
                                    self._dbg(
                                        f"[HIT CAND UPDATE] point=({final_x},{final_y})"
                                    )
                        else:
                            self._dbg("[FUSION] dist_ok=False -> verworfen")

                time.sleep(0.001)

        except Exception as e:
            print(f"[VISION ERROR] {e}")

    def stop(self):
        self.running = False
        for cam in self.cameras:
            try:
                cam.cap.release()
            except Exception:
                pass
        try:
            self.debugger.close()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    def reset_references(self):
        self._dbg("[RESET REFERENCES] Starte Neuaufnahme der Referenzen")
        for idx, cam in enumerate(self.cameras):
            cam.load_config()
            cam.compute_homography()
            if cam.H is None:
                self._dbg(f"[RESET REFERENCES] Cam {cam.cam_id} hat keine Homography")
                continue

            for _ in range(12):
                cam.cap.read()

            ret, frame = cam.cap.read()
            if not ret or frame is None:
                self._dbg(f"[RESET REFERENCES] Cam {cam.cam_id} liefert kein Frame")
                continue

            warped = cam.warp_to_board(frame, 600)
            if warped is None:
                self._dbg(f"[RESET REFERENCES] Cam {cam.cam_id} warp fehlgeschlagen")
                continue

            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

            self.abs_detectors[idx].set_reference(warped)
            self.shape_detectors[idx].set_reference(warped)
            self.takeout_detectors[idx].set_clean_board(warped)
            cam.reference_gray = gray

            self._dbg(f"[RESET REFERENCES] Cam {cam.cam_id} Referenzen gesetzt")

        self._block_hits(2.0)

    def update_references(self):
        self._dbg("[UPDATE REFERENCES] Aktualisiere Referenzen nach Treffer")
        for idx, cam in enumerate(self.cameras):
            if cam.H is None:
                self._dbg(f"[UPDATE REFERENCES] Cam {cam.cam_id} hat keine Homography")
                continue

            for _ in range(8):
                cam.cap.read()

            ret, frame = cam.cap.read()
            if not ret or frame is None:
                self._dbg(f"[UPDATE REFERENCES] Cam {cam.cam_id} liefert kein Frame")
                continue

            warped = cam.warp_to_board(frame, 600)
            if warped is None:
                self._dbg(f"[UPDATE REFERENCES] Cam {cam.cam_id} warp fehlgeschlagen")
                continue

            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            self.abs_detectors[idx].set_reference(warped)
            self.shape_detectors[idx].set_reference(warped)
            cam.reference_gray = gray

            self._dbg(f"[UPDATE REFERENCES] Cam {cam.cam_id} Referenzen aktualisiert")

        self._block_hits(0.8)

    def _pick_best_obj(self, objs, source_name, conf_scale=1.0):
        if not objs:
            return None

        best = max(objs, key=lambda o: float(o.get("confidence", 0.0)))
        return {
            "src": source_name,
            "tip_board": best["tip_board"],
            "confidence": float(best.get("confidence", 0.0)) * conf_scale,
            "contour": best.get("contour", None),
            "extra": best.get("extra", {})
        }

    def _fuse_local_candidates(self, cands, merge_dist=20.0):
        if not cands:
            return None
        if len(cands) == 1:
            return cands[0]

        best_group = []
        for i, a in enumerate(cands):
            group = [a]
            for j, b in enumerate(cands):
                if i == j:
                    continue
                d = np.linalg.norm(np.array(a["tip_board"]) - np.array(b["tip_board"]))
                if d < merge_dist:
                    group.append(b)

            if len(group) > len(best_group):
                best_group = group
            elif len(group) == len(best_group):
                if sum(g["confidence"] for g in group) > sum(g["confidence"] for g in best_group):
                    best_group = group

        if len(best_group) == 1:
            best_single = max(cands, key=lambda x: x["confidence"])
            self._dbg(
                f"[LOCAL FUSION] Single src={best_single['src']} "
                f"tip=({best_single['tip_board'][0]:.1f},{best_single['tip_board'][1]:.1f}) "
                f"conf={best_single['confidence']:.1f}"
            )
            return best_single

        weights = np.array([g["confidence"] for g in best_group], dtype=np.float32)
        pts = np.array([g["tip_board"] for g in best_group], dtype=np.float32)
        fused = np.average(pts, axis=0, weights=weights)

        best_contour_src = max(best_group, key=lambda x: x["confidence"])
        src_name = "+".join(sorted(set(g["src"] for g in best_group)))

        self._dbg(
            "[LOCAL FUSION] "
            + " | ".join(
                [
                    f"src={g['src']} tip=({g['tip_board'][0]:.1f},{g['tip_board'][1]:.1f}) conf={g['confidence']:.1f}"
                    for g in best_group
                ]
            )
            + f" || fused=({fused[0]:.1f},{fused[1]:.1f}) src={src_name}"
        )

        return {
            "src": src_name,
            "tip_board": (float(fused[0]), float(fused[1])),
            "confidence": float(np.sum(weights)),
            "contour": best_contour_src.get("contour", None),
            "extra": best_contour_src.get("extra", {})
        }

    def _select_best(self, abs_objs, shape_objs, vec_objs):
        cands = []

        best_abs = self._pick_best_obj(abs_objs, "abs", conf_scale=1.0)
        best_shape = self._pick_best_obj(shape_objs, "shape", conf_scale=1.0)
        best_vec = self._pick_best_obj(vec_objs, "vec", conf_scale=0.45)

        # vec nie alleine akzeptieren
        if best_vec is not None and best_abs is None and best_shape is None:
            best_vec = None

        if best_abs is not None:
            cands.append(best_abs)
        if best_shape is not None:
            cands.append(best_shape)
        if best_vec is not None:
            cands.append(best_vec)

        if not cands:
            return None

        return self._fuse_local_candidates(cands, merge_dist=20.0)

    def _fuse_hits(self, cam_hits):
        cam_hits = sorted(cam_hits, key=lambda d: d["confidence"], reverse=True)

        if len(cam_hits) < 2:
            return None

        p1 = np.array(cam_hits[0]["tip_board"], dtype=np.float32)
        p2 = np.array(cam_hits[1]["tip_board"], dtype=np.float32)
        dist = float(np.linalg.norm(p1 - p2))
        dist_ok = dist < 65

        top = cam_hits[:3]
        weights = np.array([min(h["confidence"], 1e6) for h in top], dtype=np.float32)
        pts = np.array([h["tip_board"] for h in top], dtype=np.float32)

        if float(np.sum(weights)) <= 1e-6:
            return None

        final = np.average(pts, axis=0, weights=weights)

        self._dbg(
            "[FUSION] "
            + " | ".join(
                [
                    f"cam={h['cam_id']} src={h['src']} "
                    f"tip=({h['tip_board'][0]:.1f},{h['tip_board'][1]:.1f}) "
                    f"conf={h['confidence']:.1f}"
                    for h in top
                ]
            )
            + f" || final=({final[0]:.1f},{final[1]:.1f}) dist12={dist:.1f} dist_ok={dist_ok}"
        )

        return (int(final[0]), int(final[1]), dist_ok)

    def _emit_score(self, bx, by):
        if not self._hits_allowed():
            self._dbg("[EMIT] verworfen, weil HIT BLOCK aktiv")
            return

        if self.last_hit_board is not None:
            od = float(np.linalg.norm(np.array((bx, by)) - np.array(self.last_hit_board)))
            if od < 18:
                print("[DEBUG] Punkt zu nah am alten - verworfen")
                self._dbg(
                    f"[EMIT] Abstand zu last_hit_board=({self.last_hit_board[0]:.1f},{self.last_hit_board[1]:.1f}) "
                    f"ist nur {od:.1f}"
                )
                return

        score_dict = self._score_from_board(bx, by)

        self.last_hit_board = (bx, by)
        self.last_hit_time = time.time()
        self.board_empty_state = False
        self.takeout_empty_frames = 0

        self._dbg(
            f"[EMIT] board=({bx:.1f},{by:.1f}) "
            f"sector={score_dict.get('sector')} "
            f"ring={score_dict.get('ring', '-')} "
            f"missed={score_dict.get('is_missed', False)}"
        )

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