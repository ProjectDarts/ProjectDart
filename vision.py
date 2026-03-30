import cv2
import numpy as np
import json
import os
import sys
import time
import threading

from vision_absdiff import AbsDiffDetector, fuse_three_cameras
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
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -7)
        self.cap.set(cv2.CAP_PROP_GAIN, 10)

        if self.cam_id == 2:
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 100)
        else:
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)

        self.reference_gray = None

        self.running = True
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.latest_ts = 0.0
        self.frame_counter = 0
        self.last_consumed_counter = -1

        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

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

    def _reader_loop(self):
        fail_count = 0
        while self.running:
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self.frame_lock:
                        self.latest_frame = frame
                        self.latest_ts = time.time()
                        self.frame_counter += 1
                    fail_count = 0
                else:
                    fail_count += 1
                    if fail_count > 20:
                        time.sleep(0.01)
            except Exception:
                fail_count += 1
                time.sleep(0.01)

    def get_latest_frame(self, only_new=False):
        with self.frame_lock:
            if self.latest_frame is None:
                return None, None, None

            if only_new and self.frame_counter == self.last_consumed_counter:
                return None, None, None

            frame = self.latest_frame.copy()
            ts = self.latest_ts
            counter = self.frame_counter
            self.last_consumed_counter = counter
            return frame, ts, counter

    def stop(self):
        self.running = False
        try:
            if self.thread.is_alive():
                self.thread.join(timeout=0.5)
        except Exception:
            pass
        try:
            self.cap.release()
        except Exception:
            pass


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

        self.debugger = VisionDebugger(warp_size=800)

        self.running = True
        self.console_debug = True

        self.last_hit_time = 0.0
        self.last_hit_board = None
        self.last_hit_contours = {}

        self.hit_candidate = None
        self.hit_candidate_time = 0.0

        self.WINKEL_OFFSET = 0

        self.hit_blocked_until = time.time() + 0.8
        self.hit_cooldown_until = 0.0

        self.pending_reference_update = False
        self.pending_reference_started_at = 0.0
        self.stable_frames = 0
        self.required_stable_frames = 5

        self.loop_idle_sleep = 0.0005
        self.max_frame_age_sec = 0.20

        self.max_fuse_pair_dist = 35.0

        self.last_candidate_counts = [None, None, None]
        self.last_cam_state = [None, None, None]
        self.last_local_best_label = [None, None, None]

    def _dbg(self, msg):
        if self.console_debug:
            print(msg)

    def _log_cam_error_once(self, idx, msg):
        if self.last_cam_state[idx] != msg:
            self.last_cam_state[idx] = msg
            self._dbg(f"[CAM {idx}] {msg}")

    def _block_hits(self, seconds=1.0):
        until = time.time() + seconds
        self.hit_blocked_until = max(self.hit_blocked_until, until)
        self.hit_candidate = None
        self._dbg(f"[HIT BLOCK] Treffer gesperrt für {seconds:.2f}s bis {self.hit_blocked_until:.3f}")

    def _hits_allowed(self):
        now = time.time()
        return now >= self.hit_blocked_until and now >= self.hit_cooldown_until

    def run(self):
        print("[VISION] System bereit...")
        try:
            self.reset_references()

            while self.running:
                now = time.time()
                board_is_moving = False
                any_frame_processed = False

                warped_frames = [None, None, None]
                raw_frames = [None, None, None]
                cam_moving = [False, False, False]
                cam_candidates = [[], [], []]
                local_best = [None, None, None]
                display_hits = [None, None, None]

                for idx, cam in enumerate(self.cameras):
                    if cam.H is None:
                        self._log_cam_error_once(idx, "KEIN_HOMOGRAPHY")
                        continue
                    
                    frame, frame_ts, _counter = cam.get_latest_frame(only_new=True)
                    if frame is None:
                        # normaler Fall: gerade noch kein neuer Frame da
                        continue

                    if frame_ts is not None and (now - frame_ts) > self.max_frame_age_sec:
                        self._log_cam_error_once(idx, f"FRAME_TOO_OLD age={(now - frame_ts):.3f}s")
                        continue

                    # sobald wieder alles OK ist, Error-State zurücksetzen
                    self.last_cam_state[idx] = None
                    any_frame_processed = True
                    raw_frames[idx] = frame

                    warped = cam.warp_to_board(frame, 600)
                    if warped is None:
                        self._log_cam_state_once(idx, "WARP_FAILED")
                        continue
                    warped_frames[idx] = warped

                    gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

                    if cam.reference_gray is not None:
                        diff_motion = cv2.absdiff(gray_warped, cam.reference_gray)
                        mean_val = float(cv2.mean(diff_motion)[0])
                        _, max_val, _, _ = cv2.minMaxLoc(diff_motion)

                        if mean_val > self.FREEZE_MEAN and max_val < self.FREEZE_MAX:
                            cam_moving[idx] = True
                            board_is_moving = True
                            self._dbg(
                                f"[CAM {cam.cam_id}] BOARD_MOVING mean={mean_val:.2f} max={max_val:.2f}"
                            )

                    candidates, _abs_dbg = self.abs_detectors[idx].detect_candidates(warped, gray_warped)
                    cam_candidates[idx] = candidates

                    cand_count = len(candidates)
                    if self.last_candidate_counts[idx] != cand_count:
                        self.last_candidate_counts[idx] = cand_count
                        self._dbg(f"[CAM {cam.cam_id}] CANDIDATES abs={cand_count}")

                    local_best[idx] = self._pick_best_candidate(candidates, "abs")

                    if local_best[idx] is not None:
                        bx, by = local_best[idx]["tip_board"]
                        label = self._format_score_label(bx, by)
                        summary = f"{label}@({bx:.1f},{by:.1f})"
                        if self.last_local_best_label[idx] != summary:
                            self.last_local_best_label[idx] = summary
                            self._dbg(
                                f"[CAM {cam.cam_id}] LOCAL best=({bx:.1f},{by:.1f}) "
                                f"field={label} conf={local_best[idx]['confidence']:.1f}"
                            )
                    else:
                        if self.last_local_best_label[idx] != "NONE":
                            self.last_local_best_label[idx] = "NONE"
                            self._dbg(f"[CAM {cam.cam_id}] LOCAL none")

                fused_hit = None

                if (
                    self._hits_allowed()
                    and not board_is_moving
                    and all(warped_frames[i] is not None for i in range(3))
                    and not any(cam_moving)
                ):
                    fused_hit = fuse_three_cameras(
                        cam_candidates[0],
                        cam_candidates[1],
                        cam_candidates[2],
                        max_pair_dist=self.max_fuse_pair_dist
                    )

                    if fused_hit is not None and (now - self.last_hit_time > 0.12):
                        final_x, final_y = fused_hit["tip_board"]

                        self._dbg(
                            "[FUSION] "
                            f"final=({final_x:.1f},{final_y:.1f}) "
                            f"field={self._format_score_label(final_x, final_y)} "
                            f"cluster_score={fused_hit['cluster_score']:.2f} "
                            f"max_pair_dist={fused_hit['max_pair_dist']:.2f}"
                        )

                        for cam_idx, cand in enumerate(fused_hit["per_camera"]):
                            bx, by = cand["tip_board"]
                            label = self._format_score_label(bx, by)
                            display_hits[cam_idx] = {
                                "tip_board": (bx, by),
                                "confidence": float(cand["confidence"]),
                                "method": f"abs | {label}"
                            }

                            self._dbg(
                                f"  [CAM {cam_idx}] tip=({bx:.1f},{by:.1f}) "
                                f"field={label} conf={cand['confidence']:.1f} "
                                f"side={cand.get('extra', {}).get('endpoint_side', '?')}"
                            )

                        current_point = (int(final_x), int(final_y))

                        if self.hit_candidate is None:
                            self.hit_candidate = current_point
                            self.hit_candidate_time = now
                            self._dbg(f"[HIT CAND START] point=({current_point[0]},{current_point[1]})")
                        else:
                            dist_prev = float(
                                np.linalg.norm(np.array(current_point) - np.array(self.hit_candidate))
                            )
                            age = now - self.hit_candidate_time

                            if dist_prev < 18 and age < 0.22:
                                self.hit_candidate = None
                                self._emit_score(current_point[0], current_point[1])
                            else:
                                self.hit_candidate = current_point
                                self.hit_candidate_time = now
                                self._dbg(
                                    f"[HIT CAND UPDATE] point=({current_point[0]},{current_point[1]}) "
                                    f"dist_prev={dist_prev:.1f}"
                                )

                for idx, cam in enumerate(self.cameras):
                    frame = raw_frames[idx]
                    if frame is None:
                        continue

                    shown = display_hits[idx]

                    if shown is None and local_best[idx] is not None:
                        bx, by = local_best[idx]["tip_board"]
                        label = self._format_score_label(bx, by)
                        shown = {
                            "tip_board": (bx, by),
                            "confidence": float(local_best[idx]["confidence"]),
                            "method": f"{local_best[idx]['src']} | {label}"
                        }

                    if shown is not None:
                        bx, by = shown["tip_board"]
                        tip_full = None
                        if cam.invH is not None:
                            tip_full = pt_transform(cam.invH, (bx, by))

                        self.debugger.show(
                            cam_id=cam.cam_id,
                            frame_bgr=frame,
                            H_cam_to_board=cam.H,
                            tip_full=tip_full,
                            tip_board=(bx, by),
                            line_full=None,
                            method=shown["method"],
                            conf=float(shown["confidence"])
                        )
                    else:
                        self.debugger.show(
                            cam_id=cam.cam_id,
                            frame_bgr=frame,
                            H_cam_to_board=cam.H,
                            tip_full=None,
                            tip_board=None,
                            line_full=None,
                            method=None,
                            conf=None
                        )

                if board_is_moving:
                    self.hit_candidate = None
                    self.stable_frames = 0
                else:
                    self.stable_frames += 1

                if self.pending_reference_update:
                    enough_stable = self.stable_frames >= self.required_stable_frames
                    min_wait_done = (now - self.pending_reference_started_at) >= 0.05
                    if enough_stable and min_wait_done:
                        self.update_references_fast()
                        self.pending_reference_update = False
                        self._dbg("[REFRESH] Referenzen aktualisiert")

                if not any_frame_processed:
                    time.sleep(0.002)
                else:
                    time.sleep(self.loop_idle_sleep)

        except Exception as e:
            print(f"[VISION ERROR] {e}")

    def stop(self):
        self.running = False
        for cam in self.cameras:
            try:
                cam.stop()
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

        start = time.time()
        while time.time() - start < 0.3:
            all_ready = True
            for cam in self.cameras:
                frame, _, _ = cam.get_latest_frame(only_new=False)
                if frame is None:
                    all_ready = False
                    break
            if all_ready:
                break
            time.sleep(0.01)

        for idx, cam in enumerate(self.cameras):
            cam.load_config()
            cam.compute_homography()

            if cam.H is None:
                self._dbg(f"[RESET REFERENCES] Cam {cam.cam_id} übersprungen: H=None / config ungültig")
                continue

            frame, _, _ = cam.get_latest_frame(only_new=False)
            if frame is None:
                self._dbg(f"[RESET REFERENCES] Cam {cam.cam_id} übersprungen: kein Frame")
                continue

            warped = cam.warp_to_board(frame, 600)
            if warped is None:
                self._dbg(f"[RESET REFERENCES] Cam {cam.cam_id} übersprungen: warp fehlgeschlagen")
                continue

            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            self.abs_detectors[idx].set_reference(warped)
            cam.reference_gray = gray
            self.last_candidate_counts[idx] = None
            self.last_cam_state[idx] = None
            self.last_local_best_label[idx] = None
            self._dbg(f"[RESET REFERENCES] Cam {cam.cam_id} Referenzen gesetzt")

        self.pending_reference_update = False
        self.stable_frames = 0
        self._block_hits(0.5)

    def update_references_fast(self):
        for idx, cam in enumerate(self.cameras):
            if cam.H is None:
                continue

            frame, _, _ = cam.get_latest_frame(only_new=False)
            if frame is None:
                continue

            warped = cam.warp_to_board(frame, 600)
            if warped is None:
                continue

            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            self.abs_detectors[idx].set_reference(warped)
            cam.reference_gray = gray
            self.last_candidate_counts[idx] = None
            self.last_local_best_label[idx] = None

        self.stable_frames = 0
        self._block_hits(0.10)

    def _schedule_reference_update(self):
        self.pending_reference_update = True
        self.pending_reference_started_at = time.time()
        self.stable_frames = 0

    def _pick_best_candidate(self, objs, source_name, conf_scale=1.0):
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

    def _format_score_label(self, x, y):
        s = self._score_from_board(x, y)

        if s.get("is_missed", False):
            return "Miss"

        ring = s.get("ring", "single")
        sector = s.get("sector", 0)

        if ring == "bull":
            return "Bull"
        if ring == "single_bull":
            return "25"
        if ring == "double":
            return f"Double {sector}"
        if ring == "triple":
            return f"Triple {sector}"

        return f"Single {sector}"

    def _emit_score(self, bx, by):
        if self.last_hit_board is not None:
            if np.linalg.norm(np.array((bx, by)) - np.array(self.last_hit_board)) < 18:
                self._dbg("[EMIT] Verworfen, zu nah am letzten Punkt")
                return

        score_dict = self._score_from_board(bx, by)
        self.last_hit_board = (bx, by)
        self.last_hit_time = time.time()

        self._dbg(
            f"[EMIT] board=({bx:.1f},{by:.1f}) sector={score_dict.get('sector')} ring={score_dict.get('ring', '-')}"
        )

        self.hit_callback(score_dict)

        self.hit_candidate = None
        self.hit_cooldown_until = time.time() + 0.20
        self._schedule_reference_update()

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

        angle = (np.degrees(np.arctan2(-rel_y, rel_x)) + 360 + self.WINKEL_OFFSET) % 360
        segments = [6, 13, 4, 18, 1, 20, 5, 12, 9, 14, 11, 8, 16, 7, 19, 3, 17, 2, 15, 10]
        val = segments[int((angle + 9) / 18) % 20]

        ring = "single"
        if dist <= self.radii["bull"]:
            ring = "bull"
        elif dist <= self.radii["single_bull"]:
            ring = "single_bull"
        elif self.radii["triple_inner"] <= dist <= self.radii["triple_outer"]:
            ring = "triple"
        elif self.radii["double_inner"] <= dist <= self.radii["double_outer"]:
            ring = "double"

        return {
            "sector": int(val),
            "is_missed": False,
            "ring": ring,
            "board_x": float(x),
            "board_y": float(y)
        }