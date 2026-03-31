import cv2
import numpy as np
import json
import os
import sys
import time
import threading
from dataclasses import dataclass, field
from collections import deque

from vision_absdiff import AbsDiffDetector, fuse_warped_masks
from vision_debug import VisionDebugger


def get_external_path(filename: str) -> str:
    """
    Liefert einen absoluten Pfad zu einer externen Datei.
    """
    if getattr(sys, "frozen", False):
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, filename)


def pt_transform(H: np.ndarray, pt):
    """
    Transformiert einen einzelnen 2D-Punkt mit einer Homographie-Matrix.
    """
    p = np.array([[pt]], dtype=np.float32)
    out = cv2.perspectiveTransform(p, H)[0][0]
    return (float(out[0]), float(out[1]))


@dataclass
class VisionConfig:
    """
    Zentrale Konfiguration für das Vision-System.
    """

    # -----------------------------------------
    # Board / Warp
    # -----------------------------------------
    warp_size: int = 600
    board_usage_factor: float = 0.70
    board_extra_radius_mm: float = 55.0
    double_outer_radius_mm: float = 170.0

    # Homographie-Zielpunkte relativ zum Double-Kreis
    board_anchor_a: float = 0.156
    board_anchor_b: float = 0.987

    # -----------------------------------------
    # Kamera
    # -----------------------------------------
    cam_indices: tuple = (0, 1, 2)
    capture_backend = cv2.CAP_DSHOW
    frame_width: int = 1920
    frame_height: int = 1080
    fps: int = 30

    # Einheitliche fixe Kamera-Parameter
    auto_exposure: float = 1
    exposure: float = -7
    gain: float = 10
    brightness: float = 130

    camera_warmup_sec: float = 0.35
    camera_open_retries: int = 3
    camera_retry_sleep_sec: float = 0.15

    # -----------------------------------------
    # Timing / Loop
    # -----------------------------------------
    loop_idle_sleep: float = 0.0007
    no_frame_sleep: float = 0.002
    max_frame_age_sec: float = 0.25

    # -----------------------------------------
    # Hit-Tracking
    # -----------------------------------------
    hit_initial_block_sec: float = 0.8
    hit_confirm_window_sec: float = 0.55
    hit_confirm_dist_px: float = 18.0
    hit_min_confirm_count: int = 2
    hit_cooldown_sec: float = 0.25
    last_hit_reject_dist_px: float = 18.0

    # -----------------------------------------
    # Referenz-Management
    # -----------------------------------------
    post_hit_reference_guard_sec: float = 0.70
    required_stable_frames_for_update: int = 12
    required_stable_frames_after_reset: int = 6
    reference_update_min_wait_sec: float = 0.15
    reference_blend_frames: int = 3
    reference_post_update_block_sec: float = 0.12

    # -----------------------------------------
    # Stabilität / Bewegung
    # -----------------------------------------
    stability_baseline_window: int = 50
    stability_bootstrap_frames: int = 12

    # feste Untergrenzen
    stability_min_mean_threshold: float = 9.0
    stability_min_max_threshold: float = 36.0

    # adaptive Schwellen = baseline + Faktor * std
    stability_mean_std_factor: float = 3.0
    stability_max_std_factor: float = 3.0

    # Heuristik: mean hoch, max nicht extrem hoch => globales Wackeln / Drift
    stability_global_motion_peak_cap_factor: float = 1.65

    # -----------------------------------------
    # Fusion
    # -----------------------------------------
    mask_consensus_max_dist: float = 22.0
    allow_single_cam_tracking_fallback: bool = True
    single_cam_fallback_min_conf: float = 220.0

    # -----------------------------------------
    # Detector-Parameter (für alle Kameras gleich)
    # -----------------------------------------
    detector_freeze_mean: float = 20.0
    detector_freeze_max: float = 70.0
    detector_min_area: float = 140.0
    detector_max_area: float = 18000.0
    detector_min_length: float = 12.0
    detector_merge_dist: float = 20.0
    detector_max_width: float = 54.0
    detector_min_slenderness: float = 1.15
    detector_min_aspect: float = 1.02
    detector_color_diff_threshold: float = 22.0
    detector_grad_diff_threshold: float = 18.0
    detector_min_foreground_ratio: float = 0.00008
    detector_max_foreground_ratio: float = 0.12
    detector_use_virtual_greenscreen: bool = True

    # -----------------------------------------
    # Logging / Debug
    # -----------------------------------------
    console_debug: bool = True
    log_camera_open: bool = True
    log_stability_changes_only: bool = True


@dataclass
class CandidateTrack:
    """
    Verwaltet einen temporären Trefferkandidaten über mehrere Loops.
    """
    point: tuple
    first_seen: float
    last_seen: float
    count: int = 1
    best_confidence: float = 0.0
    source: str = ""
    used_cams: tuple = field(default_factory=tuple)

    def update(self, point, now, confidence=0.0, source="", used_cams=()):
        self.point = point
        self.last_seen = now
        self.count += 1
        self.best_confidence = max(self.best_confidence, float(confidence))
        if source:
            self.source = source
        if used_cams:
            self.used_cams = tuple(used_cams)

    def age(self, now):
        return now - self.first_seen

    def since_last_seen(self, now):
        return now - self.last_seen


class AdaptiveStabilityModel:
    """
    Laufende Baseline für Stabilitätsmetriken einer Kamera.
    Nutzt nur ruhige Frames zur Baseline-Aktualisierung.
    """

    def __init__(self, window_size=50):
        self.window_size = int(max(5, window_size))
        self.mean_hist = deque(maxlen=self.window_size)
        self.max_hist = deque(maxlen=self.window_size)

    def ready(self):
        return len(self.mean_hist) >= 5 and len(self.max_hist) >= 5

    def push(self, mean_val, max_val):
        self.mean_hist.append(float(mean_val))
        self.max_hist.append(float(max_val))

    def stats(self):
        if not self.ready():
            return None

        mean_mu = float(np.mean(self.mean_hist))
        mean_std = float(np.std(self.mean_hist))

        max_mu = float(np.mean(self.max_hist))
        max_std = float(np.std(self.max_hist))

        return {
            "mean_mu": mean_mu,
            "mean_std": mean_std,
            "max_mu": max_mu,
            "max_std": max_std,
        }


class CameraHandler:
    """
    Verwaltet eine einzelne Kamera inklusive:
    - Laden der Kamerakonfiguration
    - Berechnung der Homographie
    - Öffnen und Konfigurieren der Kamera
    - permanentes Einlesen der Frames in einem Hintergrund-Thread
    - thread-sicheres Bereitstellen des neuesten Frames
    """

    def __init__(self, cam_id: int, cfg: VisionConfig, dbg_func):
        self.cam_id = cam_id
        self.cfg = cfg
        self._dbg = dbg_func

        self.config_file = get_external_path(f"cam{cam_id}_config.json")

        self.src_points = []
        self.H = None
        self.invH = None

        self.reference_gray = None
        self.reference_lab = None

        self.load_config()
        self.compute_homography()

        self.cap = None
        self.open_camera()

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

        if not os.path.exists(self.config_file):
            self._dbg(f"[CAM {self.cam_id}] WARN config fehlt: {self.config_file}")
            return

        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            pts = data.get("points", [])
            if isinstance(pts, list) and len(pts) == 4:
                self.src_points = pts
            else:
                self._dbg(f"[CAM {self.cam_id}] WARN ungültige config points")
        except Exception as e:
            self._dbg(f"[CAM {self.cam_id}] ERROR config laden fehlgeschlagen: {e}")

    def compute_homography(self):
        if len(self.src_points) < 4:
            self.H = None
            self.invH = None
            return

        pts1 = np.float32(self.src_points)

        canvas_size = self.cfg.warp_size
        target_center = canvas_size / 2

        dist_double_px = (canvas_size / 2) * self.cfg.board_usage_factor
        a = self.cfg.board_anchor_a
        b = self.cfg.board_anchor_b

        top_x = target_center + dist_double_px * a
        top_y = target_center - dist_double_px * b

        right_x = target_center + dist_double_px * b
        right_y = target_center + dist_double_px * a

        bot_x = target_center - dist_double_px * a
        bot_y = target_center + dist_double_px * b

        left_x = target_center - dist_double_px * b
        left_y = target_center - dist_double_px * a

        pts2 = np.float32([
            [top_x, top_y],
            [right_x, right_y],
            [bot_x, bot_y],
            [left_x, left_y],
        ])

        self.H = cv2.getPerspectiveTransform(pts1, pts2)

        try:
            self.invH = np.linalg.inv(self.H)
        except Exception as e:
            self._dbg(f"[CAM {self.cam_id}] WARN inverse H fehlgeschlagen: {e}")
            self.invH = None

    def _apply_camera_settings(self):
        if self.cap is None:
            return

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, self.cfg.auto_exposure)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, self.cfg.exposure)
        self.cap.set(cv2.CAP_PROP_GAIN, self.cfg.gain)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.cfg.brightness)

    def open_camera(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                self._dbg(f"[CAM {self.cam_id}] WARN release vor reopen: {e}")

        self.cap = None

        for attempt in range(1, self.cfg.camera_open_retries + 1):
            try:
                cap = cv2.VideoCapture(self.cam_id, self.cfg.capture_backend)
                if cap is None or not cap.isOpened():
                    if cap is not None:
                        cap.release()
                    self._dbg(f"[CAM {self.cam_id}] WARN open fehlgeschlagen Versuch {attempt}")
                    time.sleep(self.cfg.camera_retry_sleep_sec)
                    continue

                self.cap = cap
                self._apply_camera_settings()

                t0 = time.time()
                got_frame = False
                while (time.time() - t0) < self.cfg.camera_warmup_sec:
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        got_frame = True
                        break
                    time.sleep(0.01)

                if not got_frame:
                    self._dbg(f"[CAM {self.cam_id}] WARN open ok, aber kein Warmup-Frame in Versuch {attempt}")
                else:
                    if self.cfg.log_camera_open:
                        self._dbg(f"[CAM {self.cam_id}] OPEN ok (Versuch {attempt})")
                return

            except Exception as e:
                self._dbg(f"[CAM {self.cam_id}] ERROR open Versuch {attempt}: {e}")
                time.sleep(self.cfg.camera_retry_sleep_sec)

        self._dbg(f"[CAM {self.cam_id}] ERROR Kamera konnte nicht geöffnet werden")

    def warp_to_board(self, frame_bgr, size=None):
        if self.H is None:
            return None
        if size is None:
            size = self.cfg.warp_size
        return cv2.warpPerspective(frame_bgr, self.H, (size, size))

    def _reader_loop(self):
        fail_count = 0

        while self.running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    self._dbg(f"[CAM {self.cam_id}] WARN cap nicht offen, reopen")
                    self.open_camera()
                    time.sleep(0.05)
                    continue

                ret, frame = self.cap.read()

                if ret and frame is not None:
                    with self.frame_lock:
                        self.latest_frame = frame
                        self.latest_ts = time.time()
                        self.frame_counter += 1
                    fail_count = 0
                else:
                    fail_count += 1
                    if fail_count in (10, 30, 60):
                        self._dbg(f"[CAM {self.cam_id}] WARN read fail_count={fail_count}")
                    if fail_count >= 60:
                        self._dbg(f"[CAM {self.cam_id}] WARN zu viele Read-Fehler, reopen")
                        self.open_camera()
                        fail_count = 0
                    time.sleep(0.01)

            except Exception as e:
                fail_count += 1
                self._dbg(f"[CAM {self.cam_id}] ERROR reader_loop: {e}")
                time.sleep(0.02)

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
                self.thread.join(timeout=0.6)
        except Exception as e:
            self._dbg(f"[CAM {self.cam_id}] WARN thread join: {e}")

        try:
            if self.cap is not None:
                self.cap.release()
        except Exception as e:
            self._dbg(f"[CAM {self.cam_id}] WARN cap release: {e}")


class DartVisionSystem:
    """
    Zentrales Vision-System zur Dart-Treffererkennung.
    """

    def __init__(self, hit_callback, config: VisionConfig | None = None):
        self.cfg = config or VisionConfig()
        self.hit_callback = hit_callback

        self.console_debug = self.cfg.console_debug

        self.running = True
        self.state_lock = threading.Lock()

        self.board_mask = np.zeros((self.cfg.warp_size, self.cfg.warp_size), dtype=np.uint8)

        total_radius_mm = self.cfg.double_outer_radius_mm + self.cfg.board_extra_radius_mm
        self.px_per_mm_calc = (
            (self.cfg.warp_size * self.cfg.board_usage_factor) / (total_radius_mm * 2.0)
        )

        valid_board_radius_px = int(total_radius_mm * self.px_per_mm_calc)
        cv2.circle(
            self.board_mask,
            (self.cfg.warp_size // 2, self.cfg.warp_size // 2),
            valid_board_radius_px,
            255,
            -1,
        )

        self.radii = {
            "bull": 6.35 * self.px_per_mm_calc,
            "single_bull": 15.9 * self.px_per_mm_calc,
            "triple_inner": 97.0 * self.px_per_mm_calc,
            "triple_outer": 107.0 * self.px_per_mm_calc,
            "double_inner": 160.0 * self.px_per_mm_calc,
            "double_outer": 170.0 * self.px_per_mm_calc,
        }

        self.cameras = [CameraHandler(i, self.cfg, self._dbg) for i in self.cfg.cam_indices]

        self.abs_detectors = [
            AbsDiffDetector(
                self.board_mask,
                freeze_mean=self.cfg.detector_freeze_mean,
                freeze_max=self.cfg.detector_freeze_max,
            )
            for _ in self.cameras
        ]
        self._apply_detector_config()

        self.debugger = VisionDebugger(warp_size=800)

        self.stability_models = [
            AdaptiveStabilityModel(window_size=self.cfg.stability_baseline_window)
            for _ in self.cameras
        ]

        self.last_hit_time = 0.0
        self.last_hit_board = None

        self.hit_track = None

        self.hit_blocked_until = time.time() + self.cfg.hit_initial_block_sec
        self.hit_cooldown_until = 0.0

        self.pending_reference_update = False
        self.pending_reference_started_at = 0.0
        self.post_hit_reference_earliest_at = 0.0

        self.stable_frames = 0
        self.required_stable_frames = self.cfg.required_stable_frames_for_update

        self.WINKEL_OFFSET = 0

        self.last_candidate_counts = [None] * len(self.cameras)
        self.last_reject_log = [None] * len(self.cameras)
        self.last_local_best_label = [None] * len(self.cameras)
        self.last_motion_summary = [None] * len(self.cameras)
        self.last_fusion_summary = None
        self.last_tracker_summary = None

    def _dbg(self, msg):
        if self.console_debug:
            print(msg)

    def _apply_detector_config(self):
        for det in self.abs_detectors:
            det.min_area = self.cfg.detector_min_area
            det.max_area = self.cfg.detector_max_area
            det.min_length = self.cfg.detector_min_length
            det.merge_dist = self.cfg.detector_merge_dist
            det.max_width = self.cfg.detector_max_width
            det.min_slenderness = self.cfg.detector_min_slenderness
            det.min_aspect = self.cfg.detector_min_aspect
            det.color_diff_threshold = self.cfg.detector_color_diff_threshold
            det.grad_diff_threshold = self.cfg.detector_grad_diff_threshold
            det.min_foreground_ratio = self.cfg.detector_min_foreground_ratio
            det.max_foreground_ratio = self.cfg.detector_max_foreground_ratio
            det.use_virtual_greenscreen = self.cfg.detector_use_virtual_greenscreen

    def _prepare_gray(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        return gray

    def _prepare_lab(self, frame_bgr):
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        lab = cv2.GaussianBlur(lab, (5, 5), 0)
        return lab

    def _store_camera_references(self, cam: CameraHandler, warped_bgr):
        cam.reference_gray = self._prepare_gray(warped_bgr)
        cam.reference_lab = self._prepare_lab(warped_bgr)

    def _blend_reference_frames(self, frames):
        valid = [f.astype(np.float32) for f in frames if f is not None]
        if not valid:
            return None
        if len(valid) == 1:
            return valid[0].astype(np.uint8)

        stacked = np.stack(valid, axis=0)
        med = np.median(stacked, axis=0)
        return med.astype(np.uint8)

    def _compute_board_stability_metrics(self, warped_bgr, cam: CameraHandler):
        if cam.reference_gray is None or cam.reference_lab is None:
            return None

        cur_gray = self._prepare_gray(warped_bgr)
        cur_lab = self._prepare_lab(warped_bgr)

        ref_gray = cam.reference_gray
        ref_lab = cam.reference_lab

        diff_gray = cv2.absdiff(cur_gray, ref_gray).astype(np.float32)

        diff_lab = cur_lab.astype(np.float32) - ref_lab.astype(np.float32)
        dL = diff_lab[..., 0]
        da = diff_lab[..., 1]
        db = diff_lab[..., 2]

        color_dist = np.sqrt((0.6 * dL) ** 2 + da ** 2 + db ** 2)

        valid = self.board_mask > 0
        if not np.any(valid):
            return None

        combined = 0.75 * color_dist + 0.25 * diff_gray
        vals = combined[valid]

        if vals.size == 0:
            return None

        return {
            "mean_val": float(np.mean(vals)),
            "max_val": float(np.max(vals)),
        }

    def _is_board_moving_from_metrics(self, cam_idx, metrics):
        if metrics is None:
            return False

        mean_val = metrics["mean_val"]
        max_val = metrics["max_val"]

        model = self.stability_models[cam_idx]
        stats = model.stats()

        if stats is None:
            mean_thr = self.cfg.stability_min_mean_threshold
            max_thr = self.cfg.stability_min_max_threshold
        else:
            mean_thr = max(
                self.cfg.stability_min_mean_threshold,
                stats["mean_mu"] + self.cfg.stability_mean_std_factor * max(0.5, stats["mean_std"])
            )
            max_thr = max(
                self.cfg.stability_min_max_threshold,
                stats["max_mu"] + self.cfg.stability_max_std_factor * max(1.0, stats["max_std"])
            )

        global_motion_peak_cap = max_thr * self.cfg.stability_global_motion_peak_cap_factor

        moving = (
            mean_val > mean_thr and
            max_val < global_motion_peak_cap
        )

        return moving

    def _maybe_update_stability_baseline(self, cam_idx, metrics, moving):
        if metrics is None or moving:
            return

        model = self.stability_models[cam_idx]
        model.push(metrics["mean_val"], metrics["max_val"])

    def _block_hits(self, seconds=1.0):
        with self.state_lock:
            until = time.time() + seconds
            self.hit_blocked_until = max(self.hit_blocked_until, until)
            self.hit_track = None
        self._dbg(f"[HIT BLOCK] Treffer gesperrt für {seconds:.2f}s bis {self.hit_blocked_until:.3f}")

    def _hits_allowed(self):
        now = time.time()
        return now >= self.hit_blocked_until and now >= self.hit_cooldown_until

    def _clear_hit_track(self):
        with self.state_lock:
            self.hit_track = None

    def _update_or_create_hit_track(self, point, now, confidence=0.0, source="", used_cams=()):
        with self.state_lock:
            if self.hit_track is None:
                self.hit_track = CandidateTrack(
                    point=point,
                    first_seen=now,
                    last_seen=now,
                    count=1,
                    best_confidence=float(confidence),
                    source=source,
                    used_cams=tuple(used_cams),
                )
                return "created"

            dist = float(np.linalg.norm(np.array(point) - np.array(self.hit_track.point)))
            age = self.hit_track.age(now)

            if dist <= self.cfg.hit_confirm_dist_px and age <= self.cfg.hit_confirm_window_sec:
                self.hit_track.update(
                    point=point,
                    now=now,
                    confidence=confidence,
                    source=source,
                    used_cams=used_cams,
                )
                return "updated"

            self.hit_track = CandidateTrack(
                point=point,
                first_seen=now,
                last_seen=now,
                count=1,
                best_confidence=float(confidence),
                source=source,
                used_cams=tuple(used_cams),
            )
            return "replaced"

    def _candidate_track_confirmed(self, now):
        with self.state_lock:
            tr = self.hit_track
            if tr is None:
                return None

            if tr.age(now) > self.cfg.hit_confirm_window_sec:
                self.hit_track = None
                return None

            if tr.count >= self.cfg.hit_min_confirm_count:
                point = tr.point
                self.hit_track = None
                return point

            return None

    def _track_single_cam_candidate(self, result, cam_idx, now):
        if result is None:
            return

        local_best = self._pick_best_candidate(result["candidates"], "abs")
        if local_best is None:
            return

        if float(local_best["confidence"]) < self.cfg.single_cam_fallback_min_conf:
            return

        bx, by = local_best["tip_board"]
        action = self._update_or_create_hit_track(
            point=(int(bx), int(by)),
            now=now,
            confidence=float(local_best["confidence"]),
            source=f"single_cam_{cam_idx}",
            used_cams=(cam_idx,),
        )

        summary = (
            action,
            cam_idx,
            round(bx, 1),
            round(by, 1),
            round(float(local_best["confidence"]), 1),
        )
        if summary != self.last_tracker_summary:
            self.last_tracker_summary = summary
            self._dbg(
                f"[TRACK] {action} single_cam={cam_idx} "
                f"point=({bx:.1f},{by:.1f}) conf={local_best['confidence']:.1f}"
            )

    def run(self):
        print("[VISION] System bereit...")

        try:
            self.reset_references()

            while self.running:
                now = time.time()

                board_is_moving = False
                any_frame_processed = False

                raw_frames = [None] * len(self.cameras)
                warped_frames = [None] * len(self.cameras)
                cam_moving = [False] * len(self.cameras)

                mask_results = [None] * len(self.cameras)
                mask_list = [None] * len(self.cameras)

                for idx, cam in enumerate(self.cameras):
                    if cam.H is None:
                        continue

                    frame, frame_ts, _counter = cam.get_latest_frame(only_new=True)
                    if frame is None:
                        continue

                    if frame_ts is not None and (now - frame_ts) > self.cfg.max_frame_age_sec:
                        continue

                    any_frame_processed = True
                    raw_frames[idx] = frame

                    warped = cam.warp_to_board(frame, self.cfg.warp_size)
                    if warped is None:
                        continue

                    warped_frames[idx] = warped

                    metrics = self._compute_board_stability_metrics(warped, cam)
                    moving = self._is_board_moving_from_metrics(idx, metrics)
                    cam_moving[idx] = moving
                    if moving:
                        board_is_moving = True

                    self._maybe_update_stability_baseline(idx, metrics, moving)

                    if metrics is not None:
                        motion_summary = (
                            round(metrics["mean_val"], 2),
                            round(metrics["max_val"], 2),
                            moving,
                        )
                        if self.last_motion_summary[idx] != motion_summary:
                            self.last_motion_summary[idx] = motion_summary
                            self._dbg(
                                f"[CAM {cam.cam_id}] STABILITY "
                                f"mean={metrics['mean_val']:.2f} "
                                f"max={metrics['max_val']:.2f} "
                                f"moving={moving}"
                            )

                    result = self.abs_detectors[idx].detect_mask_candidates(warped)
                    mask_results[idx] = result
                    mask_list[idx] = result["mask"]

                    cand_count = len(result["candidates"])

                    if self.last_candidate_counts[idx] != cand_count:
                        self.last_candidate_counts[idx] = cand_count
                        self._dbg(f"[CAM {cam.cam_id}] CANDIDATES abs={cand_count}")

                    if cand_count == 0:
                        reject_key = tuple(sorted(result["reject_stats"].items()))
                        meta_reason = result["meta"].get("reason") if result["meta"] else None
                        reject_signature = (reject_key, meta_reason)
                        if self.last_reject_log[idx] != reject_signature:
                            self.last_reject_log[idx] = reject_signature
                            self._dbg(
                                f"[CAM {cam.cam_id}] REJECTS {result['reject_stats']} "
                                f"meta_reason={meta_reason}"
                            )
                    else:
                        self.last_reject_log[idx] = None

                    local_best = self._pick_best_candidate(result["candidates"], "abs")
                    if local_best is not None:
                        bx, by = local_best["tip_board"]
                        label = self._format_score_label(bx, by)
                        summary = f"{label}@({bx:.1f},{by:.1f})"
                        if self.last_local_best_label[idx] != summary:
                            self.last_local_best_label[idx] = summary
                            self._dbg(
                                f"[CAM {cam.cam_id}] LOCAL best=({bx:.1f},{by:.1f}) "
                                f"field={label} conf={local_best['confidence']:.1f}"
                            )
                    else:
                        if self.last_local_best_label[idx] != "NONE":
                            self.last_local_best_label[idx] = "NONE"
                            self._dbg(f"[CAM {cam.cam_id}] LOCAL none")

                fused_hit = None
                active_masks = sum(
                    1 for m in mask_list if m is not None and cv2.countNonZero(m) > 0
                )

                if (
                    self._hits_allowed()
                    and not board_is_moving
                    and active_masks >= 2
                    and not any(cam_moving)
                ):
                    fused_hit = fuse_warped_masks(
                        mask_list=mask_list,
                        board_mask=self.board_mask,
                        max_dist=self.cfg.mask_consensus_max_dist,
                    )

                    if fused_hit is not None and (now - self.last_hit_time > 0.10):
                        final_x, final_y = fused_hit["tip_board"]
                        final_label = self._format_score_label(final_x, final_y)

                        fusion_summary = (
                            round(final_x, 1),
                            round(final_y, 1),
                            final_label,
                            tuple(fused_hit["used_cams"]),
                            tuple(round(d, 1) for d in fused_hit["per_cam_dist"]),
                        )
                        if self.last_fusion_summary != fusion_summary:
                            self.last_fusion_summary = fusion_summary
                            self._dbg(
                                "[FUSION] "
                                f"final=({final_x:.1f},{final_y:.1f}) "
                                f"field={final_label} "
                                f"used_cams={fused_hit['used_cams']} "
                                f"per_cam_dist={[round(d, 2) for d in fused_hit['per_cam_dist']]} "
                                f"score={fused_hit['score']:.2f}"
                            )

                        action = self._update_or_create_hit_track(
                            point=(int(final_x), int(final_y)),
                            now=now,
                            confidence=max(1.0, 10000.0 / max(1.0, fused_hit["score"])),
                            source="fusion",
                            used_cams=tuple(fused_hit["used_cams"]),
                        )

                        tracker_summary = (
                            action,
                            round(final_x, 1),
                            round(final_y, 1),
                            tuple(fused_hit["used_cams"]),
                        )
                        if tracker_summary != self.last_tracker_summary:
                            self.last_tracker_summary = tracker_summary
                            self._dbg(
                                f"[TRACK] {action} fusion "
                                f"point=({final_x:.1f},{final_y:.1f}) "
                                f"used_cams={fused_hit['used_cams']}"
                            )

                elif (
                    self.cfg.allow_single_cam_tracking_fallback
                    and self._hits_allowed()
                    and not board_is_moving
                ):
                    for idx, result in enumerate(mask_results):
                        self._track_single_cam_candidate(result, idx, now)

                confirmed_point = self._candidate_track_confirmed(now)
                if confirmed_point is not None and (now - self.last_hit_time) > 0.10:
                    self._emit_score(confirmed_point[0], confirmed_point[1])

                for idx, cam in enumerate(self.cameras):
                    frame = raw_frames[idx]
                    if frame is None:
                        continue

                    shown_tip = None
                    shown_method = None
                    shown_conf = None

                    if fused_hit is not None:
                        bx, by = fused_hit["tip_board"]
                        shown_tip = (bx, by)
                        shown_method = f"fusion | {self._format_score_label(bx, by)}"
                        shown_conf = max(1.0, 10000.0 / max(1.0, fused_hit["score"]))
                    else:
                        result = mask_results[idx]
                        if result is not None:
                            local_best = self._pick_best_candidate(result["candidates"], "abs")
                            if local_best is not None:
                                bx, by = local_best["tip_board"]
                                shown_tip = (bx, by)
                                shown_method = f"abs | {self._format_score_label(bx, by)}"
                                shown_conf = float(local_best["confidence"])

                    if shown_tip is not None:
                        bx, by = shown_tip
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
                            method=shown_method,
                            conf=shown_conf,
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
                            conf=None,
                        )

                if board_is_moving:
                    self._clear_hit_track()
                    self.stable_frames = 0
                else:
                    self.stable_frames += 1

                if self.pending_reference_update:
                    enough_stable = self.stable_frames >= self.required_stable_frames
                    min_wait_done = (now - self.pending_reference_started_at) >= self.cfg.reference_update_min_wait_sec
                    guard_done = now >= self.post_hit_reference_earliest_at

                    if enough_stable and min_wait_done and guard_done:
                        self.update_references_fast()
                        self.pending_reference_update = False
                        self._dbg("[REFRESH] Referenzen aktualisiert")

                if not any_frame_processed:
                    time.sleep(self.cfg.no_frame_sleep)
                else:
                    time.sleep(self.cfg.loop_idle_sleep)

        except Exception as e:
            print(f"[VISION ERROR] {e}")

    def stop(self):
        self.running = False

        for cam in self.cameras:
            try:
                cam.stop()
            except Exception as e:
                self._dbg(f"[VISION STOP] WARN cam.stop: {e}")

        try:
            self.debugger.close()
        except Exception as e:
            self._dbg(f"[VISION STOP] WARN debugger.close: {e}")

        try:
            cv2.destroyAllWindows()
        except Exception as e:
            self._dbg(f"[VISION STOP] WARN destroyAllWindows: {e}")

    def reset_references(self):
        self._dbg("[RESET REFERENCES] Starte Neuaufnahme der Referenzen")

        start = time.time()
        while time.time() - start < 0.6:
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
                self._dbg(f"[RESET REFERENCES] Cam {cam.cam_id} keine gültige Homographie")
                continue

            collected = []
            for _ in range(max(1, self.cfg.reference_blend_frames)):
                frame, _, _ = cam.get_latest_frame(only_new=False)
                if frame is not None:
                    warped = cam.warp_to_board(frame, self.cfg.warp_size)
                    if warped is not None:
                        collected.append(warped)
                time.sleep(0.01)

            blended = self._blend_reference_frames(collected)
            if blended is None:
                self._dbg(f"[RESET REFERENCES] Cam {cam.cam_id} kein Referenzframe")
                continue

            self.abs_detectors[idx].set_reference(blended)
            self._store_camera_references(cam, blended)

            self.stability_models[idx] = AdaptiveStabilityModel(
                window_size=self.cfg.stability_baseline_window
            )

            self.last_candidate_counts[idx] = None
            self.last_reject_log[idx] = None
            self.last_local_best_label[idx] = None
            self.last_motion_summary[idx] = None

            self._dbg(f"[RESET REFERENCES] Cam {cam.cam_id} Referenzen gesetzt")

        self.pending_reference_update = False
        self.stable_frames = 0
        self.required_stable_frames = self.cfg.required_stable_frames_after_reset

        self._block_hits(0.6)

    def update_references_fast(self):
        for idx, cam in enumerate(self.cameras):
            if cam.H is None:
                continue

            collected = []
            for _ in range(max(1, self.cfg.reference_blend_frames)):
                frame, _, _ = cam.get_latest_frame(only_new=False)
                if frame is not None:
                    warped = cam.warp_to_board(frame, self.cfg.warp_size)
                    if warped is not None:
                        collected.append(warped)
                time.sleep(0.01)

            blended = self._blend_reference_frames(collected)
            if blended is None:
                self._dbg(f"[REFRESH] Cam {cam.cam_id} kein Update-Frame")
                continue

            self.abs_detectors[idx].set_reference(blended)
            self._store_camera_references(cam, blended)

            self.last_candidate_counts[idx] = None
            self.last_reject_log[idx] = None
            self.last_local_best_label[idx] = None
            self.last_motion_summary[idx] = None

        self.stable_frames = 0
        self.required_stable_frames = self.cfg.required_stable_frames_for_update
        self._block_hits(self.cfg.reference_post_update_block_sec)

    def _schedule_reference_update(self):
        with self.state_lock:
            self.pending_reference_update = True
            self.pending_reference_started_at = time.time()
            self.post_hit_reference_earliest_at = time.time() + self.cfg.post_hit_reference_guard_sec
            self.stable_frames = 0
            self.required_stable_frames = self.cfg.required_stable_frames_for_update

    def _pick_best_candidate(self, objs, source_name, conf_scale=1.0):
        if not objs:
            return None

        best = max(objs, key=lambda o: float(o.get("confidence", 0.0)))
        return {
            "src": source_name,
            "tip_board": best["tip_board"],
            "confidence": float(best.get("confidence", 0.0)) * conf_scale,
            "contour": best.get("contour", None),
            "extra": best.get("extra", {}),
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
            dist_last = np.linalg.norm(np.array((bx, by)) - np.array(self.last_hit_board))
            if dist_last < self.cfg.last_hit_reject_dist_px:
                self._dbg("[EMIT] Verworfen, zu nah am letzten Punkt")
                return

        score_dict = self._score_from_board(bx, by)

        self.last_hit_board = (bx, by)
        self.last_hit_time = time.time()

        self._dbg(
            f"[EMIT] board=({bx:.1f},{by:.1f}) "
            f"sector={score_dict.get('sector')} ring={score_dict.get('ring', '-')}"
        )

        self.hit_callback(score_dict)

        with self.state_lock:
            self.hit_track = None
            self.hit_cooldown_until = time.time() + self.cfg.hit_cooldown_sec

        self._schedule_reference_update()

    def _score_from_board(self, x, y):
        center = self.cfg.warp_size / 2.0
        rel_x, rel_y = x - center, y - center
        dist = float(np.linalg.norm([rel_x, rel_y]))

        if dist > float(self.radii["double_outer"]):
            return {
                "sector": 0,
                "is_missed": True,
                "board_x": float(x),
                "board_y": float(y),
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
            "board_y": float(y),
        }
