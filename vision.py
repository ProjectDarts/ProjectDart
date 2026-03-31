import cv2
import numpy as np
import json
import os
import sys
import time
import threading

from vision_absdiff import AbsDiffDetector, fuse_warped_masks
from vision_debug import VisionDebugger


def get_external_path(filename: str) -> str:
    """
    Liefert einen absoluten Pfad zu einer externen Datei.

    Unterstützt zwei Betriebsarten:
    1. Normale Python-Ausführung:
       Pfad relativ zum Speicherort dieser .py-Datei
    2. Kompilierte EXE (z. B. PyInstaller):
       Pfad relativ zum Speicherort der ausführbaren Datei
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


class CameraHandler:
    """
    Verwaltet eine einzelne Kamera inklusive:
    - Laden der Kamerakonfiguration
    - Berechnung der Homographie
    - Öffnen und Konfigurieren der Kamera
    - permanentes Einlesen der Frames in einem Hintergrund-Thread
    - thread-sicheres Bereitstellen des neuesten Frames
    """

    def __init__(self, cam_id: int):
        self.cam_id = cam_id
        self.config_file = get_external_path(f"cam{cam_id}_config.json")

        self.src_points = []
        self.H = None
        self.invH = None

        self.load_config()
        self.compute_homography()

        self.cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)

        # Möglichst aktuelle Frames bevorzugen
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Einheitliche Kameraeinstellungen für alle Kameras
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # Für robuste Bildvergleiche möglichst wenig automatische Eingriffe
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -7)
        self.cap.set(cv2.CAP_PROP_GAIN, 10)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 130)

        # Referenzen für Board-Stabilitätsprüfung
        self.reference_gray = None
        self.reference_lab = None

        self.running = True
        self.frame_lock = threading.Lock()

        self.latest_frame = None
        self.latest_ts = 0.0

        self.frame_counter = 0
        self.last_consumed_counter = -1

        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

    def load_config(self):
        """
        Lädt die Kalibrierungspunkte der Kamera aus der JSON-Datei.
        Erwartet:
            { "points": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] }
        """
        self.src_points = []

        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                pts = data.get("points", [])
                if isinstance(pts, list) and len(pts) == 4:
                    self.src_points = pts
            except Exception:
                print(f"[ERROR] Config für Cam {self.cam_id} konnte nicht geladen werden.")

    def compute_homography(self):
        """
        Berechnet die Homographie Kamera -> normierte Boardansicht.
        """
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
        """
        Transformiert ein Kamerabild in die normierte Boardansicht.
        """
        if self.H is None:
            return None
        return cv2.warpPerspective(frame_bgr, self.H, (size, size))

    def _reader_loop(self):
        """
        Hintergrundthread zum permanenten Einlesen der Kamera.
        """
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
        """
        Gibt den aktuellsten Frame thread-sicher zurück.
        """
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
        """
        Stoppt den Kamerathread und gibt die Kameraressourcen frei.
        """
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
    """
    Zentrales Vision-System zur Dart-Treffererkennung.

    Hauptaufgaben:
    - Verwaltung aller Kameras
    - Board-Normalisierung über Homographien
    - Stabilitäts-/Bewegungserkennung des Boards
    - Kandidatenerkennung pro Kamera via AbsDiffDetector
    - Fusion mehrerer Kameramasken zu einem finalen Trefferpunkt
    - Entprellung / Cooldown / Candidate-Bestätigung
    - Referenzbild-Management
    - Umrechnung Board-Koordinate -> Dart-Score
    - Debug-Ausgabe
    """

    def __init__(self, hit_callback):
        self.hit_callback = hit_callback

        self.cameras = [CameraHandler(i) for i in range(3)]

        # ------------------------------------------------------------
        # Board-Maske
        # ------------------------------------------------------------
        self.board_mask = np.zeros((600, 600), dtype=np.uint8)

        total_radius_mm = 170.0 + 55.0
        self.px_per_mm_calc = (600 * 0.70) / (total_radius_mm * 2)

        cv2.circle(
            self.board_mask,
            (300, 300),
            int(225 * self.px_per_mm_calc),
            255,
            -1
        )

        # ------------------------------------------------------------
        # Ringradien des Dartboards in Pixeln
        # ------------------------------------------------------------
        self.radii = {
            "bull": 6.35 * self.px_per_mm_calc,
            "single_bull": 15.9 * self.px_per_mm_calc,
            "triple_inner": 97.0 * self.px_per_mm_calc,
            "triple_outer": 107.0 * self.px_per_mm_calc,
            "double_inner": 160.0 * self.px_per_mm_calc,
            "double_outer": 170.0 * self.px_per_mm_calc,
        }

        # ------------------------------------------------------------
        # Schwellwerte für globale Stabilitätsprüfung
        # ------------------------------------------------------------
        # Nicht mehr rein "gray absdiff", sondern robuster:
        # Lab-Farbdistanz + Gray-Differenz als Zusatzsignal
        self.STABILITY_MEAN_THRESHOLD = 11.0
        self.STABILITY_MAX_THRESHOLD = 42.0

        # ------------------------------------------------------------
        # Pro Kamera ein identischer Detector
        # ------------------------------------------------------------
        self.abs_detectors = [
            AbsDiffDetector(self.board_mask, freeze_mean=20, freeze_max=70)
            for _ in range(3)
        ]

        # Optional: globale gemeinsame Detector-Parameter
        for det in self.abs_detectors:
            det.min_area = 140
            det.max_area = 18000
            det.min_length = 12
            det.merge_dist = 20
            det.max_width = 54.0
            det.min_slenderness = 1.15
            det.min_aspect = 1.02
            det.color_diff_threshold = 22.0
            det.grad_diff_threshold = 18.0
            det.min_foreground_ratio = 0.00008
            det.max_foreground_ratio = 0.12
            det.use_virtual_greenscreen = True

        self.debugger = VisionDebugger(warp_size=800)

        self.running = True
        self.console_debug = True

        # Letzter bestätigter Treffer
        self.last_hit_time = 0.0
        self.last_hit_board = None

        # Candidate-System
        self.hit_candidate = None
        self.hit_candidate_time = 0.0

        self.WINKEL_OFFSET = 0

        # Sperrzeiten
        self.hit_blocked_until = time.time() + 0.8
        self.hit_cooldown_until = 0.0

        # Referenzupdate-Management
        self.pending_reference_update = False
        self.pending_reference_started_at = 0.0

        self.stable_frames = 0
        self.required_stable_frames = 5

        # Loop-Timing
        self.loop_idle_sleep = 0.0005
        self.max_frame_age_sec = 0.20

        # Debug-Caches
        self.last_candidate_counts = [None, None, None]
        self.last_reject_log = [None, None, None]
        self.last_local_best_label = [None, None, None]
        self.last_fusion_summary = None
        self.last_motion_summary = [None, None, None]

        # Fusion
        self.mask_consensus_max_dist = 20.0

    def _dbg(self, msg):
        if self.console_debug:
            print(msg)

    def _prepare_gray(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        return gray

    def _prepare_lab(self, frame_bgr):
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        lab = cv2.GaussianBlur(lab, (5, 5), 0)
        return lab

    def _compute_board_stability_metrics(self, warped_bgr, cam: CameraHandler):
        """
        Robuste Stabilitätsprüfung gegen die gespeicherte Referenz.

        Idee:
        - Lab-Farbdistanz als Hauptsignal
        - Gray-Differenz als Zusatzsignal
        - nur innerhalb der Board-Maske
        """
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

        # Mischsignal: Farbe zählt stärker, Gray etwas schwächer
        combined = 0.75 * color_dist + 0.25 * diff_gray

        vals = combined[valid]
        if vals.size == 0:
            return None

        mean_val = float(np.mean(vals))
        max_val = float(np.max(vals))

        return {
            "mean_val": mean_val,
            "max_val": max_val,
            "combined": combined,
        }

    def _is_board_moving_from_metrics(self, metrics):
        """
        Heuristik:
        - mean hoch und max nicht extrem hoch => eher globaler Drift / Wackeln
        """
        if metrics is None:
            return False

        mean_val = metrics["mean_val"]
        max_val = metrics["max_val"]

        return mean_val > self.STABILITY_MEAN_THRESHOLD and max_val < self.STABILITY_MAX_THRESHOLD

    def _block_hits(self, seconds=1.0):
        until = time.time() + seconds
        self.hit_blocked_until = max(self.hit_blocked_until, until)
        self.hit_candidate = None
        self._dbg(f"[HIT BLOCK] Treffer gesperrt für {seconds:.2f}s bis {self.hit_blocked_until:.3f}")

    def _hits_allowed(self):
        now = time.time()
        return now >= self.hit_blocked_until and now >= self.hit_cooldown_until

    def run(self):
        """
        Hauptloop des Vision-Systems.
        """
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

                mask_results = [None, None, None]
                mask_list = [None, None, None]

                for idx, cam in enumerate(self.cameras):
                    if cam.H is None:
                        continue

                    frame, frame_ts, _counter = cam.get_latest_frame(only_new=True)
                    if frame is None:
                        continue

                    if frame_ts is not None and (now - frame_ts) > self.max_frame_age_sec:
                        continue

                    any_frame_processed = True
                    raw_frames[idx] = frame

                    warped = cam.warp_to_board(frame, 600)
                    if warped is None:
                        continue

                    warped_frames[idx] = warped

                    # ----------------------------------------------------
                    # Robuste Board-Stabilitätsprüfung
                    # ----------------------------------------------------
                    metrics = self._compute_board_stability_metrics(warped, cam)
                    if self._is_board_moving_from_metrics(metrics):
                        cam_moving[idx] = True
                        board_is_moving = True

                    if metrics is not None:
                        motion_summary = (
                            round(metrics["mean_val"], 2),
                            round(metrics["max_val"], 2),
                            cam_moving[idx],
                        )
                        if self.last_motion_summary[idx] != motion_summary:
                            self.last_motion_summary[idx] = motion_summary
                            self._dbg(
                                f"[CAM {cam.cam_id}] STABILITY "
                                f"mean={metrics['mean_val']:.2f} "
                                f"max={metrics['max_val']:.2f} "
                                f"moving={cam_moving[idx]}"
                            )

                    # ----------------------------------------------------
                    # Kandidatenmaske per Foreground/AbsDiff ermitteln
                    # ----------------------------------------------------
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

                # --------------------------------------------------------
                # Fusion nur unter sicheren Bedingungen
                # --------------------------------------------------------
                if (
                    self._hits_allowed()
                    and not board_is_moving
                    and active_masks >= 2
                    and not any(cam_moving)
                ):
                    fused_hit = fuse_warped_masks(
                        mask_list=mask_list,
                        board_mask=self.board_mask,
                        max_dist=self.mask_consensus_max_dist,
                    )

                    if fused_hit is not None and (now - self.last_hit_time > 0.12):
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

                # --------------------------------------------------------
                # Debuganzeige aktualisieren
                # --------------------------------------------------------
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

                # --------------------------------------------------------
                # Stabilität des Boards überwachen
                # --------------------------------------------------------
                if board_is_moving:
                    self.hit_candidate = None
                    self.stable_frames = 0
                else:
                    self.stable_frames += 1

                # --------------------------------------------------------
                # Geplantes Referenzupdate nur bei stabiler Lage
                # --------------------------------------------------------
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
        """
        Stoppt das gesamte Vision-System sauber.
        """
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

    def _store_camera_references(self, cam: CameraHandler, warped_bgr):
        """
        Speichert alle Referenzrepräsentationen für eine Kamera konsistent.
        """
        gray = self._prepare_gray(warped_bgr)
        lab = self._prepare_lab(warped_bgr)

        cam.reference_gray = gray
        cam.reference_lab = lab

    def reset_references(self):
        """
        Setzt alle Referenzbilder vollständig neu.
        """
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
                continue

            frame, _, _ = cam.get_latest_frame(only_new=False)
            if frame is None:
                continue

            warped = cam.warp_to_board(frame, 600)
            if warped is None:
                continue

            self.abs_detectors[idx].set_reference(warped)
            self._store_camera_references(cam, warped)

            self.last_candidate_counts[idx] = None
            self.last_reject_log[idx] = None
            self.last_local_best_label[idx] = None
            self.last_motion_summary[idx] = None

            self._dbg(f"[RESET REFERENCES] Cam {cam.cam_id} Referenzen gesetzt")

        self.pending_reference_update = False
        self.stable_frames = 0

        self._block_hits(0.5)

    def update_references_fast(self):
        """
        Aktualisiert die Referenzbilder schnell mit den aktuellen Frames.
        """
        for idx, cam in enumerate(self.cameras):
            if cam.H is None:
                continue

            frame, _, _ = cam.get_latest_frame(only_new=False)
            if frame is None:
                continue

            warped = cam.warp_to_board(frame, 600)
            if warped is None:
                continue

            self.abs_detectors[idx].set_reference(warped)
            self._store_camera_references(cam, warped)

            self.last_candidate_counts[idx] = None
            self.last_reject_log[idx] = None
            self.last_local_best_label[idx] = None
            self.last_motion_summary[idx] = None

        self.stable_frames = 0
        self._block_hits(0.10)

    def _schedule_reference_update(self):
        """
        Plant ein Referenzupdate ein, das später bei stabiler Boardlage
        ausgeführt werden soll.
        """
        self.pending_reference_update = True
        self.pending_reference_started_at = time.time()
        self.stable_frames = 0

    def _pick_best_candidate(self, objs, source_name, conf_scale=1.0):
        """
        Wählt aus einer Kandidatenliste den Eintrag mit der höchsten Confidence.
        """
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
        """
        Formatiert eine Boardposition als lesbares Score-Label.
        """
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
        """
        Meldet einen bestätigten Treffer an das übergeordnete System.
        """
        if self.last_hit_board is not None:
            if np.linalg.norm(np.array((bx, by)) - np.array(self.last_hit_board)) < 18:
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

        self.hit_candidate = None
        self.hit_cooldown_until = time.time() + 0.20
        self._schedule_reference_update()

    def _score_from_board(self, x, y):
        """
        Wandelt eine Boardkoordinate in einen Dart-Score um.
        """
        rel_x, rel_y = x - 300, y - 300
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
