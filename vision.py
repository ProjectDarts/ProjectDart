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
    def __init__(self, cam_id):
        self.cam_id = cam_id
        self.config_file = get_external_path(f"cam{cam_id}_config.json")
        self.src_points = []
        self.load_config()
        
        # Geschätzte Verzerrungskoeffizienten für Linsenkorrektur
        self.distortion_values = {0: 1.8, 1: 1.8, 2: 1.8}
        
        print(f"[DEBUG] Initialisiere Kamera {cam_id}...")
        self.cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print(f"[ERROR] Kamera {cam_id} konnte nicht geöffnet werden!")
        else:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            time.sleep(1.0)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            # Exposure Einstellungen für Stabilität
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) # 1 = Manual
            self.cap.set(cv2.CAP_PROP_EXPOSURE, -7) 
            self.cap.set(cv2.CAP_PROP_GAIN, 10)
            if self.cam_id == 2: self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 100)
            else: self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)
            print(f"[DEBUG] Kamera {cam_id} initialisiert.")
        
        self.reference_gray = None
        self.matrix = None
        
        # Output Auflösung fest auf 1000x1000
        self.output_width = 1000
        self.output_height = 1000
        self.compute_warp_matrix()
    
    def load_config(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    data = json.load(f)
                    self.src_points = data.get("points", [])
            except:
                print(f"[ERROR] Config für Cam {self.cam_id} konnte nicht geladen werden.")
                self.src_points = [] 

    def get_cam_intrinsic(self, frame):
        """Erstellt eine Schätzung der Kamera-Matrix aus der Bildgröße."""
        h, w = frame.shape[:2]
        focal_length = w  # Grobe Schätzung
        camera_matrix = np.array([
            [focal_length, 0, w/2],
            [0, focal_length, h/2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Verzerrung schätzen
        k = self.distortion_values.get(self.cam_id, 0.0)
        dist_coeffs = np.array([k, 0, 0, 0, 0], dtype=np.float32)
        
        return camera_matrix, dist_coeffs

    def compute_warp_matrix(self):
        if len(self.src_points) < 4: return
        pts1 = np.float32(self.src_points)
        
        target_center = self.output_width / 2
        
        # Faktor für korrekte Boardplatzierung
        self.nutzungs_faktor = 0.70 
        target_radius_px = (self.output_width / 2) * self.nutzungs_faktor
        
        # ZIELPUNKTE IM BILD (absolut auf 1000x1000 bezogen)
        pts2 = np.float32([
            [target_center, target_center - target_radius_px], # Top
            [target_center + target_radius_px, target_center], # Right
            [target_center, target_center + target_radius_px], # Bottom
            [target_center - target_radius_px, target_center]  # Left
        ])
        
        try:
            # Punkte entzerren, bevor die Matrix berechnet wird
            h, w = 1080, 1920 # Annahme der Rohauflösung
            focal_length = w
            camera_matrix = np.array([
                [focal_length, 0, w/2],
                [0, focal_length, h/2],
                [0, 0, 1]
            ], dtype=np.float32)
            k = self.distortion_values.get(self.cam_id, 0.0)
            dist_coeffs = np.array([k, 0, 0, 0, 0], dtype=np.float32)
            
            pts1_undistorted = cv2.undistortPoints(pts1.reshape(-1, 1, 2), camera_matrix, dist_coeffs, P=camera_matrix)
            pts1_undistorted = pts1_undistorted.reshape(-1, 2)
            
            self.matrix = cv2.getPerspectiveTransform(pts1_undistorted, pts2)
        except Exception as e:
            print(f"[ERROR] Warp Matrix Fehler für Cam {self.cam_id}: {e}")
            self.matrix = None

    def get_warped(self, frame):
        if self.matrix is None or frame is None: return None
        
        # Bild entzerren, VOR dem WarpPerspective
        cam_mat, dist_coeffs = self.get_cam_intrinsic(frame)
        undistorted_frame = cv2.undistort(frame, cam_mat, dist_coeffs)
        
        # WARP auf 1000x1000
        return cv2.warpPerspective(undistorted_frame, self.matrix, (self.output_width, self.output_height))

class DartVisionSystem:
    def __init__(self, hit_callback):
        self.hit_callback = hit_callback
        self.cameras = [CameraHandler(i) for i in range(3)]
        
        self.canvas_size = 1000
        self.board_mask = np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)
        
        # mm_to_px basierend auf dem Faktor
        target_radius_px = (self.canvas_size / 2) * 0.70
        self.mm_to_px = target_radius_px / 170.0 
        
        center_px = self.canvas_size // 2
        # Maske etwas größer als das Board
        cv2.circle(self.board_mask, (center_px, center_px), int(175 * self.mm_to_px), 255, -1)
        
        self.FREEZE_MEAN = 10 
        self.FREEZE_MAX = 50
        
        self.detectors = [
            AbsDiffDetector(self.board_mask, self.FREEZE_MEAN, self.FREEZE_MAX)
            for _ in range(3)
        ]
        self.takeout_detectors = [TakeoutDetector(self.board_mask) for _ in range(3)]
        
        # TRIPLE-RING KORREKTUR: 1mm nach innen gesetzt
        self.radii = {
            "bull": 6.35 * self.mm_to_px, 
            "single_bull": 15.9 * self.mm_to_px,
            "triple_outer": (107.0 - 1.0) * self.mm_to_px, # 1mm nach innen
            "triple_inner": (107.0 - 8.0 - 1.0) * self.mm_to_px, # 1mm nach innen
            "double_outer": 170.0 * self.mm_to_px, 
            "double_inner": (170.0 - 8.0) * self.mm_to_px  
        }
        
        # WINKEL OFFSET für die Berechnung (20 oben)
        self.WINKEL_OFFSET = 0 
        
        self.last_hit_time = 0
        self.running = True
        self.waiting_for_reset = False
        self.quiet_frames = 0
        
        self.last_hit_coords = None
        self.last_hit_score = None
        self.last_hit_contours = {}
        
        self.hit_candidate = None
        self.hit_candidate_time = 0

    def draw_spider_overlay(self, frame):
        """Zeichnet das Dartboard-Raster basierend auf dem OFFSET."""
        center = self.canvas_size // 2
        color = (0, 255, 0) # HELLGRÜN (BGR)
        
        # Kreise zeichnen
        cv2.circle(frame, (center, center), int(self.radii["double_outer"]), color, 2)
        cv2.circle(frame, (center, center), int(self.radii["double_inner"]), color, 2)
        cv2.circle(frame, (center, center), int(self.radii["triple_outer"]), color, 2)
        cv2.circle(frame, (center, center), int(self.radii["triple_inner"]), color, 2)
        cv2.circle(frame, (center, center), int(self.radii["bull"]), color, 1)
        cv2.circle(frame, (center, center), int(self.radii["single_bull"]), color, 1)

        # Sektorenlinien berechnen mit OFFSET (20 ist oben = -90°)
        for i in range(20):
            # Der Winkel wird hier mit dem OFFSET angepasst
            angle = i * 18 - 90 + self.WINKEL_OFFSET
            
            x1 = int(center + self.radii["double_outer"] * np.cos(np.radians(angle)))
            y1 = int(center + self.radii["double_outer"] * np.sin(np.radians(angle)))
            
            cv2.line(frame, (center, center), (x1, y1), color, 1)

    def run(self):
        print("[VISION] System bereit (1000x1000)...") 
        try:
            print("[DEBUG] Versuche Referenzen zu setzen...")
            self.reset_references()
            print("[VISION] Referenzen gesetzt. Starte Loop...")
            
            while self.running:
                valid_cam_data = [] 
                max_area_found = 0
                debug_frames = {}
                all_cameras_empty = True
                board_is_moving = False
                
                # --- ZWEI-STUFEN-LOGIK: CHECKEN OB EINE CAM WAS SIEHT ---
                for i, cam in enumerate(self.cameras):
                    if cam.cap is None or not cam.cap.isOpened(): continue
                    ret, frame = cam.cap.read()
                    if not ret: continue
                    
                    warped = cam.get_warped(frame)
                    if warped is None: continue
                    
                    # Hier schauen wir nur kurz auf die Fläche, ohne teure PCA
                    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                    if cam.reference_gray is not None:
                        diff = cv2.absdiff(gray, cam.reference_gray)
                        _, thr = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
                        thr = cv2.bitwise_and(thr, self.board_mask)
                        contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for cnt in contours:
                            if cv2.contourArea(cnt) > 200: # Grobe Änderung
                                all_cameras_empty = False
                                break

                # --- MODUS UMSCHALTEN ---
                if not all_cameras_empty:
                    for detector in self.detectors: detector.high_sensitivity_mode = True
                else:
                    for detector in self.detectors: detector.high_sensitivity_mode = False

                # --- EIGENTLICHE VERARBEITUNG ---
                for i, cam in enumerate(self.cameras):
                    if cam.cap is None or not cam.cap.isOpened(): continue
                    ret, frame = cam.cap.read()
                    if not ret: continue
                    
                    warped = cam.get_warped(frame)
                    if warped is None: continue
                    
                    self.draw_spider_overlay(warped)
                    gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                    
                    # --- MOTION FREEZE CHECK ---
                    if cam.reference_gray is not None:
                        diff_motion = cv2.absdiff(gray_warped, cam.reference_gray)
                        mean_val = cv2.mean(diff_motion)[0]
                        _, max_val, _, _ = cv2.minMaxLoc(diff_motion)
                        
                        if mean_val > self.FREEZE_MEAN and max_val < self.FREEZE_MAX:
                            board_is_moving = True
                            debug_frames[cam.cam_id] = warped
                            continue 
                    
                    detected_objects, debug_img = self.detectors[i].detect(warped, gray_warped)
                    takeout_detected, takeout_debug = self.takeout_detectors[i].check_takeout(warped, self.last_hit_contours)
                    
                    # Zeichne alte Treffer (Persistenz)
                    if self.last_hit_coords and cam.cam_id in self.last_hit_contours:
                        px = int(self.last_hit_coords[0])
                        py = int(self.last_hit_coords[1])
                        cv2.drawContours(takeout_debug, [self.last_hit_contours[cam.cam_id]], 0, (0, 255, 255), 2)
                        cv2.circle(takeout_debug, (px, py), 10, (255, 255, 0), 3)

                    best_obj = None
                    max_conf = 0
                    
                    for obj in detected_objects:
                        if obj["confidence"] > max_conf:
                            max_conf = obj["confidence"]
                            best_obj = obj
                    
                    if best_obj:
                        area = best_obj["area"]
                        if area > max_area_found: max_area_found = area
                        
                        # FILTER: Mindestkonfidenz 
                        if best_obj["confidence"] < 3000:
                            debug_frames[cam.cam_id] = takeout_debug
                            continue
                        
                        # Takeout Initialisierung
                        if area > 25000 and best_obj["confidence"] > 25000:
                            if not self.waiting_for_reset: self.waiting_for_reset = True
                        
                        elif area > 300: 
                            if (time.time() - self.last_hit_time > 0.3):
                                tip = best_obj["tip"]
                                valid_cam_data.append((cam.cam_id, tip, area, best_obj["contour"], best_obj["confidence"]))
                                cv2.circle(takeout_debug, (int(tip[0]), int(tip[1])), 7, (0, 0, 255), -1) 
                    
                    debug_frames[cam.cam_id] = takeout_debug
                
                # --- Takeout Race Condition Fix ---
                if board_is_moving:
                    self.hit_candidate = None
                    time.sleep(0.01)

                # --- TAKE OUT ERGEBNIS AN SPIELLOGIK ---
                if all_cameras_empty and self.last_hit_coords:
                    print("[INFO] Alle Darts entfernt.")
                    self.last_hit_coords = None
                    self.last_hit_contours = {}
                    self.reset_references()
                    self.hit_callback("NEXT_PLAYER")
                    
                # --- SCORE BERECHNUNG & OUTLIER FILTERUNG ---
                if len(valid_cam_data) >= 2:
                    all_points = np.array([c[1] for c in valid_cam_data])
                    median_point = np.median(all_points, axis=0)
                    
                    # FILTER: Outlier
                    filtered_data = []
                    for data in valid_cam_data:
                        dist_to_median = np.linalg.norm(np.array(data[1]) - median_point)
                        if dist_to_median < 150: 
                            filtered_data.append(data)
                    
                    # Nur mit den guten Daten weiterarbeiten
                    if len(filtered_data) >= 2:
                        filtered_data.sort(key=lambda x: x[4], reverse=True)
                        
                        weights = [c[4] for c in filtered_data[:3]]
                        points = [c[1] for c in filtered_data[:3]]
                        
                        final_x = int(np.average([p[0] for p in points], weights=weights))
                        final_y = int(np.average([p[1] for p in points], weights=weights))
                        
                        c1, c2 = filtered_data[0], filtered_data[1]
                        dist = np.linalg.norm(np.array(c1[1]) - np.array(c2[1]))

                        # SCHWELLE & PLAUSIBILITÄTSPRÜFUNG
                        if dist < 150:
                            current_point = (final_x, final_y)
                            if self.hit_candidate is None:
                                self.hit_candidate = current_point
                                self.hit_candidate_time = time.time()
                                continue
                            
                            dist_prev = np.linalg.norm(np.array(current_point) - np.array(self.hit_candidate))
                            if dist_prev < 20 and (time.time() - self.hit_candidate_time) < 0.2:
                                pass
                            else:
                                self.hit_candidate = current_point
                                self.hit_candidate_time = time.time()
                                continue

                            is_new_dart = True
                            if self.last_hit_coords:
                                old_dist = np.linalg.norm(np.array((final_x, final_y)) - np.array(self.last_hit_coords))
                                if old_dist < 40:
                                    is_new_dart = False
                            
                            if is_new_dart:
                                hit_result = self.get_score(final_x, final_y)
                                if not hit_result.get("is_missed", True) or hit_result.get("sector", 0) > 0:
                                    self.last_hit_score = hit_result
                                    print(f"[SCORE FOUND] {self.last_hit_score} at {final_x},{final_y}")                
                                    self.last_hit_coords = (final_x, final_y)
                                    # Speichere Konturen aller Cams, die diesen Wurf sehen
                                    self.last_hit_contours = {data[0]: data[3] for data in filtered_data}                
                                    
                                    self.hit_callback(self.last_hit_score)
                                    self.last_hit_time = time.time()
                                    time.sleep(0.3) 
                                    self.update_all_references()
                                    self.waiting_for_reset = False
                                    self.hit_candidate = None
                
                # --- Fenster anzeigen ---
                for cam_id, img in debug_frames.items():
                    cv2.imshow(f"Cam {cam_id} Debug", img)
                cv2.waitKey(1)
                
                # RESET LOGIK
                if self.waiting_for_reset:
                    if max_area_found < 500: self.quiet_frames += 1
                    else: self.quiet_frames = 0
                    if self.quiet_frames > 15:
                        print("[INFO] Pfeile wurden gezogen.")
                        self.reset_references()
                        self.waiting_for_reset = False
                        self.quiet_frames = 0
                        self.last_hit_coords = None
                        self.last_hit_contours = {}
                
        except Exception as e: 
            import traceback
            print(f"[ERROR] Hauptschleifen-Fehler: {e}")
            traceback.print_exc()

    def update_all_references(self):
        for i, cam in enumerate(self.cameras):
            if cam.cap is None or not cam.cap.isOpened(): continue
            for _ in range(5): cam.cap.read()
            r, f = cam.cap.read()
            if r:
                w = cam.get_warped(f)
                if w is not None:
                    self.detectors[i].set_reference(w)
                    cam.reference_gray = cv2.cvtColor(w, cv2.COLOR_BGR2GRAY)

    def reset_references(self):
        for i, cam in enumerate(self.cameras):
            print(f"[DEBUG] Lade Config für Cam {i}...")
            cam.load_config()
            cam.compute_warp_matrix()
            if cam.matrix is not None and cam.cap is not None and cam.cap.isOpened():
                print(f"[DEBUG] Setze Referenz für Cam {i}...")
                for _ in range(10): cam.cap.read()
                ret, frame = cam.cap.read()
                if ret:
                    warped = cam.get_warped(frame)
                    if warped is not None:
                        self.detectors[i].set_reference(warped)
                        cam.reference_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                        self.takeout_detectors[i].set_clean_board(warped)
                        print(f"[DEBUG] Referenz für Cam {i} gesetzt.")
                else:
                    print(f"[ERROR] Kamera {i} liefert kein Bild für Referenz!")

    def get_score(self, x, y):
        """Berechnet Sektor, Multiplikator und Missed-Status."""
        rel_x, rel_y = x - 500, y - 500
        dist = np.linalg.norm([rel_x, rel_y])
        
        # Missed Überprüfung
        if dist > self.radii["double_outer"]:
            return {"sector": 0, "multiplier": 1, "is_missed": True}
        
        angle = (np.degrees(np.arctan2(-rel_y, rel_x)) + 360) % 360
        angle = (angle + self.WINKEL_OFFSET) % 360 
        segments = [6, 13, 4, 18, 1, 20, 5, 12, 9, 14, 11, 8, 16, 7, 19, 3, 17, 2, 15, 10]
        val = segments[int((angle + 9) / 18) % 20]
        
        # Rückgabe als Dict
        if dist <= self.radii["bull"]: return {"sector": 25, "multiplier": 2, "is_missed": False}
        if dist <= self.radii["single_bull"]: return {"sector": 25, "multiplier": 1, "is_missed": False}
        if self.radii["triple_inner"] <= dist <= self.radii["triple_outer"]: return {"sector": val, "multiplier": 3, "is_missed": False}
        if self.radii["double_inner"] <= dist <= self.radii["double_outer"]: return {"sector": val, "multiplier": 2, "is_missed": False}
        
        return {"sector": val, "multiplier": 1, "is_missed": False}

    def stop(self):
        self.running = False
        for cam in self.cameras: 
            if cam.cap is not None: cam.cap.release()
        cv2.destroyAllWindows()