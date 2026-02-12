import cv2
import numpy as np

class AbsDiffDetector:
    def __init__(self, board_mask, freeze_mean=20, freeze_max=70):
        self.board_mask = board_mask
        self.reference_frame = None
        self.FREEZE_MEAN = freeze_mean
        self.FREEZE_MAX = freeze_max
        self.board_center = (500, 500)
        
        # MODUS-FLAG
        self.high_sensitivity_mode = False
        
        self.radii = {
            "outer": 450, "double": 420, "triple": 260,
            "single_bull": 40, "bull": 15
        }

    def set_reference(self, frame):
        self.reference_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def draw_virtual_board(self, img, color=(255, 255, 0)):
        center = self.board_center
        for r in [self.radii["outer"], self.radii["double"], self.radii["triple"]]:
            cv2.circle(img, center, r, color, 1)
        # ... (Rest der Zeichnung bleibt gleich)
        cv2.circle(img, center, self.radii["bull"], color, 1)

    def detect(self, warped_frame, gray=None):
        if self.reference_frame is None:
            return [], warped_frame
        
        if gray is None:
            gray = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2GRAY)
            
        diff = cv2.absdiff(self.reference_frame, gray)
        
        # --- 🚀 PARAMETER ---
        if self.high_sensitivity_mode:
            thresh_val = 25 # Etwas höher als vorher, um Rauschen zu unterdrücken
            min_area = 50   # Etwas höher
        else:
            thresh_val = 50 
            min_area = 150 
        
        diff_blurred = cv2.GaussianBlur(diff, (5, 5), 0)
        _, thr = cv2.threshold(diff_blurred, thresh_val, 255, cv2.THRESH_BINARY)
        thr = cv2.bitwise_and(thr, self.board_mask)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
        
        contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        debug_img = warped_frame.copy()
        self.draw_virtual_board(debug_img, color=(255, 255, 0))
        
        raw_objects = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area: continue
            
            # --- PCA FÜR HAUPTACHSE ---
            if len(cnt) < 5: continue
            cnt_pts = cnt.reshape(-1, 2).astype(np.float32)
            mean, eigenvecs = cv2.PCACompute(cnt_pts, mean=None)
            center = mean[0]
            main_axis = eigenvecs[0]
            
            # --- SPITZE VS FLIGHT UNTERSCHEIDEN ---
            projected = np.dot(cnt_pts - center, main_axis)
            min_proj, max_proj = np.min(projected), np.max(projected)
            
            # Die Kontur in zwei Hälften teilen (Spitze/Flight)
            left_half = cnt_pts[np.dot(cnt_pts - center, main_axis) < 0]
            right_half = cnt_pts[np.dot(cnt_pts - center, main_axis) > 0]
            
            if len(left_half) < 2 or len(right_half) < 2: continue
            
            # Breite der Hälften messen
            def get_width(pts):
                if len(pts) < 2: return 0
                return np.linalg.norm(np.max(pts, axis=0) - np.min(pts, axis=0))
            
            width_left = get_width(left_half)
            width_right = get_width(right_half)
            
            # 🚀 FILTER: Wenn beide Seiten breit sind, ist es kein Pfeil
            if width_left > 30 and width_right > 30: continue 

            # Spitze ist das schmälere Ende
            if width_left < width_right:
                actual_tip = center + main_axis * min_proj
                tip_width = width_left
            else:
                actual_tip = center + main_axis * max_proj
                tip_width = width_right

            # --- BEWERTUNG ---
            dist_to_center = np.linalg.norm(actual_tip - self.board_center)
            
            # Sektor Berechnung
            dx = actual_tip[0] - self.board_center[0]
            dy = actual_tip[1] - self.board_center[1]
            angle = np.degrees(np.arctan2(dy, dx))
            if angle < 0: angle += 360
            sector_num = int(((angle + 9) % 360) / 18)
            sector_map = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
            sector = sector_map[sector_num]
            
            # Konfidenz: Maß für "Schmalheit" an der Spitze
            confidence = area * (100 / (tip_width + 1)) 
            
            raw_objects.append({
                "contour": cnt,
                "area": area,
                "confidence": confidence,
                "tip": tuple(actual_tip.astype(int)),
                "sector": sector,
                "is_missed": dist_to_center > self.radii["outer"]
            })
            
        # Merging
        merged_objects = []
        for obj in raw_objects:
            merged = False
            for m in merged_objects:
                if np.linalg.norm(np.array(obj["tip"]) - np.array(m["tip"])) < 50:
                    if obj["confidence"] > m["confidence"]:
                        m.update(obj)
                    merged = True
                    break
            if not merged:
                merged_objects.append(obj)
        
        # --- Debug Visualisierung ---
        for obj in merged_objects:
            contour_color = (0, 0, 255) # Rot für erkannt
            cv2.drawContours(debug_img, [obj["contour"]], 0, contour_color, 2)
            cv2.circle(debug_img, obj["tip"], 7, (0, 0, 255), -1)
            text = f"{obj['sector']} (Conf:{int(obj['confidence'])})"
            cv2.putText(debug_img, text, (obj["tip"][0]+15, obj["tip"][1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        return merged_objects, debug_img