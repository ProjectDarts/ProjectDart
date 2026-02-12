import cv2
import numpy as np

class TakeoutDetector:
    def __init__(self, board_mask):
        self.board_mask = board_mask
        self.clean_board = None

    def set_clean_board(self, frame):
        """Speichert das saubere Board ohne Pfeile."""
        self.clean_board = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def check_takeout(self, warped_frame, last_hit_contours):
        """Prüft, ob Pfeile aus den vergangenen Positionen entfernt wurden."""
        # --- BUGFIX: Nicht als leer melden, wenn noch keine Referenz existiert ---
        if self.clean_board is None:
            return False, warped_frame

        gray = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2GRAY)
        
        # --- ROBUSTERER ABSDIFF ---
        # Differenz zum leeren Board
        diff = cv2.absdiff(self.clean_board, gray)
        diff = cv2.GaussianBlur(diff, (5, 5), 0) # Leicht glätten
        
        # 🚀 Schwellwert leicht angepasst für 1000x1000
        _, thr = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
        thr = cv2.bitwise_and(thr, self.board_mask)
        
        # --- 🚀 BUGFIX 3: contours initialisieren ---
        takeout_detected = True 
        debug_img = warped_frame.copy()
        contours = []
        
        # Nur prüfen, wenn wir Darts zu entfernen haben
        if last_hit_contours:
            contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                # 🚀 FLÄCHENPRÜFUNG FÜR 1000x1000 ANGEPASST
                # Area Threshold erhöht, um Lichtreflexe besser zu ignorieren
                if cv2.contourArea(cnt) > 1200: # 🚀 HIER: von 900 auf 1200 erhöht
                    takeout_detected = False
                    break
        else:
            # Wenn keine Darts stecken, darf der TakeoutDetector nichts finden
            # 🚀 Schwellwert für Gesamtdifferenz für 1000x1000 angepasst
            if cv2.countNonZero(thr) > 4000: # 🚀 HIER: von 3000 auf 4000 erhöht
                takeout_detected = False

        # Debug-Visualisierung
        if not takeout_detected:
            cv2.drawContours(debug_img, contours, -1, (0, 0, 255), 1)
        
        return takeout_detected, debug_img