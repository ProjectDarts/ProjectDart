import cv2
import json
import os
import sys
import time
import numpy as np

# --- PFAD LOGIK ---
def get_config_path(filename):
    if getattr(sys, 'frozen', False):
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, filename)

class Calibrator:
    def __init__(self, cam_ids=[0, 1, 2]):
        self.cam_ids = cam_ids
        self.current_cam_idx = 0
        self.points = []
        # --- NEUE REIHENFOLGE FÜR PROFISYSTEM ---
        self.points_desc = ["Oben (20/1)", "Rechts (6/10)", "Unten (3/19)", "Links (11/14)"]
        self.cap = None
        self.mouse_x = 0
        self.mouse_y = 0
        self.is_dragging = False
        self.setup_cam()

    def setup_cam(self):
        if self.cap is not None:
            self.cap.release()
            
        if self.current_cam_idx >= len(self.cam_ids):
            print("[INFO] Alle Kameras kalibriert!")
            return False
        
        cam_id = self.cam_ids[self.current_cam_idx]
        print(f"[INFO] Initialisiere Kamera ID: {cam_id}...")
        
        self.cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        
        if not self.cap.isOpened():
            print(f"[ERROR] Kamera ID {cam_id} konnte nicht geöffnet werden.")
            self.current_cam_idx += 1
            return self.setup_cam()
        
        # --- UMSCHALTUNG AUF 1080p ---
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # Wichtig: Höhere Auflösung braucht etwas mehr Zeit zum initialisieren
        time.sleep(1.5)
        for _ in range(10): self.cap.read()
            
        self.points = []
        return True

    def run(self):
        window_name = "Kalibrierung - 1080p"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # Fenster skalierbar machen

        def local_mouse_callback(event, x, y, flags, param):
            self.mouse_x, self.mouse_y = x, y
            
            if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
                self.is_dragging = True
            
            elif event == cv2.EVENT_LBUTTONUP and self.is_dragging:
                self.is_dragging = False
                if len(self.points) < 4:
                    self.points.append([x, y])
                    print(f"Punkt {len(self.points)} ({self.points_desc[len(self.points)-1]}) gesetzt: {x}, {y}")
        
        cv2.setMouseCallback(window_name, local_mouse_callback)

        while self.current_cam_idx < len(self.cam_ids):
            ret, frame = self.cap.read()
            if not ret or frame is None: 
                self.current_cam_idx += 1
                if not self.setup_cam(): break
                continue

            cam_id = self.cam_ids[self.current_cam_idx]
            
            # --- BILD ANZEIGEN ---
            display_frame = frame.copy()

            # --- OVERLAY TEXT ---
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_color = (255, 255, 0)
            
            # Anzeige, welchen Punkt man gerade anklicken soll
            current_desc = self.points_desc[min(len(self.points), 3)] if len(self.points) < 4 else "Fertig"
            info_txt = f"CAM: {cam_id} | KLICK AUF: {current_desc} | 1920x1080"
            
            # Textgröße an 1080p anpassen
            cv2.putText(display_frame, info_txt, (20, 50), font, 1.2, text_color, 3)
            cv2.putText(display_frame, "LEERTASTE = Speichern | q = Abbrechen", (20, 100), font, 1.0, (200,200,200), 2)

            # --- LUPE ZEICHNEN ---
            if self.is_dragging or len(self.points) < 4:
                zoom_factor = 3
                zoom_size = 100 # Etwas größer bei 1080p
                x1 = max(0, self.mouse_x - zoom_size // 2)
                y1 = max(0, self.mouse_y - zoom_size // 2)
                x2 = min(1920, self.mouse_x + zoom_size // 2)
                y2 = min(1080, self.mouse_y + zoom_size // 2)
                
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    zoom_img = cv2.resize(roi, (zoom_size * zoom_factor, zoom_size * zoom_factor), interpolation=cv2.INTER_NEAREST)
                    
                    # Lupe oben rechts platzieren
                    h_z, w_z = zoom_img.shape[:2]
                    display_frame[20:20+h_z, 1920-w_z-20:1920-20] = zoom_img
                    
                    # Fadenkreuz in der Lupe
                    cv2.line(display_frame, (1920-w_z//2-20, 20), (1920-w_z//2-20, 20+h_z), (0, 0, 255), 2)
                    cv2.line(display_frame, (1920-w_z-20, 20+h_z//2), (1920-20, 20+h_z//2), (0, 0, 255), 2)

            # Punkte einzeichnen
            for i, p in enumerate(self.points):
                cv2.circle(display_frame, (p[0], p[1]), 10, (0, 255, 0), -1)
                cv2.putText(display_frame, str(i+1), (p[0]+20, p[1]+20), font, 1.5, (0, 255, 0), 3)

            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' ') and len(self.points) == 4:
                filename = get_config_path(f"cam{cam_id}_config.json")
                try:
                    with open(filename, "w") as f:
                        json.dump({"points": self.points}, f)
                    print(f"Erfolg: {filename} gespeichert.")
                except Exception as e:
                    print(f"[ERROR] Speichern fehlgeschlagen: {e}")
                
                self.current_cam_idx += 1
                if not self.setup_cam(): break
            
            elif key == ord('q') or key == 27:
                print("[INFO] Kalibrierung abgebrochen.")
                break

        if self.cap is not None: self.cap.release()
        cv2.destroyAllWindows()

def start_calibration():
    print("[SYSTEM] Kalibrierung gestartet...")
    cal = Calibrator(cam_ids=[0, 1, 2])
    cal.run()

if __name__ == "__main__":
    start_calibration()