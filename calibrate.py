import cv2
import json
import os
import sys
import time
import numpy as np


# =========================================================
# PFAD-LOGIK
# =========================================================
# Diese Funktion sorgt dafür, dass Konfigurationsdateien
# immer im gleichen Verzeichnis gespeichert werden.
#
# Wichtig:
# - Im normalen Python-Betrieb = Verzeichnis der .py Datei
# - Als EXE (PyInstaller) = Verzeichnis der EXE
# =========================================================
def get_config_path(filename):
    if getattr(sys, 'frozen', False):
        # Wenn als EXE gestartet
        base_path = os.path.dirname(sys.executable)
    else:
        # Wenn als Python-Skript gestartet
        base_path = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(base_path, filename)


# =========================================================
# KALIBRIERUNGS-KLASSE
# =========================================================
# Diese Klasse übernimmt:
# - Kamera initialisieren
# - Kameras durchschalten
# - Klickpunkte erfassen
# - Lupe anzeigen
# - Punkte speichern
# =========================================================
class Calibrator:

    def __init__(self, cam_ids=[0, 1, 2]):
        # Liste aller zu kalibrierenden Kameras
        self.cam_ids = cam_ids

        # Start mit erster Kamera
        self.current_cam_idx = 0

        # Hier werden die 4 Klickpunkte gespeichert
        self.points = []

        # Beschreibung der Klick-Reihenfolge
        # Sehr wichtig für Homography / Board Mapping
        self.points_desc = [
            "Oben (20/1)",
            "Rechts (6/10)",
            "Unten (3/19)",
            "Links (11/14)"
        ]

        # Kamera-Objekt
        self.cap = None

        # Aktuelle Mausposition
        self.mouse_x = 0
        self.mouse_y = 0

        # Merkt, ob gerade mit gedrückter Maustaste gearbeitet wird
        self.is_dragging = False

        # Erste Kamera initialisieren
        self.setup_cam()


    # =====================================================
    # KAMERA INITIALISIEREN
    # =====================================================
    # - alte Kamera schließen
    # - neue Kamera öffnen
    # - auf 1080p setzen
    # - kurz warten bis Kamera stabil läuft
    # =====================================================
    def setup_cam(self):

        # Falls bereits Kamera offen → schließen
        if self.cap is not None:
            self.cap.release()

        # Prüfen ob alle Kameras fertig sind
        if self.current_cam_idx >= len(self.cam_ids):
            print("[INFO] Alle Kameras kalibriert!")
            return False

        # Aktuelle Kamera-ID holen
        cam_id = self.cam_ids[self.current_cam_idx]
        print(f"[INFO] Initialisiere Kamera ID: {cam_id}...")

        # Kamera öffnen
        self.cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)

        # Fehler wenn Kamera nicht geöffnet werden konnte
        if not self.cap.isOpened():
            print(f"[ERROR] Kamera ID {cam_id} konnte nicht geöffnet werden.")

            # Zur nächsten Kamera springen
            self.current_cam_idx += 1
            return self.setup_cam()

        # Kamera auf FullHD setzen
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # Kamera braucht etwas Zeit zum Stabilisieren
        time.sleep(1.5)

        # Einige Frames verwerfen damit Bild sauber ist
        for _ in range(10):
            self.cap.read()

        # Punkte für neue Kamera zurücksetzen
        self.points = []

        return True


    # =====================================================
    # HAUPTSCHLEIFE DER KALIBRIERUNG
    # =====================================================
    def run(self):

        window_name = "Kalibrierung - 1080p"

        # Skalierbares Fenster erstellen
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)


        # =================================================
        # MAUS CALLBACK
        # =================================================
        # Reagiert auf Mausbewegung und Klicks
        # =================================================
        def local_mouse_callback(event, x, y, flags, param):

            # Mausposition speichern
            self.mouse_x, self.mouse_y = x, y

            # Linke Maustaste gedrückt
            if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
                self.is_dragging = True

            # Linke Maustaste losgelassen
            elif event == cv2.EVENT_LBUTTONUP and self.is_dragging:
                self.is_dragging = False

                # Maximal 4 Punkte speichern
                if len(self.points) < 4:
                    self.points.append([x, y])

                    print(
                        f"Punkt {len(self.points)} "
                        f"({self.points_desc[len(self.points)-1]}) "
                        f"gesetzt: {x}, {y}"
                    )

        # Callback registrieren
        cv2.setMouseCallback(window_name, local_mouse_callback)


        # =================================================
        # HAUPT LOOP
        # =================================================
        while self.current_cam_idx < len(self.cam_ids):

            # Frame lesen
            ret, frame = self.cap.read()

            # Falls Frame fehlerhaft → nächste Kamera
            if not ret or frame is None:
                self.current_cam_idx += 1
                if not self.setup_cam():
                    break
                continue

            cam_id = self.cam_ids[self.current_cam_idx]

            # Kopie für Anzeige erzeugen
            display_frame = frame.copy()

            # Schriftart
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_color = (255, 255, 0)

            # Welcher Punkt ist aktuell dran?
            current_desc = (
                self.points_desc[min(len(self.points), 3)]
                if len(self.points) < 4
                else "Fertig"
            )

            # Info-Text
            info_txt = (
                f"CAM: {cam_id} | "
                f"KLICK AUF: {current_desc} | "
                f"1920x1080"
            )

            # Overlay-Text anzeigen
            cv2.putText(display_frame, info_txt, (20, 50),
                        font, 1.2, text_color, 3)

            cv2.putText(display_frame,
                        "LEERTASTE = Speichern | q = Abbrechen",
                        (20, 100),
                        font, 1.0,
                        (200, 200, 200), 2)


            # =================================================
            # LUPE / ZOOM-FUNKTION
            # =================================================
            if self.is_dragging or len(self.points) < 4:

                zoom_factor = 3
                zoom_size = 100

                # Bereich um Maus berechnen
                x1 = max(0, self.mouse_x - zoom_size // 2)
                y1 = max(0, self.mouse_y - zoom_size // 2)
                x2 = min(1920, self.mouse_x + zoom_size // 2)
                y2 = min(1080, self.mouse_y + zoom_size // 2)

                roi = frame[y1:y2, x1:x2]

                if roi.size > 0:

                    # Bereich vergrößern
                    zoom_img = cv2.resize(
                        roi,
                        (zoom_size * zoom_factor,
                         zoom_size * zoom_factor),
                        interpolation=cv2.INTER_NEAREST
                    )

                    h_z, w_z = zoom_img.shape[:2]

                    # Lupe oben rechts einfügen
                    display_frame[
                        20:20+h_z,
                        1920-w_z-20:1920-20
                    ] = zoom_img

                    # Fadenkreuz
                    cv2.line(display_frame,
                             (1920-w_z//2-20, 20),
                             (1920-w_z//2-20, 20+h_z),
                             (0, 0, 255), 2)

                    cv2.line(display_frame,
                             (1920-w_z-20, 20+h_z//2),
                             (1920-20, 20+h_z//2),
                             (0, 0, 255), 2)


            # =================================================
            # BEREITS GESETZTE PUNKTE EINZEICHNEN
            # =================================================
            for i, p in enumerate(self.points):

                # Grüner Punkt
                cv2.circle(display_frame,
                           (p[0], p[1]),
                           10,
                           (0, 255, 0),
                           -1)

                # Punktnummer
                cv2.putText(display_frame,
                            str(i+1),
                            (p[0]+20, p[1]+20),
                            font,
                            1.5,
                            (0, 255, 0),
                            3)

            # Bild anzeigen
            cv2.imshow(window_name, display_frame)

            # Tastatur abfragen
            key = cv2.waitKey(1) & 0xFF


            # =================================================
            # SPEICHERN
            # =================================================
            if key == ord(' ') and len(self.points) == 4:

                filename = get_config_path(f"cam{cam_id}_config.json")

                try:
                    with open(filename, "w") as f:
                        json.dump({"points": self.points}, f)

                    print(f"Erfolg: {filename} gespeichert.")

                except Exception as e:
                    print(f"[ERROR] Speichern fehlgeschlagen: {e}")

                # Nächste Kamera
                self.current_cam_idx += 1

                if not self.setup_cam():
                    break


            # =================================================
            # ABBRUCH
            # =================================================
            elif key == ord('q') or key == 27:
                print("[INFO] Kalibrierung abgebrochen.")
                break


        # Kamera sauber schließen
        if self.cap is not None:
            self.cap.release()

        cv2.destroyAllWindows()


# =========================================================
# STARTFUNKTION
# =========================================================
def start_calibration():
    print("[SYSTEM] Kalibrierung gestartet...")

    cal = Calibrator(cam_ids=[0, 1, 2])
    cal.run()


# =========================================================
# SCRIPT STARTPUNKT
# =========================================================
if __name__ == "__main__":
    start_calibration()
