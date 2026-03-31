import cv2
import numpy as np
import os
import sys
import configparser


def get_external_path(filename):
    """
    Liefert den absoluten Pfad zu einer Datei, die neben dem Skript
    bzw. neben der ausführbaren Datei liegt.

    Hintergrund:
    - Im normalen Python-Betrieb liegt die Datei relativ zu __file__
    - In einer kompilierten/frozen Anwendung (z. B. PyInstaller)
      liegt sie relativ zur ausführbaren Datei

    Dadurch kann dieselbe Logik sowohl im Entwicklungsmodus
    als auch im gebauten Programm verwendet werden.
    """
    if getattr(sys, "frozen", False):
        # Wenn das Programm "eingefroren" / kompiliert läuft,
        # nehmen wir das Verzeichnis der EXE
        base_path = os.path.dirname(sys.executable)
    else:
        # Im normalen Skriptbetrieb nehmen wir das Verzeichnis
        # der aktuellen Python-Datei
        base_path = os.path.dirname(os.path.abspath(__file__))

    # Dateiname an das Basisverzeichnis anhängen
    return os.path.join(base_path, filename)


class VisionDebugger:
    """
    Debugger für Entwicklung:
    - Full-Frame Overlay (Tip + optional Vector-Linie)
    - Warp/Board Overlay (Tip in Boardspace)
    Ein/Aus über vision_debug.ini

    Methoden-Farben:
    - abs   = rot
    - vec   = gelb
    - shape = magenta
    - fusion / kombiniert = cyan

    Unterstützt jetzt auch Texte wie:
    - "abs | Single 7"
    - "abs | Double 19"
    - "fusion | Triple 20"

    Zweck der Klasse:
    Diese Klasse ist dafür da, Debug-Fenster für die Bildverarbeitung
    anzuzeigen. Sie kann sowohl das Originalbild ("FULL") als auch
    die entzerrte / transformierte Brettansicht ("WARP") darstellen.
    Zusätzlich können Trefferpunkte, Linien und HUD-Texte eingezeichnet
    werden.
    """

    def __init__(self, warp_size=800):
        """
        Initialisiert den Debugger und lädt die Konfiguration aus der INI-Datei.

        Parameter:
        - warp_size:
          Fallback-Größe der Warp-Darstellung, falls in der INI-Datei
          nichts anderes definiert ist.
        """
        (
            self.enabled,
            self.warp_size,
            self.show_full,
            self.show_warp,
            self.full_window_w,
            self.full_window_h,
            self.warp_window_w,
            self.warp_window_h,
            self.window_topmost,
        ) = self._read_ini(warp_size)

        # Merkt sich, welche Fenster bereits angelegt wurden,
        # damit sie nicht bei jedem Frame neu erstellt werden müssen.
        self._windows_created = set()

    def _read_ini(self, fallback_warp_size):
        """
        Liest die Datei 'vision_debug.ini' ein und extrahiert daraus
        alle Debug-Einstellungen.

        Falls die Datei nicht existiert oder einzelne Einträge fehlen,
        werden sinnvolle Standardwerte verwendet.

        Rückgabe:
        Tuple mit:
        - enabled          : Debugging aktiv/inaktiv
        - warp_size        : Größe des Warp-Bildes
        - show_full        : FULL-Fenster anzeigen
        - show_warp        : WARP-Fenster anzeigen
        - full_window_w    : Breite FULL-Fenster
        - full_window_h    : Höhe FULL-Fenster
        - warp_window_w    : Breite WARP-Fenster
        - warp_window_h    : Höhe WARP-Fenster
        - window_topmost   : Fenster immer im Vordergrund
        """
        ini_path = get_external_path("vision_debug.ini")
        cfg = configparser.ConfigParser()

        # Standardwerte, falls keine INI existiert oder Einträge fehlen
        enabled = True
        warp_size = fallback_warp_size
        show_full = True
        show_warp = True

        full_window_w = 1280
        full_window_h = 720
        warp_window_w = 1200
        warp_window_h = 1200

        window_topmost = False

        # Nur lesen, wenn die Datei wirklich existiert
        if os.path.exists(ini_path):
            cfg.read(ini_path, encoding="utf-8")

            # "debugging" = 1 -> aktiv, 0 -> deaktiviert
            enabled = int(cfg.get("vision", "debugging", fallback="1").strip()) == 1

            # Größe des Warp-Bildes
            warp_size = int(cfg.get("vision", "warp_size", fallback=str(fallback_warp_size)).strip())

            # Einzelne Fenster an/aus
            show_full = int(cfg.get("vision", "show_full", fallback="1").strip()) == 1
            show_warp = int(cfg.get("vision", "show_warp", fallback="1").strip()) == 1

            # Fenstergrößen
            full_window_w = int(cfg.get("vision", "full_window_w", fallback="1280").strip())
            full_window_h = int(cfg.get("vision", "full_window_h", fallback="720").strip())
            warp_window_w = int(cfg.get("vision", "warp_window_w", fallback="1200").strip())
            warp_window_h = int(cfg.get("vision", "warp_window_h", fallback="1200").strip())

            # Fenster immer im Vordergrund?
            window_topmost = int(cfg.get("vision", "window_topmost", fallback="0").strip()) == 1

        return (
            enabled,
            warp_size,
            show_full,
            show_warp,
            full_window_w,
            full_window_h,
            warp_window_w,
            warp_window_h,
            window_topmost,
        )

    def _ensure_windows(self, cam_id):
        """
        Stellt sicher, dass die benötigten OpenCV-Fenster existieren.

        Pro Kamera werden zwei mögliche Fenster angelegt:
        - "Cam X - FULL"
        - "Cam X - WARP"

        Bereits existierende Fenster werden nicht erneut erstellt.
        """
        if not self.enabled:
            # Falls Debugging komplett deaktiviert ist, nichts tun
            return

        if self.show_full:
            name = f"Cam {cam_id} - FULL"
            if name not in self._windows_created:
                # Fenster anlegen, frei skalierbar
                cv2.namedWindow(name, cv2.WINDOW_NORMAL)

                # Fenstergröße setzen
                cv2.resizeWindow(name, self.full_window_w, self.full_window_h)

                # Optional: Fenster immer im Vordergrund halten
                if self.window_topmost:
                    try:
                        cv2.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 1)
                    except Exception:
                        # Nicht jede OpenCV-/OS-Kombination unterstützt das
                        pass

                self._windows_created.add(name)

        if self.show_warp:
            name = f"Cam {cam_id} - WARP"
            if name not in self._windows_created:
                # Fenster anlegen, frei skalierbar
                cv2.namedWindow(name, cv2.WINDOW_NORMAL)

                # Fenstergröße setzen
                cv2.resizeWindow(name, self.warp_window_w, self.warp_window_h)

                # Optional: Fenster immer im Vordergrund halten
                if self.window_topmost:
                    try:
                        cv2.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 1)
                    except Exception:
                        pass

                self._windows_created.add(name)

    def _normalize_method_key(self, method):
        """
        Normalisiert den Methoden-String auf einen reinen Methoden-Key.

        Beispiele:
        - "abs" -> "abs"
        - "Abs" -> "abs"
        - "abs | double 19" -> "abs"
        - "fusion | Triple 20" -> "fusion"

        Zweck:
        Der linke Teil vor "|" beschreibt die Erkennungsmethode,
        der rechte Teil optional das erkannte Feld.
        """
        if method is None:
            return None

        # In String umwandeln, kleinschreiben, Leerzeichen außen entfernen
        m = str(method).lower().strip()

        # Nur der linke Teil vor "|" ist der Methoden-Key
        if "|" in m:
            m = m.split("|", 1)[0].strip()

        return m

    def _extract_field_label(self, method):
        """
        Extrahiert den rechten Textanteil hinter "|" als Feldbezeichnung.

        Beispiele:
        - "abs | Double 19" -> "Double 19"
        - "fusion | Triple 20" -> "Triple 20"
        - "vec" -> None

        Rückgabe:
        - String mit Feldlabel
        - oder None, wenn kein "|" vorhanden ist
        """
        if method is None:
            return None

        m = str(method).strip()
        if "|" in m:
            return m.split("|", 1)[1].strip()

        return None

    def _get_method_color(self, method):
        """
        Ordnet einer Methode eine BGR-Farbe zu.

        Achtung:
        OpenCV verwendet standardmäßig BGR statt RGB.

        Farbzuordnung:
        - fusion / combined / kombi / "+" -> cyan
        - abs   -> rot
        - vec   -> gelb
        - shape -> magenta
        - unbekannt / None -> weiß
        """
        key = self._normalize_method_key(method)

        if key is None:
            return (255, 255, 255)

        # Fusion / kombinierte Methoden
        if "+" in key or "fusion" in key or "combined" in key or "kombi" in key:
            return (255, 255, 0)  # cyan

        # Einzelmethoden
        if key == "abs":
            return (0, 0, 255)    # rot
        if key == "vec":
            return (0, 255, 255)  # gelb
        if key == "shape":
            return (255, 0, 255)  # magenta

        # Fallback-Farbe
        return (255, 255, 255)

    def _draw_tip_marker(self, img, x, y, color, inner_radius=8, outer_radius=16):
        """
        Zeichnet einen Treffer-/Tip-Marker auf ein Bild.

        Darstellung:
        - innerer gefüllter Kreis
        - äußerer Ring

        Parameter:
        - img          : Zielbild
        - x, y         : Markerposition
        - color        : BGR-Farbe
        - inner_radius : Radius des inneren Kreises
        - outer_radius : Radius des äußeren Rings
        """
        cv2.circle(img, (int(x), int(y)), inner_radius, color, -1)
        cv2.circle(img, (int(x), int(y)), outer_radius, color, 2)

    def _fit_to_canvas(self, img, canvas_w, canvas_h, pad=20):
        """
        Bild proportional in feste Zielgröße einpassen und mittig platzieren.

        Vorgehen:
        1. Nutzbaren Bereich berechnen (Canvas minus Padding)
        2. Skalierungsfaktor so wählen, dass das Bild vollständig hineinpasst
        3. Bild proportional skalieren
        4. Auf schwarzen Canvas zentriert einfügen

        Vorteil:
        Auch Bilder mit unterschiedlichem Seitenverhältnis können sauber
        im selben Debugfenster dargestellt werden, ohne verzerrt zu werden.
        """
        h, w = img.shape[:2]

        # Nutzbarer Bereich innerhalb des Canvas
        usable_w = max(1, canvas_w - 2 * pad)
        usable_h = max(1, canvas_h - 2 * pad)

        # Kleineren Skalierungsfaktor wählen, damit das Bild komplett passt
        scale = min(usable_w / w, usable_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        # Bild proportional skalieren
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Schwarze Ziel-Fläche erzeugen
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # Offset berechnen, damit das Bild zentriert wird
        x_off = (canvas_w - new_w) // 2
        y_off = (canvas_h - new_h) // 2

        # Skaliertes Bild mittig einsetzen
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        return canvas

    def show(
        self,
        cam_id,
        frame_bgr,
        H_cam_to_board,
        tip_full=None,
        tip_board=None,
        line_full=None,
        method=None,
        conf=None,
        extra_lines=None,
    ):
        """
        Zeigt die Debug-Ansichten für eine Kamera an.

        Parameter:
        - cam_id:
          ID oder Nummer der Kamera, wird im Fenstertitel angezeigt

        - frame_bgr:
          Originalbild der Kamera im BGR-Format

        - H_cam_to_board:
          Homographie-Matrix für die Transformation vom Kamerabild
          in den Board-/Warp-Raum

        - tip_full:
          Trefferpunkt im Originalbild als (x, y)

        - tip_board:
          Trefferpunkt im Boardspace als (x, y), typischerweise
          in einem 600x600-Koordinatensystem

        - line_full:
          Optionale Hauptlinie im Originalbild als (x1, y1, x2, y2)

        - method:
          Methode bzw. Anzeigetext, z. B.
          "abs", "vec", "fusion | Triple 20"

        - conf:
          Optionaler Confidence-Wert, wird im HUD angezeigt

        - extra_lines:
          Weitere optionale Linien im Originalbild,
          Liste von Linien im Format (x1, y1, x2, y2)
        """
        if not self.enabled:
            # Keine Ausgabe, wenn Debugging deaktiviert ist
            return

        # Fenster bei Bedarf anlegen
        self._ensure_windows(cam_id)

        # Farbe passend zur Methode bestimmen
        method_color = self._get_method_color(method)

        # Optionales Feldlabel aus dem Methoden-String extrahieren
        field_label = self._extract_field_label(method)

        # ---------- FULL FRAME ----------
        if self.show_full:
            # Kopie des Originalbilds, damit das Eingangssignal
            # nicht direkt verändert wird
            full = frame_bgr.copy()

            # Hauptlinie einzeichnen, falls vorhanden
            if line_full is not None:
                x1, y1, x2, y2 = line_full
                cv2.line(full, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

            # Zusätzliche Hilfslinien einzeichnen
            if extra_lines:
                for l in extra_lines:
                    if l is None:
                        continue
                    x1, y1, x2, y2 = l
                    cv2.line(full, (int(x1), int(y1)), (int(x2), int(y2)), (255, 200, 0), 1)

            # Trefferpunkt im FULL-Bild einzeichnen
            if tip_full is not None:
                tx, ty = tip_full
                self._draw_tip_marker(full, tx, ty, method_color, inner_radius=8, outer_radius=16)

            # HUD-Grundtext aufbauen
            hud = f"Cam {cam_id}"
            if method:
                hud += f" | {method}"
            if conf is not None:
                hud += f" | conf={conf:.1f}"

            # HUD oben links anzeigen
            cv2.putText(
                full,
                hud,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2
            )

            # Koordinaten des Trefferpunkts im Full-Frame anzeigen
            if tip_full is not None:
                cv2.putText(
                    full,
                    f"tip_full=({tip_full[0]:.1f},{tip_full[1]:.1f})",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (220, 220, 220),
                    2
                )

            # Falls ein Feldlabel existiert, separat anzeigen
            if field_label:
                cv2.putText(
                    full,
                    f"Field: {field_label}",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    method_color,
                    2
                )

            # Bild in feste Fenstergröße einpassen
            full_display = self._fit_to_canvas(
                full,
                self.full_window_w,
                self.full_window_h,
                pad=20
            )

            # FULL-Fenster anzeigen
            cv2.imshow(f"Cam {cam_id} - FULL", full_display)

        # ---------- WARP / BOARD SPACE ----------
        if self.show_warp and H_cam_to_board is not None:
            # Kamerabild per Homographie in den Boardspace transformieren
            warp = cv2.warpPerspective(frame_bgr, H_cam_to_board, (self.warp_size, self.warp_size))

            # Mittelpunkt markieren
            c = self.warp_size // 2
            cv2.circle(warp, (c, c), 4, (255, 255, 255), -1)

            # Trefferpunkt im Boardspace einzeichnen
            if tip_board is not None:
                bx, by = tip_board

                # Umrechnung vom angenommenen 600x600-Boardspace
                # auf die tatsächliche warp_size-Darstellung
                sx = int((bx / 600.0) * self.warp_size)
                sy = int((by / 600.0) * self.warp_size)

                self._draw_tip_marker(warp, sx, sy, method_color, inner_radius=6, outer_radius=14)

                # Board-Koordinaten anzeigen
                cv2.putText(
                    warp,
                    f"tip_board=({bx:.1f},{by:.1f})",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2
                )

            # Große Feldanzeige oben
            if field_label:
                cv2.putText(
                    warp,
                    field_label,
                    (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    method_color,
                    3
                )

            # HUD für Warp-Ansicht
            warp_hud = f"Cam {cam_id}"
            if method:
                warp_hud += f" | {method}"
            if conf is not None:
                warp_hud += f" | conf={conf:.1f}"

            # HUD unten platzieren
            cv2.putText(
                warp,
                warp_hud,
                (20, self.warp_size - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                method_color,
                2
            )

            # Warp-Bild in Fenstergröße einpassen
            warp_display = self._fit_to_canvas(
                warp,
                self.warp_window_w,
                self.warp_window_h,
                pad=40
            )

            # WARP-Fenster anzeigen
            cv2.imshow(f"Cam {cam_id} - WARP", warp_display)

        # Kurzes waitKey ist nötig, damit OpenCV die Fenster aktualisiert
        cv2.waitKey(1)

    def close(self):
        """
        Schließt alle OpenCV-Debugfenster und leert die interne Fensterliste.

        Diese Methode sollte am Ende sauber aufgerufen werden,
        um offene GUI-Fenster aufzuräumen.
        """
        if not self.enabled:
            return

        try:
            cv2.destroyAllWindows()
        except Exception:
            # Sicherheitshalber Fehler beim Fensterschließen ignorieren
            pass

        self._windows_created.clear()
