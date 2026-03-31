import cv2
import numpy as np


class ShapeDetector:
    """
    Kontur-/Form-basierte Darterkennung im bereits entzerrten Board-Bild
    ("Boardspace", typischerweise 600x600 Pixel).

    Grundidee:
    - Es wird ein Referenzbild ohne Dart gespeichert.
    - Ein aktuelles Frame wird mit diesem Referenzbild verglichen.
    - Aus der Differenz werden neue Objekte/Konturen extrahiert.
    - Über geometrische Merkmale wird versucht, dart-ähnliche Formen
      von Störungen, Flights oder unsauberen Blob-Strukturen zu trennen.
    - Als Ergebnis wird eine vermutete Dartspitze im Board-Koordinatensystem
      zurückgegeben.

    Rückgabeformat pro erkanntem Objekt:
    {
        "tip_board": (x, y),        # geschätzte Spitze im Boardspace
        "confidence": float,        # Heuristik-Score
        "contour": cnt,             # Originalkontur aus OpenCV
        "extra": {
            "area": ...,
            "length": ...,
            "width": ...,
            "elongation": ...,
            "solidity": ...
        }
    }
    """

    def __init__(self, board_mask, freeze_mean=20, freeze_max=70):
        """
        Initialisiert den ShapeDetector.

        Parameter:
        - board_mask:
            Binärmaske des relevanten Boardbereichs. Alles außerhalb
            des Boards wird später weggefiltert.
        - freeze_mean:
            Schwellwert für den mittleren Differenzwert. Dient zur
            Erkennung von globalem Wackeln / Kamera-Bewegung.
        - freeze_max:
            Obergrenze für den maximalen Differenzwert. Zusammen mit
            freeze_mean hilft das, "leichtes globales Zittern" von
            echten lokalen Änderungen (z. B. Dart) zu unterscheiden.
        """
        self.board_mask = board_mask

        # Graustufen-Referenzbild des leeren Boards.
        # Wird über set_reference(...) gesetzt.
        self.reference_gray = None

        # Parameter zum Ausblenden globaler Bildbewegung.
        self.FREEZE_MEAN = float(freeze_mean)
        self.FREEZE_MAX = float(freeze_max)

        # ---------------------------
        # Tuning-Parameter / Filter
        # ---------------------------

        # Minimal erlaubte Konturfläche.
        # Kleinere Konturen sind meist Rauschen oder sehr kleine Artefakte.
        self.min_area = 300

        # Maximal erlaubte Konturfläche.
        # Größere Flächen deuten eher auf massive Bildänderungen,
        # Fehlsegmentierung oder große Störungen hin.
        self.max_area = 16000

        # Minimal erforderliche Länge der minAreaRect-Hauptausdehnung.
        self.min_length = 16.0

        # Mindestverhältnis Länge/Breite.
        # Dart-ähnliche Objekte sollen deutlich länglich sein.
        self.min_elongation = 3.5

        # Maximal erlaubte Breite.
        # Zu breite Objekte sind meist keine Dartschäfte / Spitzenregionen.
        self.max_width = 40.0

        # Abstand, unterhalb dessen zwei Kandidaten als "gleiches Objekt"
        # betrachtet und zusammengeführt werden.
        self.merge_dist = 20.0

    def set_reference(self, frame_bgr):
        """
        Speichert ein Referenzbild des leeren Boards in Graustufen.

        Dieses Referenzbild wird später für absdiff(...) benutzt,
        um neue Objekte zu erkennen.
        """
        self.reference_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    def detect(self, warped_frame_bgr, gray=None):
        """
        Führt die eigentliche Form-/Konturerkennung auf einem entzerrten
        Board-Frame aus.

        Parameter:
        - warped_frame_bgr:
            Aktuelles Board-Bild in BGR.
        - gray:
            Optional bereits vorbereitete Graustufen-Version des Frames.
            Falls None, wird sie hier erzeugt.

        Rückgabe:
        - merged:
            Liste gefilterter / zusammengeführter Kandidaten.
        - debug_img:
            Debug-Bild mit eingezeichneten vermuteten Spitzen.
        """

        # Ohne Referenzbild ist keine Differenz-basierte Erkennung möglich.
        if self.reference_gray is None:
            return [], warped_frame_bgr

        # Falls kein Graustufenbild übergeben wurde, hier erzeugen.
        if gray is None:
            gray = cv2.cvtColor(warped_frame_bgr, cv2.COLOR_BGR2GRAY)

        # Absolute Differenz zwischen Referenz und aktuellem Frame.
        # Helle Pixel bedeuten: Hier hat sich etwas verändert.
        diff = cv2.absdiff(self.reference_gray, gray)

        # Bildgröße bestimmen.
        h, w = diff.shape[:2]

        # Boardzentrum im Boardspace.
        # Dieses Zentrum wird später benutzt, um die Dartachse zu orientieren
        # und die vermutete Spitze eher in Richtung Boardmitte zu suchen.
        board_center = np.array([w / 2.0, h / 2.0], dtype=np.float32)

        # -------------------------------------------------
        # Globales Wackeln / Kamerazittern grob ausblenden
        # -------------------------------------------------
        #
        # Idee:
        # - Wenn die durchschnittliche Differenz recht hoch ist,
        #   aber kein einzelner Pixel extrem heraussticht,
        #   handelt es sich oft eher um leichtes globales Wackeln
        #   als um ein neues lokales Objekt.
        mean_val = cv2.mean(diff)[0]
        _, max_val, _, _ = cv2.minMaxLoc(diff)

        if mean_val > self.FREEZE_MEAN and max_val < self.FREEZE_MAX:
            return [], warped_frame_bgr

        # ----------------
        # Vorverarbeitung
        # ----------------

        # Leichtes Glätten der Differenz, um Pixelrauschen zu reduzieren
        # und die spätere Binarisierung robuster zu machen.
        diff = cv2.GaussianBlur(diff, (5, 5), 0)

        # Otsu-Binarisierung:
        # Automatische Schwellwertwahl zur Trennung von "unverändert" vs.
        # "verändert".
        _, thr = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Kleine Standard-Kerne für Morphologie.
        kernel3 = np.ones((3, 3), np.uint8)
        kernel5 = np.ones((5, 5), np.uint8)

        # Opening:
        # Entfernt kleine helle Flecken / Rauschen.
        thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel3, iterations=1)

        # Closing:
        # Schließt kleine Löcher / Lücken in den erkannten Blobs.
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel5, iterations=1)

        # Alles außerhalb des Boards maskieren.
        thr = cv2.bitwise_and(thr, self.board_mask)

        # Anteil weißer Pixel im Schwellbild.
        # Wenn fast nichts aktiv ist, lohnt sich die Kontursuche nicht.
        white_ratio = cv2.countNonZero(thr) / float(w * h)
        if white_ratio < 0.0002:
            return [], warped_frame_bgr

        # Externe Konturen extrahieren.
        # RETR_EXTERNAL: Nur äußere Konturen, keine Hierarchie der Innenlöcher.
        contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Debug-Bild zum Einzeichnen der Treffer.
        debug_img = warped_frame_bgr.copy()

        # Liste aller Rohkandidaten vor Zusammenführung.
        raw_objects = []

        # ----------------------------------------
        # Jede gefundene Kontur einzeln bewerten
        # ----------------------------------------
        for cnt in contours:
            # Flächeninhalt der Kontur.
            area = float(cv2.contourArea(cnt))

            # Fläche außerhalb des erlaubten Bereichs -> verwerfen.
            if area < self.min_area or area > self.max_area:
                continue

            # Für PCA / stabile Formauswertung brauchen wir genug Punkte.
            if len(cnt) < 5:
                continue

            # Konturpunkte in ein einfaches Nx2-Float-Array umformen.
            pts = cnt.reshape(-1, 2).astype(np.float32)

            # --------------------------------------
            # minAreaRect: minimale umschließende Box
            # --------------------------------------
            #
            # Liefert die kleinste gedrehte Bounding Box.
            # Daraus lesen wir Länge und Breite ab.
            rect = cv2.minAreaRect(cnt)
            rw, rh = rect[1]

            # Länge = größere Ausdehnung, Breite = kleinere Ausdehnung.
            length = float(max(rw, rh))
            width = float(max(1.0, min(rw, rh)))  # Schutz vor Division durch 0

            # Zu kurze Objekte ignorieren.
            if length < self.min_length:
                continue

            # Zu breite Objekte ignorieren.
            if width > self.max_width:
                continue

            # Elongation = Längen-/Breitenverhältnis.
            # Große Werte bedeuten "lang und schmal".
            elongation = length / width
            if elongation < self.min_elongation:
                continue

            # ----------
            # Solidity
            # ----------
            #
            # Solidity = Konturfläche / Fläche der konvexen Hülle.
            # Werte nahe 1 bedeuten kompakte, relativ "volle" Formen.
            hull = cv2.convexHull(cnt)
            hull_area = float(cv2.contourArea(hull))

            # Sicherheitscheck gegen degenerierte Hüllen.
            if hull_area <= 1.0:
                continue

            solidity = area / hull_area

            # ------------------------
            # PCA auf den Konturpunkten
            # ------------------------
            #
            # Liefert Schwerpunkt + Hauptrichtungen der Punktwolke.
            # Die erste Eigenvektor-Richtung entspricht der Hauptachse
            # des länglichen Objekts.
            mean, eigenvecs = cv2.PCACompute(pts, mean=None)
            center = mean[0]
            axis = eigenvecs[0]

            # Hauptachse so orientieren, dass sie in Richtung Boardzentrum zeigt.
            # Dadurch wird "vorne" später konsistent als Richtung Brettmitte definiert.
            if np.dot(axis, (board_center - center)) < 0:
                axis = -axis

            # Projektion aller Konturpunkte auf die Hauptachse.
            # Positive Werte liegen "vor" dem Schwerpunkt in Achsenrichtung.
            projections = np.dot(pts - center, axis)

            # Nur Punkte betrachten, die auf der "vorderen" Seite liegen,
            # also in Richtung Boardzentrum.
            forward_pts = pts[projections > 0]

            if len(forward_pts) >= 3:
                # Gute Schätzung:
                # Von den Vorwärts-Punkten die board-nächsten Punkte nehmen
                # und daraus mitteln. Das stabilisiert die Spitzenschätzung
                # gegenüber Ausreißern.
                d = np.linalg.norm(forward_pts - board_center, axis=1)
                idx = np.argsort(d)[: min(5, len(forward_pts))]
                tip = np.mean(forward_pts[idx], axis=0)
            else:
                # Fallback:
                # Wenn die Vorwärts-Seite zu wenig Punkte hat, allgemein die
                # board-nächsten Konturpunkte mitteln.
                d = np.linalg.norm(pts - board_center, axis=1)
                idx = np.argsort(d)[: min(5, len(pts))]
                tip = np.mean(pts[idx], axis=0)

            # Abstand der geschätzten Spitze zum Boardzentrum.
            tip_dist = float(np.linalg.norm(board_center - tip))

            # Abstand des Konturschwerpunkts zum Boardzentrum.
            center_dist = float(np.linalg.norm(board_center - center))

            # Heuristik gegen Flight-/Heck-Treffer:
            #
            # Wenn die geschätzte "Spitze" deutlich weiter außen liegt als
            # der Schwerpunkt der Form, ist das oft kein Tip, sondern eher
            # ein hinterer Objektteil / Flight-artige Struktur.
            if tip_dist > center_dist + 14:
                continue

            # -----------------
            # Confidence-Scoring
            # -----------------
            #
            # Heuristische Gewichtung:
            # - größere Fläche -> tendenziell stabiler
            # - höhere Elongation -> dart-ähnlicher
            # - vernünftige Solidity -> kompaktere Form
            # - Nähe zur Boardmitte gibt kleinen Zusatzbonus
            center_bonus = 1.0 / (1.0 + tip_dist / 280.0)
            shape_bonus = min(elongation / 6.0, 2.0)
            solidity_bonus = min(max(solidity, 0.3), 1.2)

            confidence = area * shape_bonus * solidity_bonus * (1.0 + center_bonus)

            # Kandidat speichern.
            raw_objects.append({
                "tip_board": (float(tip[0]), float(tip[1])),
                "confidence": float(confidence),
                "contour": cnt,
                "extra": {
                    "area": area,
                    "length": length,
                    "width": width,
                    "elongation": float(elongation),
                    "solidity": float(solidity)
                }
            })

        # ---------------------------------
        # Nahe Kandidaten zusammenfassen
        # ---------------------------------
        #
        # Es kann passieren, dass mehrere ähnliche Konturen praktisch
        # auf dieselbe Spitze zeigen. Dann behalten wir nur den stärksten.
        raw_objects.sort(key=lambda o: o["confidence"], reverse=True)

        merged = []
        for obj in raw_objects:
            keep = True

            for m in merged:
                # Liegen zwei Spitzen sehr nah beieinander, wird der spätere
                # Kandidat verworfen, da bereits ein besser bewerteter
                # Kandidat vorhanden ist.
                if np.linalg.norm(np.array(obj["tip_board"]) - np.array(m["tip_board"])) < self.merge_dist:
                    keep = False
                    break

            if keep:
                merged.append(obj)

        # Die besten Treffer zur Kontrolle ins Debug-Bild einzeichnen.
        for obj in merged[:10]:
            tx, ty = obj["tip_board"]
            cv2.circle(debug_img, (int(tx), int(ty)), 5, (255, 0, 255), -1)

        return merged, debug_img
