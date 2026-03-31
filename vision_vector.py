import cv2
import numpy as np


class VectorDetector:
    """
    Vektor-/linienbasierte Darterkennung im Full-Frame.

    Grundidee:
    - Im gesamten Kamerabild werden Kanten per Canny extrahiert.
    - Daraus werden lineare Strukturen per HoughLinesP gesucht.
    - Für jede gefundene Linie wird angenommen:
      Der Endpunkt, der näher am Boardzentrum liegt, ist die Dartspitze ("tip"),
      der andere Endpunkt ist das hintere Ende ("tail").

    Ausgabe:
    - Eine Liste von Kandidaten mit:
      - tip: geschätzte Dartspitze im Full-Frame
      - confidence: heuristische Gütebewertung
      - line: die zugrunde liegende Linie
      - extra: zusätzliche Debug-/Analysewerte

    Wichtige Schutzmechanismen:
    - harte ROI-Maskierung: nur relevante Board-Bereiche berücksichtigen
    - Linien durch/nahe Boardzentrum verwerfen
    - typische Spider-/Boardlinien unterdrücken
    - nahe beieinanderliegende Tip-Kandidaten zusammenfassen
    """

    def __init__(self):
        # Untere/obere Schwelle für Canny-Kantendetektion.
        # Höhere Werte = weniger, aber robustere Kanten.
        self.canny1 = 70
        self.canny2 = 180

        # Vor dem Canny leicht weichzeichnen, um Bildrauschen zu reduzieren.
        self.blur = (5, 5)

        # Mindestanzahl "Votes" für HoughLinesP.
        # Je höher, desto selektiver werden nur deutliche Linien akzeptiert.
        self.hough_threshold = 100

        # Minimale Länge einer Linie in Pixeln.
        # Kurze Strukturen (Rauschen, kleine Kantenfragmente) werden damit unterdrückt.
        self.min_line_len = 80

        # Maximal erlaubte Lücke zwischen zwei Liniensegmenten,
        # damit HoughLinesP sie noch als eine Linie zusammenzieht.
        self.max_line_gap = 2

        # Maximal wie viele Endkandidaten am Ende zurückgegeben werden sollen.
        self.max_candidates = 3

        # Wenn zwei erkannte Tips näher als dieser Abstand liegen,
        # gelten sie als derselbe Kandidat und werden gemerged.
        self.tip_merge_dist = 22

        # ---- Filter gegen Spider-/Boardlinien ----

        # Tip darf nicht exakt im Zentrum liegen.
        # Das verhindert, dass zentrale Boardstrukturen als Dartspitze gewertet werden.
        self.min_tip_center_dist = 28

        # Tip darf auch nicht zu weit außerhalb liegen.
        # Das begrenzt die Suche auf einen plausiblen Bereich um das Board herum.
        self.max_tip_center_dist = 235

        # Der Linienmittelpunkt darf nicht zu nah am Zentrum liegen.
        # Spider-Linien und zentrale Boardstrukturen liegen oft sehr zentrumsnah.
        self.min_mid_center_dist = 70

        # Wenn eine Linie das Zentrum schneidet oder ihm zu nahe kommt,
        # ist sie eher eine Boardlinie als ein Dart.
        self.min_line_center_dist = 28

    def _point_line_distance(self, p, a, b):
        """
        Berechnet den Abstand eines Punkts p zu einem Liniensegment a-b.

        Wichtig:
        - Nicht Abstand zur unendlichen Geraden,
          sondern explizit zum Segment.
        - Falls das Segment praktisch ein Punkt ist, wird Punkt-zu-Punkt-Abstand benutzt.

        Verwendung hier:
        - Um zu prüfen, wie nahe eine gefundene Linie am Boardzentrum vorbeiläuft.
        - Linien, die das Zentrum schneiden, werden später verworfen.
        """
        p = np.array(p, dtype=np.float32)
        a = np.array(a, dtype=np.float32)
        b = np.array(b, dtype=np.float32)

        # Richtungsvektor des Segments
        ab = b - a

        # Quadrat der Segmentlänge
        ab_len2 = float(np.dot(ab, ab))

        # Degenerierter Fall: a und b praktisch identisch
        if ab_len2 < 1e-6:
            return float(np.linalg.norm(p - a))

        # Projektion von p auf das Segment parametrisiert mit t in [0,1]
        t = float(np.dot(p - a, ab) / ab_len2)
        t = max(0.0, min(1.0, t))

        # Tatsächlicher Projektionspunkt auf dem Segment
        proj = a + t * ab

        # Euklidischer Abstand von p zum Projektionspunkt
        return float(np.linalg.norm(p - proj))

    def _prepare_roi_mask(self, board_roi_mask_full):
        """
        Bereitet die Board-ROI für die Vektorerkennung vor.

        Idee:
        - Die übergebene ROI-Maske wird etwas nach innen erodiert.
        - Dadurch spielen Randbereiche und die äußere Boardstruktur weniger stark rein.

        Warum sinnvoll:
        - An Boardrändern entstehen oft starke, aber irrelevante Kanten.
        - Eine leicht kleinere ROI macht die Linienenkennung stabiler.
        """
        if board_roi_mask_full is None:
            return None

        # Erosionskernel: zieht die Maske leicht nach innen.
        kernel = np.ones((9, 9), np.uint8)

        # Eine Iteration Erosion reicht hier als konservative Innenverkleinerung.
        roi = cv2.erode(board_roi_mask_full, kernel, iterations=1)
        return roi

    def detect(self, frame_bgr, board_center_full=None, board_roi_mask_full=None):
        """
        Führt die linienbasierte Darterkennung auf einem BGR-Frame aus.

        Parameter:
        - frame_bgr: komplettes Kamerabild
        - board_center_full: Boardzentrum im Full-Frame
        - board_roi_mask_full: optionale ROI-Maske des Boards im Full-Frame

        Rückgabe:
        - Liste gemergter Kandidaten, sortiert nach Confidence
        """
        h, w = frame_bgr.shape[:2]

        # Falls kein Boardzentrum bekannt ist, wird als Fallback die Bildmitte verwendet.
        # Das ist nur ein Default und idealerweise durch echte Kalibrierung ersetzt.
        if board_center_full is None:
            board_center_full = (w / 2.0, h / 2.0)

        # Boardzentrum als Float-Vektor für Distanzrechnungen
        bc = np.array(board_center_full, dtype=np.float32)

        # In Graustufen umwandeln, da Canny nur Intensitätskanten braucht.
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Leicht glätten, um Rauschen vor der Kantendetektion zu reduzieren.
        gray = cv2.GaussianBlur(gray, self.blur, 0)

        # Kanten extrahieren
        edges = cv2.Canny(gray, self.canny1, self.canny2)

        # ROI vorbereiten und anwenden:
        # Nur Kanten innerhalb der Boardregion bleiben erhalten.
        roi_mask = self._prepare_roi_mask(board_roi_mask_full)
        if roi_mask is not None:
            edges = cv2.bitwise_and(edges, roi_mask)

        # Probabilistische Hough-Transformation:
        # Sucht Liniensegmente in den Kanten.
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_len,
            maxLineGap=self.max_line_gap
        )

        # Keine Linien gefunden => keine Kandidaten
        if lines is None:
            return []

        # Hier sammeln wir alle rohen, noch ungemergten Kandidaten.
        raw = []

        # Sicherheitslimit:
        # Nicht unbegrenzt alle Linien verarbeiten, damit Ausreißerframes nicht teuer werden.
        for l in lines[:250]:
            x1, y1, x2, y2 = l[0]

            # Segmentendpunkte als Float-Vektoren
            p1 = np.array([x1, y1], dtype=np.float32)
            p2 = np.array([x2, y2], dtype=np.float32)

            # Linienlänge
            length = float(np.linalg.norm(p2 - p1))

            # Zusätzlicher Schutz:
            # Auch wenn Hough schon minLineLength bekam, hier nochmal hart prüfen.
            if length < self.min_line_len:
                continue

            # Annahme:
            # Die Dartspitze ist der Endpunkt, der näher am Boardzentrum liegt.
            d1 = float(np.linalg.norm(bc - p1))
            d2 = float(np.linalg.norm(bc - p2))

            tip = p1 if d1 < d2 else p2
            tail = p2 if d1 < d2 else p1

            # Distanzen der Spitze und des Hecks zum Zentrum
            tip_dist = float(np.linalg.norm(bc - tip))
            tail_dist = float(np.linalg.norm(bc - tail))

            # Mittelpunkt der Linie
            mid = (p1 + p2) * 0.5
            mid_dist = float(np.linalg.norm(bc - mid))

            # ------------------------------------------------------------
            # Filter 1:
            # Tip nicht zu nah am Zentrum und nicht zu weit außen
            # ------------------------------------------------------------
            if tip_dist < self.min_tip_center_dist:
                continue

            if tip_dist > self.max_tip_center_dist:
                continue

            # ------------------------------------------------------------
            # Filter 2:
            # Linienmittelpunkt nicht zu zentrumsnah
            # ------------------------------------------------------------
            # Spider-Linien und zentrale Boardstrukturen haben häufig
            # ihren Schwerpunkt nahe am Zentrum.
            if mid_dist < self.min_mid_center_dist:
                continue

            # ------------------------------------------------------------
            # Filter 3:
            # Linie darf das Boardzentrum nicht schneiden / fast schneiden
            # ------------------------------------------------------------
            dist_line_to_center = self._point_line_distance(bc, p1, p2)
            if dist_line_to_center < self.min_line_center_dist:
                continue

            # ------------------------------------------------------------
            # Filter 4:
            # Tail muss plausibel weiter außen liegen als die Spitze
            # ------------------------------------------------------------
            # Ein Dart zeigt typischerweise von außen nach innen.
            # Wenn Tail nicht klar weiter draußen liegt, ist die Linie fragwürdig.
            if tail_dist <= tip_dist + 20:
                continue

            # ------------------------------------------------------------
            # Filter 5:
            # Zusätzliche ROI-Stabilisierung
            # ------------------------------------------------------------
            # Idee:
            # Wenn beide Endpunkte komplett außerhalb der ROI liegen,
            # dann ist die Linie vermutlich irrelevant.
            if roi_mask is not None:
                x1c = int(np.clip(x1, 0, w - 1))
                y1c = int(np.clip(y1, 0, h - 1))
                x2c = int(np.clip(x2, 0, w - 1))
                y2c = int(np.clip(y2, 0, h - 1))

                if roi_mask[y1c, x1c] == 0 and roi_mask[y2c, x2c] == 0:
                    continue

            # ------------------------------------------------------------
            # Confidence-Berechnung
            # ------------------------------------------------------------
            # Ziel:
            # Eine heuristische Bewertung, wie "dart-ähnlich" die Linie ist.
            #
            # Bestandteile:
            # - lange Linien sind besser
            # - Spitze eher moderat zentrumsnah ist gut
            # - großer Abstand der Linie zum Zentrum ist anti-spider-gut
            # - Tail deutlich weiter außen als Tip ist gut

            # Bonus für plausible Tip-Distanz:
            # Je weiter der Tip weg ist, desto kleiner der Bonus.
            center_bonus = 1.0 / (1.0 + (tip_dist / 260.0))

            # Linien mit größerem Abstand zum Zentrum werden bevorzugt,
            # weil zentrale Spider-/Boardlinien oft problematisch sind.
            anti_spider_bonus = min(1.5, dist_line_to_center / 35.0)

            # Verhältnis Tail zu Tip:
            # Wenn Tail deutlich weiter außen liegt, steigt der Bonus.
            tail_bonus = min(1.4, tail_dist / max(tip_dist, 1.0))

            # Endgültige heuristische Confidence
            confidence = length * (1.0 + 1.8 * center_bonus) * anti_spider_bonus * tail_bonus

            # Rohkandidat abspeichern
            raw.append({
                "tip": (float(tip[0]), float(tip[1])),
                "confidence": float(confidence),
                "line": (int(x1), int(y1), int(x2), int(y2)),
                "length": float(length),
                "extra": {
                    "tip_dist": float(tip_dist),
                    "tail_dist": float(tail_dist),
                    "mid_dist": float(mid_dist),
                    "line_center_dist": float(dist_line_to_center),
                }
            })

        # Beste Kandidaten zuerst
        raw.sort(key=lambda o: o["confidence"], reverse=True)

        # ------------------------------------------------------------
        # Merge ähnlicher Tip-Kandidaten
        # ------------------------------------------------------------
        # Mehrere Linien können denselben Dart beschreiben.
        # Deshalb werden Tips, die räumlich nah beieinander liegen, zusammengefasst.
        merged = []

        for obj in raw:
            keep = True

            for m in merged:
                # Wenn der neue Tip nahe bei einem schon akzeptierten Tip liegt,
                # wird er verworfen.
                if np.linalg.norm(np.array(obj["tip"]) - np.array(m["tip"])) < self.tip_merge_dist:
                    keep = False
                    break

            if keep:
                merged.append(obj)

            # Nur die besten N Kandidaten behalten
            if len(merged) >= self.max_candidates:
                break

        return merged
