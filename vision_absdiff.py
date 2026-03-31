import cv2
import numpy as np


class AbsDiffDetector:
    """
    AbsDiff-Detector im Boardspace (warped 600x600).

    Idee:
    - Es wird immer gegen ein gesetztes Referenzbild gearbeitet.
    - Das aktuelle Warped-Bild wird in Graustufen + Blur vorbereitet.
    - Aus der Absolutdifferenz zwischen Referenz und aktuellem Bild
      wird eine Änderungsmaske erzeugt.
    - Aus dieser Maske werden Konturen extrahiert und per Geometrie-
      Heuristiken auf dart-ähnliche Kandidaten geprüft.
    - Für jede plausible Kontur werden beide Enden als mögliche Tip-Punkte
      vorgeschlagen, weil aus der Form allein nicht immer eindeutig ist,
      welche Seite die Spitze ist.

    Rückgabe von detect_mask_candidates(...):
    {
        "mask": thr,              # finale binäre Änderungsmaske
        "dbg": dbg,               # Debug-Bild mit eingezeichneten Kandidaten
        "candidates": [...],      # gefilterte / gemergte Kandidaten
        "reject_stats": {...},    # Statistik, warum Konturen verworfen wurden
        "meta": {...}             # Meta-Infos zur Schwellenbildung
    }
    """

    def __init__(self, board_mask, freeze_mean=20, freeze_max=70):
        """
        Initialisiert den Detector.

        Parameter:
        - board_mask:
          Binärmaske des gültigen Board-Bereichs im Warped-Bild.
          Erwartung: aktive Pixel > 0.
        - freeze_mean:
          Schwelle für mittlere globale Änderung innerhalb der Board-Maske.
          Wenn der Mittelwert hoch ist, aber der Maximalwert nicht extrem,
          wird das als leichtes globales Wackeln / Helligkeitsschwanken
          interpretiert und ignoriert.
        - freeze_max:
          Obergrenze für den Maximalwert in dieser Freeze-Heuristik.

        Interne Parameter:
        - min_area / max_area:
          erlaubte Konturfläche für Kandidaten.
        - min_length:
          minimale Längsausdehnung entlang der Hauptachse.
        - merge_dist:
          Kandidaten, die näher als dieser Abstand liegen, werden
          zusammengefasst.
        - min_diff_threshold:
          untere Grenze für den endgültigen Threshold bei der
          Differenzsegmentierung.
        """
        self.board_mask = board_mask
        self.reference_frame = None

        # Freeze-Heuristik:
        # Wenn das Bild global "ein bisschen anders" ist, aber keine klaren
        # starken lokalen Änderungen enthält, nehmen wir eher Wackeln /
        # leichte Beleuchtungsänderung an statt eines echten Darts.
        self.FREEZE_MEAN = float(freeze_mean)
        self.FREEZE_MAX = float(freeze_max)

        # Geometrische Filter für Konturen:
        # "etwas toleranter" bedeutet: eher mehr Kandidaten zulassen und
        # später mergen / fusionieren.
        self.min_area = 160
        self.max_area = 18000
        self.min_length = 12
        self.merge_dist = 20

        # Minimaler Schwellwert für die Differenzmaske.
        # Otsu darf nie unter diesen Wert fallen.
        self.min_diff_threshold = 18

        # Innere Board-Maske:
        # Durch Erosion wird der Rand etwas abgeschnitten.
        # Das reduziert Artefakte am äußersten Rand des Boards.
        kernel_mask = np.ones((9, 9), np.uint8)
        self.inner_board_mask = cv2.erode(self.board_mask, kernel_mask, iterations=1)

        # Weitere Form-Heuristiken für dart-ähnliche Objekte:
        # - max_width: Objekt darf nicht zu breit sein
        # - min_slenderness: Länge/Breite muss ausreichend groß sein
        # - min_aspect: BoundingBox darf nicht zu quadratisch sein
        self.max_width = 52.0
        self.min_slenderness = 1.2
        self.min_aspect = 1.02

    def _prepare_gray(self, frame_bgr):
        """
        Wandelt ein BGR-Bild in ein geglättetes Graubild um.

        Warum Blur?
        - reduziert Pixelrauschen
        - macht die AbsDiff stabiler
        - kleine Sensor- oder Kompressionsartefakte stören weniger
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        return gray

    def set_reference(self, frame_bgr):
        """
        Setzt das Referenzbild, gegen das später verglichen wird.

        Das Referenzbild wird direkt in der gleichen vorbereiteten Form
        gespeichert, in der später auch aktuelle Frames verglichen werden.
        """
        self.reference_frame = self._prepare_gray(frame_bgr)

    def detect(self, warped_frame_bgr, gray=None):
        """
        Kompatibilitätsmethode zu altem Code.

        Früher wurde offenbar nur (candidates, dbg) erwartet.
        Intern wird inzwischen detect_mask_candidates(...) verwendet,
        aber hier wird das alte Rückgabeformat nachgebildet.
        """
        result = self.detect_mask_candidates(warped_frame_bgr, gray=gray)
        return result["candidates"], result["dbg"]

    def _build_threshold_mask(self, warped_frame_bgr, gray=None):
        """
        Erzeugt aus Referenzbild und aktuellem Bild eine binäre Änderungsmaske.

        Ablauf:
        1. aktuelles Bild vorbereiten (Gray + Blur)
        2. Absolutdifferenz zum Referenzbild bilden
        3. auf innere Board-Maske beschränken
        4. Freeze-Heuristik prüfen
        5. Otsu + Mindestschwelle verwenden
        6. Plausibilität über white_ratio prüfen
        7. Morphologie anwenden
        8. finale Maske + Meta-Infos zurückgeben

        Rückgabe:
        - thr: finale binäre Maske
        - diff_masked: rohe Differenz im Board-Bereich
        - meta: Diagnoseinfos oder Abbruchgrund
        """
        # Ohne Referenz ist kein sinnvoller AbsDiff-Vergleich möglich.
        if self.reference_frame is None:
            return None, None, {"reason": "no_reference"}

        # Falls kein Gray-Bild mitgegeben wurde, hier selbst vorbereiten.
        # Falls doch eins mitgegeben wurde, wird es trotzdem noch geglättet,
        # damit es konsistent zum Referenzbild ist.
        if gray is None:
            gray = self._prepare_gray(warped_frame_bgr)
        else:
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Pixelweise Absolutdifferenz zwischen Referenz und aktuellem Frame.
        diff = cv2.absdiff(self.reference_frame, gray)

        # Nur der innere Board-Bereich ist relevant.
        diff_masked = cv2.bitwise_and(diff, self.inner_board_mask)

        # Nur aktive Pixel innerhalb der Maskenfläche betrachten.
        mask_pixels = diff_masked[self.inner_board_mask > 0]
        if mask_pixels.size == 0:
            return None, None, {"reason": "empty_mask"}

        # Statistiken über die Änderung:
        # - mean_val: mittlere Änderung
        # - max_val: stärkste lokale Änderung
        mean_val = float(np.mean(mask_pixels))
        max_val = float(np.max(mask_pixels))

        # Freeze-Heuristik:
        # Wenn der Mittelwert hoch ist, aber kein sehr starker Peak existiert,
        # deutet das eher auf globales Wackeln oder Helligkeitseffekt hin,
        # nicht auf einen lokalen Dart.
        if mean_val > self.FREEZE_MEAN and max_val < self.FREEZE_MAX:
            return None, None, {
                "reason": "freeze",
                "mean_val": mean_val,
                "max_val": max_val,
            }

        # Zusätzliche Glättung der Differenz vor dem Thresholding.
        diff_blur = cv2.GaussianBlur(diff_masked, (5, 5), 0)

        # Otsu bestimmt automatisch einen sinnvollen Schwellwert.
        # cv2.threshold gibt den gewählten Schwellwert als erstes zurück.
        otsu_thr, _ = cv2.threshold(
            diff_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Otsu darf nicht unter die feste Untergrenze fallen.
        final_thr = max(self.min_diff_threshold, int(otsu_thr))

        # Finale Binärmaske erzeugen.
        _, thr = cv2.threshold(diff_blur, final_thr, 255, cv2.THRESH_BINARY)

        # Anzahl aktiver Pixel in der inneren Board-Maske.
        mask_nonzero = float(cv2.countNonZero(self.inner_board_mask))
        if mask_nonzero <= 0:
            return None, None, {"reason": "mask_zero"}

        # Verhältnis weißer Pixel in der Änderungsmaske zur gesamten Maskenfläche.
        # Das dient als Plausibilitätscheck:
        # - zu klein => vermutlich kein sinnvoller Treffer
        # - zu groß  => vermutlich globale Störung
        white_ratio = cv2.countNonZero(thr) / mask_nonzero

        if white_ratio < 0.00008:
            return None, None, {
                "reason": "too_small",
                "white_ratio": white_ratio,
            }

        if white_ratio > 0.12:
            return None, None, {
                "reason": "too_large",
                "white_ratio": white_ratio,
            }

        # Morphologische Nachbearbeitung:
        # OPEN  = kleine isolierte Störungen entfernen
        # CLOSE = Lücken in Strukturen schließen
        kernel_open = np.ones((3, 3), np.uint8)
        kernel_close = np.ones((5, 5), np.uint8)

        thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel_open, iterations=1)
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        # Leichte Dilatation:
        # dünne Dart-Anteile sollen nicht abbrechen.
        kernel_dilate = np.ones((3, 3), np.uint8)
        thr = cv2.dilate(thr, kernel_dilate, iterations=1)

        # Sicherheitshalber erneut auf den inneren Board-Bereich begrenzen.
        thr = cv2.bitwise_and(thr, self.inner_board_mask)

        # Diagnose-/Debug-Metadaten.
        meta = {
            "mean_val": mean_val,
            "max_val": max_val,
            "white_ratio": white_ratio,
            "threshold": final_thr,
        }
        return thr, diff_masked, meta

    def detect_mask_candidates(self, warped_frame_bgr, gray=None):
        """
        Hauptmethode:
        - baut die Änderungsmaske
        - extrahiert Konturen
        - filtert sie geometrisch
        - erzeugt Kandidaten für mögliche Tip-Positionen
        - merged nahe Kandidaten
        - zeichnet die finalen Kandidaten ins Debug-Bild

        Rückgabe:
        {
            "mask": thr,
            "dbg": dbg,
            "candidates": merged,
            "reject_stats": reject_stats,
            "meta": meta,
        }
        """
        # Änderungsmaske erzeugen.
        thr, _diff_masked, meta = self._build_threshold_mask(warped_frame_bgr, gray=gray)

        # Debug-Bild startet als Kopie des Eingabebilds.
        dbg = warped_frame_bgr.copy()

        # Statistik darüber, warum Kandidaten verworfen wurden.
        reject_stats = {
            "area": 0,         # Fläche außerhalb erlaubtem Bereich
            "points": 0,       # zu wenige Konturpunkte
            "length": 0,       # zu kurz
            "width": 0,        # zu breit
            "slenderness": 0,  # nicht schlank genug
            "aspect": 0,       # BoundingBox zu wenig gestreckt
        }

        # Wenn keine valide Maske gebaut werden konnte, direkt mit leerem
        # Kandidaten-Set zurück.
        if thr is None:
            return {
                "mask": None,
                "dbg": dbg,
                "candidates": [],
                "reject_stats": reject_stats,
                "meta": meta,
            }

        # Äußere Konturen der Änderungsmaske extrahieren.
        contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw_candidates = []

        for cnt in contours:
            # Konturfläche bestimmen.
            area = float(cv2.contourArea(cnt))

            # Zu kleine / zu große Flächen verwerfen.
            if area < self.min_area or area > self.max_area:
                reject_stats["area"] += 1
                continue

            # PCA braucht genügend Punkte; sehr kleine Konturen sind zudem
            # geometrisch meist unbrauchbar.
            if len(cnt) < 5:
                reject_stats["points"] += 1
                continue

            # Konturpunkte in flaches Nx2-Format überführen.
            pts = cnt.reshape(-1, 2).astype(np.float32)

            # PCA liefert Hauptachsen der Kontur.
            # - mean      = Schwerpunkt
            # - eigenvecs = Richtungen der Achsen
            mean, eigenvecs = cv2.PCACompute(pts, mean=None)
            center = mean[0]
            axis = eigenvecs[0]

            # Projektion aller Punkte auf die erste Hauptachse.
            # Dadurch findet man die beiden extremen Enden der längsten Richtung.
            proj = np.dot(pts - center, axis)

            i_min = int(np.argmin(proj))
            i_max = int(np.argmax(proj))

            p1 = pts[i_min]
            p2 = pts[i_max]

            # Länge als Abstand der beiden Extrempunkte.
            length = float(np.linalg.norm(p2 - p1))
            if length < self.min_length:
                reject_stats["length"] += 1
                continue

            # Breite grob als Fläche / Länge.
            # Das ist keine echte physikalische Breite, aber eine robuste
            # Näherung für "wie kompakt oder schlank" die Kontur ist.
            width = area / max(length, 1.0)
            width = max(width, 1.0)

            # Schlankheit: je höher, desto länglicher.
            slenderness = length / width

            if width > self.max_width:
                reject_stats["width"] += 1
                continue

            if slenderness < self.min_slenderness:
                reject_stats["slenderness"] += 1
                continue

            # Zusätzlicher Rechteck-Aspekt-Check.
            # Ein dart-ähnliches Objekt sollte eher gestreckt als quadratisch sein.
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = max(bw, bh) / max(1, min(bw, bh))
            if aspect < self.min_aspect:
                reject_stats["aspect"] += 1
                continue

            # Basiskonfidenz:
            # kombiniert Länge, Schlankheit und Fläche.
            # Größere, längere und schlankere Objekte bekommen höhere Scores.
            base_conf = length * slenderness * np.log(area + 1.0)

            # Statt exakt nur eines Endpunkts wird jeweils der Mittelwert
            # der 3 kleinsten und 3 größten Projektionen genutzt.
            # Das stabilisiert die Tip-Schätzung gegen Ausreißer.
            idxs_min = np.argsort(proj)[:3]
            idxs_max = np.argsort(proj)[-3:]

            tip_a = np.mean(pts[idxs_min], axis=0)
            tip_b = np.mean(pts[idxs_max], axis=0)

            # Beide Enden werden als mögliche Tipps eingetragen.
            # Warum beide?
            # Aus einer reinen Änderungskontur ist oft nicht eindeutig klar,
            # welches Ende wirklich die Spitze und welches der Schaft ist.
            raw_candidates.append({
                "tip_board": (float(tip_a[0]), float(tip_a[1])),
                "confidence": float(base_conf),
                "contour": cnt,
                "extra": {
                    "area": area,
                    "length": length,
                    "width": width,
                    "slenderness": slenderness,
                    "aspect": aspect,
                    "threshold": meta.get("threshold", None),
                    "white_ratio": meta.get("white_ratio", None),
                    "endpoint_side": "min_proj",
                }
            })

            raw_candidates.append({
                "tip_board": (float(tip_b[0]), float(tip_b[1])),
                "confidence": float(base_conf),
                "contour": cnt,
                "extra": {
                    "area": area,
                    "length": length,
                    "width": width,
                    "slenderness": slenderness,
                    "aspect": aspect,
                    "threshold": meta.get("threshold", None),
                    "white_ratio": meta.get("white_ratio", None),
                    "endpoint_side": "max_proj",
                }
            })

        # Höhere Konfidenz zuerst.
        raw_candidates.sort(key=lambda o: o["confidence"], reverse=True)
        merged = []

        # Kandidaten mergen:
        # Liegen zwei Tip-Schätzungen sehr nah beieinander, wird nur die
        # zuerst gefundene (also meist stärkere) behalten.
        for cand in raw_candidates:
            keep = True
            p = np.array(cand["tip_board"], dtype=np.float32)

            for m in merged:
                mp = np.array(m["tip_board"], dtype=np.float32)
                if np.linalg.norm(p - mp) < self.merge_dist:
                    keep = False
                    break

            if keep:
                merged.append(cand)

        # Die ersten 20 Kandidaten fürs Debug-Bild markieren.
        for o in merged[:20]:
            tx, ty = o["tip_board"]
            cv2.circle(dbg, (int(tx), int(ty)), 4, (0, 0, 255), -1)

        return {
            "mask": thr,
            "dbg": dbg,
            "candidates": merged,
            "reject_stats": reject_stats,
            "meta": meta,
        }


def _build_distance_map_from_mask(mask):
    """
    Baut aus einer binären Änderungsmaske eine Distanzkarte.

    Bedeutung:
    - Für jedes Pixel wird der Abstand zum nächsten aktiven Maskenpixel
      berechnet.
    - Ein Pixel innerhalb bzw. direkt auf einer Änderungsregion hat also
      kleinen Abstand.
    - Ein Pixel weit weg von allen Änderungen hat großen Abstand.

    Erwartung an mask:
    - 255 = Änderung / Treffer
    - 0   = kein Treffer

    Rückgabe:
    - dist: float-Matrix mit L2-Distanzen
    - None, falls die Maske leer ist
    """
    if mask is None or cv2.countNonZero(mask) == 0:
        return None

    # distanceTransform berechnet Distanzen zu Null-Pixeln.
    # Deshalb wird invertiert:
    # - Änderungs-Pixel werden 0
    # - Rest wird 255
    # Dann ist der Abstand eines Pixels zum nächsten Änderungsbereich
    # genau der gewünschte Wert.
    inv = np.where(mask > 0, 0, 255).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    return dist


def fuse_warped_masks(mask_list, board_mask, max_dist=22.0):
    """
    Fusioniert mehrere Änderungsmasken direkt im Warped-Boardspace.

    Grundidee:
    - Jede Kamera liefert eine eigene Änderungsmaske im gleichen
      Board-Koordinatensystem.
    - Für jede Maske wird eine Distanzkarte berechnet.
    - Gesucht wird der Punkt, der gleichzeitig zu allen Änderungsregionen
      möglichst nah liegt.
    - Das ist die finale Tip-Bestimmung.

    Warum Distanzkarten?
    - Exakte Überlappung der Masken ist in der Praxis selten.
    - Mit Distanzkarten kann man auch "fast deckungsgleiche" Hinweise
      robust kombinieren.

    Parameter:
    - mask_list:
      Liste von binären Warped-Masken aus mehreren Kameras.
    - board_mask:
      Gültiger Board-Bereich, damit außerhalb nichts gewählt wird.
    - max_dist:
      Maximale schlechteste Kameradistanz, die noch akzeptiert wird.
      Wenn der beste Punkt für mindestens eine Kamera zu weit weg liegt,
      wird kein Ergebnis geliefert.

    Rückgabe:
      {
        "tip_board": (x, y),   # finaler Tip im Boardspace
        "per_cam_dist": [...], # Distanz zu jeder verwendeten Kamera-Maske
        "used_cams": [...],    # Indizes der verwendeten Kameras
        "score": float,        # Optimierungsscore des besten Punkts
        "max_dist": float      # größte Einzel-Distanz unter den verwendeten Kameras
      }
    oder None
    """
    active = []

    # Nur Masken verwenden, die überhaupt Änderungen enthalten.
    for idx, m in enumerate(mask_list):
        if m is not None and cv2.countNonZero(m) > 0:
            active.append((idx, m))

    # Für eine Fusion werden mindestens zwei aktive Kameras erwartet.
    if len(active) < 2:
        return None

    dist_maps = []
    used_cams = []

    # Für jede aktive Maske Distanzkarte aufbauen.
    for idx, m in active:
        d = _build_distance_map_from_mask(m)
        if d is None:
            continue
        dist_maps.append(d)
        used_cams.append(idx)

    # Auch nach dem Distanzaufbau müssen noch mindestens zwei Kameras übrig sein.
    if len(dist_maps) < 2:
        return None

    # Suchregion:
    # Vereinigung aller Änderungen, anschließend etwas aufweiten.
    # Dadurch wird nicht das ganze Board abgesucht, sondern nur ein
    # plausibler Bereich um die gemeldeten Änderungen.
    union = np.zeros_like(board_mask, dtype=np.uint8)
    for _, m in active:
        union = cv2.bitwise_or(union, m)

    kernel = np.ones((9, 9), np.uint8)
    search_region = cv2.dilate(union, kernel, iterations=2)
    search_region = cv2.bitwise_and(search_region, board_mask)

    # Koordinaten aller erlaubten Suchpixel.
    ys, xs = np.where(search_region > 0)
    if len(xs) == 0:
        return None

    # Distanzkarten stapeln:
    # Form: [N, H, W], N = Anzahl verwendeter Kameras.
    stacked = np.stack(dist_maps, axis=0)

    # Distanzwerte nur an den Suchpunkten extrahieren:
    # Form: [N, P], P = Anzahl Suchpixel.
    vals = stacked[:, ys, xs]

    # Score-Funktion:
    # - Summe aller Distanzen bevorzugt Punkte, die insgesamt nah an allen
    #   Masken liegen.
    # - zusätzlicher max-Term bestraft Punkte, die bei einer Kamera stark
    #   abweichen.
    #
    # Gewicht 0.75 ist eine Heuristik zwischen "gut im Mittel" und
    # "nicht katastrophal in der schlechtesten Kamera".
    score = np.sum(vals, axis=0) + 0.75 * np.max(vals, axis=0)

    # Pixel mit minimalem Score ist der beste gemeinsame Kompromisspunkt.
    best_idx = int(np.argmin(score))

    bx = int(xs[best_idx])
    by = int(ys[best_idx])

    # Einzelabstände dieses besten Punkts pro Kamera.
    per_cam_dist = [float(stacked[i, by, bx]) for i in range(stacked.shape[0])]
    worst = max(per_cam_dist)

    # Plausibilitätscheck:
    # Wenn selbst der beste Punkt für eine Kamera zu weit weg liegt,
    # ist die Übereinstimmung zu schlecht -> kein Ergebnis.
    if worst > max_dist:
        return None

    return {
        "tip_board": (float(bx), float(by)),
        "per_cam_dist": per_cam_dist,
        "used_cams": used_cams,
        "score": float(score[best_idx]),
        "max_dist": float(worst),
    }
