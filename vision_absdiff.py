import cv2
import numpy as np


class AbsDiffDetector:
    """
    Foreground-/AbsDiff-Detector im Boardspace (warped 600x600).

    Ziel:
    - Änderungen gegenüber einem Referenzbild der leeren Scheibe erkennen
    - dabei nicht nur Grauwert-AbsDiff, sondern robustere Foreground-
      Segmentierung verwenden
    - virtuelles Greenscreen-Debugbild erzeugen:
        Hintergrund -> grün
        erkannte Änderung / Vordergrund -> Originalbild sichtbar

    Idee:
    - Referenz wird in Gray + Lab gespeichert
    - aktuelles Bild wird auf Referenz-Helligkeit normalisiert
    - Differenz erfolgt primär im Lab-Farbraum
    - zusätzlich wird eine Gradientendifferenz betrachtet
    - daraus entsteht eine binäre Vordergrundmaske
    - Konturen werden wie bisher geometrisch gefiltert
    - pro Kontur werden beide Enden als mögliche Spitzenkandidaten erzeugt

    Rückgabe von detect_mask_candidates(...):
    {
        "mask": fg_mask,          # finale binäre Vordergrundmaske
        "dbg": dbg,               # Debug-Bild (virtueller Greenscreen oder Original)
        "candidates": [...],      # gefilterte / gemergte Kandidaten
        "reject_stats": {...},    # Statistik, warum Konturen verworfen wurden
        "meta": {...}             # Meta-Infos zur Segmentierung
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

        - color_diff_threshold:
          Schwelle für Farbabweichung im Lab-Raum.

        - grad_diff_threshold:
          Schwelle für Änderung der Kanten-/Gradientenstärke.

        - min_foreground_ratio / max_foreground_ratio:
          Plausibilitätsgrenzen für die Vordergrundfläche.
        """
        self.board_mask = board_mask
        self.reference_frame = None
        self.reference_lab = None

        # Freeze-Heuristik:
        self.FREEZE_MEAN = float(freeze_mean)
        self.FREEZE_MAX = float(freeze_max)

        # Geometrische Filter für Konturen:
        self.min_area = 160
        self.max_area = 18000
        self.min_length = 12
        self.merge_dist = 20

        # Untere Grenze für alte Diff-Logik; bleibt als Kompatibilitätswert drin.
        self.min_diff_threshold = 18

        # Innere Board-Maske:
        kernel_mask = np.ones((9, 9), np.uint8)
        self.inner_board_mask = cv2.erode(self.board_mask, kernel_mask, iterations=1)

        # Weitere Form-Heuristiken:
        self.max_width = 52.0
        self.min_slenderness = 1.2
        self.min_aspect = 1.02

        # Neue Segmentierungs-Parameter:
        self.color_diff_threshold = 22.0
        self.grad_diff_threshold = 18.0

        # Plausibilität der Vordergrundfläche
        self.min_foreground_ratio = 0.00008
        self.max_foreground_ratio = 0.12

        # Virtueller Greenscreen im Debug-Bild
        self.use_virtual_greenscreen = True

        # Morphologie
        self.kernel_open = np.ones((3, 3), np.uint8)
        self.kernel_close = np.ones((5, 5), np.uint8)
        self.kernel_dilate = np.ones((3, 3), np.uint8)

    def _prepare_gray(self, frame_bgr):
        """
        Wandelt ein BGR-Bild in ein geglättetes Graubild um.

        Warum Blur?
        - reduziert Pixelrauschen
        - macht die Differenzbildung stabiler
        - kleine Sensor- oder Kompressionsartefakte stören weniger
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        return gray

    def _prepare_lab(self, frame_bgr):
        """
        Wandelt ein BGR-Bild in geglättetes Lab um.

        Warum Lab?
        - trennt Helligkeit und Farbinformation besser als BGR
        - Farbabweichungen des Darts gegenüber der Scheibe werden
          oft klarer als im reinen Graubild
        """
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        lab = cv2.GaussianBlur(lab, (5, 5), 0)
        return lab

    def set_reference(self, frame_bgr):
        """
        Setzt das Referenzbild, gegen das später verglichen wird.

        Das Referenzbild wird in mehreren vorbereiteten Formen gespeichert:
        - Gray für Gradientenvergleich
        - Lab für robustere Farb-/Foreground-Segmentierung
        """
        self.reference_frame = self._prepare_gray(frame_bgr)
        self.reference_lab = self._prepare_lab(frame_bgr)

    def detect(self, warped_frame_bgr, gray=None):
        """
        Kompatibilitätsmethode zu altem Code.

        Früher wurde offenbar nur (candidates, dbg) erwartet.
        Intern wird inzwischen detect_mask_candidates(...) verwendet,
        aber hier wird das alte Rückgabeformat nachgebildet.
        """
        result = self.detect_mask_candidates(warped_frame_bgr, gray=gray)
        return result["candidates"], result["dbg"]

    def _normalize_lighting_to_reference(self, frame_bgr):
        """
        Passt die Helligkeit des aktuellen Frames grob an die Referenz an.

        Idee:
        - Im Lab-Farbraum wird nur der L-Kanal global verschoben
        - Ziel ist, kleine Belichtungs- oder Autoexposure-Schwankungen
          zu reduzieren
        - Das ist bewusst einfach und robust gehalten

        Rückgabe:
        - norm_bgr: BGR-Bild mit angepasster Helligkeit
        """
        if self.reference_lab is None:
            return frame_bgr

        cur_lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        ref_lab = self.reference_lab.astype(np.float32)

        mask = self.inner_board_mask > 0
        if not np.any(mask):
            return frame_bgr

        cur_l = cur_lab[..., 0]
        ref_l = ref_lab[..., 0]

        cur_mean = float(np.mean(cur_l[mask]))
        ref_mean = float(np.mean(ref_l[mask]))

        shift = ref_mean - cur_mean
        cur_lab[..., 0] = np.clip(cur_lab[..., 0] + shift, 0, 255)

        norm_bgr = cv2.cvtColor(cur_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        return norm_bgr

    def _build_virtual_greenscreen_debug(self, frame_bgr, fg_mask):
        """
        Baut ein Debug-Bild im Stil eines virtuellen Greenscreens.

        Darstellung:
        - Hintergrund = grün
        - erkannter Vordergrund = Originalbild sichtbar

        Das ist nur für Debug/Visualisierung; die Erkennung selbst
        arbeitet auf der binären Vordergrundmaske.
        """
        dbg = np.zeros_like(frame_bgr)
        dbg[:] = (0, 255, 0)

        if fg_mask is None:
            return dbg

        fg_mask_3 = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        fg = cv2.bitwise_and(frame_bgr, fg_mask_3)
        bg = cv2.bitwise_and(dbg, cv2.bitwise_not(fg_mask_3))
        return cv2.add(fg, bg)

    def _build_threshold_mask(self, warped_frame_bgr, gray=None):
        """
        Kompatibilitäts-Wrapper zur alten Namensgebung.

        Intern wird jetzt eine Vordergrundmaske statt klassischer
        Graustufen-AbsDiff erzeugt.
        """
        return self._build_foreground_mask(warped_frame_bgr, gray=gray)

    def _build_foreground_mask(self, warped_frame_bgr, gray=None):
        """
        Erzeugt aus Referenzbild und aktuellem Bild eine binäre
        Vordergrund-/Änderungsmaske.

        Ablauf:
        1. Helligkeit des aktuellen Bilds grob an Referenz angleichen
        2. aktuelles Bild in Lab vorbereiten
        3. Farbabweichung im Lab-Raum berechnen
        4. Gradientendifferenz als Zusatzsignal berechnen
        5. beide Signale zu Vordergrundmaske kombinieren
        6. Plausibilität via foreground_ratio prüfen
        7. Morphologie anwenden
        8. finale Maske + Meta-Infos zurückgeben

        Rückgabe:
        - fg: finale binäre Maske
        - norm_bgr: normiertes BGR-Bild für Debug
        - meta: Diagnoseinfos oder Abbruchgrund
        """
        if self.reference_frame is None or self.reference_lab is None:
            return None, None, {"reason": "no_reference"}

        # Helligkeit an Referenz angleichen
        norm_bgr = self._normalize_lighting_to_reference(warped_frame_bgr)

        # Aktuelles Bild vorbereiten
        cur_lab = self._prepare_lab(norm_bgr)
        ref_lab = self.reference_lab

        # Differenz im Lab-Raum
        diff_lab = cur_lab.astype(np.float32) - ref_lab.astype(np.float32)

        dL = diff_lab[..., 0]
        da = diff_lab[..., 1]
        db = diff_lab[..., 2]

        # L etwas schwächer gewichten, Farbdifferenzen stärker
        color_dist = np.sqrt((0.6 * dL) ** 2 + da ** 2 + db ** 2)

        # Graubilder für Gradientendifferenz
        if gray is None:
            cur_gray = self._prepare_gray(norm_bgr)
        else:
            # Falls externes gray übergeben wurde, trotzdem leicht glätten
            cur_gray = cv2.GaussianBlur(gray, (5, 5), 0)

        ref_gray = self.reference_frame

        # Sobel-Gradienten berechnen
        cur_gx = cv2.Sobel(cur_gray, cv2.CV_32F, 1, 0, ksize=3)
        cur_gy = cv2.Sobel(cur_gray, cv2.CV_32F, 0, 1, ksize=3)
        ref_gx = cv2.Sobel(ref_gray, cv2.CV_32F, 1, 0, ksize=3)
        ref_gy = cv2.Sobel(ref_gray, cv2.CV_32F, 0, 1, ksize=3)

        cur_grad = cv2.magnitude(cur_gx, cur_gy)
        ref_grad = cv2.magnitude(ref_gx, ref_gy)
        grad_diff = cv2.absdiff(cur_grad, ref_grad)

        # Nur innerer Board-Bereich
        color_dist = np.where(self.inner_board_mask > 0, color_dist, 0)
        grad_diff = np.where(self.inner_board_mask > 0, grad_diff, 0)

        mask_pixels = color_dist[self.inner_board_mask > 0]
        if mask_pixels.size == 0:
            return None, None, {"reason": "empty_mask"}

        mean_val = float(np.mean(mask_pixels))
        max_val = float(np.max(mask_pixels))

        # Freeze-Heuristik:
        # global leicht anders, aber ohne starken Peak => eher Wackeln/Licht
        if mean_val > self.FREEZE_MEAN and max_val < self.FREEZE_MAX:
            return None, None, {
                "reason": "freeze",
                "mean_val": mean_val,
                "max_val": max_val,
            }

        # Binärmasken aus Farb- und Gradientensignal
        fg_color = (color_dist > self.color_diff_threshold).astype(np.uint8) * 255
        fg_grad = (grad_diff > self.grad_diff_threshold).astype(np.uint8) * 255

        # Kombination:
        # Farbe ist Hauptsignal, Gradient ist zusätzliche Hilfe
        fg = cv2.bitwise_or(fg_color, fg_grad.astype(np.uint8))
        fg = cv2.bitwise_and(fg, self.inner_board_mask)

        mask_nonzero = float(cv2.countNonZero(self.inner_board_mask))
        if mask_nonzero <= 0:
            return None, None, {"reason": "mask_zero"}

        foreground_ratio = cv2.countNonZero(fg) / mask_nonzero

        if foreground_ratio < self.min_foreground_ratio:
            return None, None, {
                "reason": "too_small",
                "white_ratio": foreground_ratio,
            }

        if foreground_ratio > self.max_foreground_ratio:
            return None, None, {
                "reason": "too_large",
                "white_ratio": foreground_ratio,
            }

        # Morphologische Nachbearbeitung
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.kernel_open, iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.kernel_close, iterations=1)
        fg = cv2.dilate(fg, self.kernel_dilate, iterations=1)

        # Sicherheitshalber erneut maskieren
        fg = cv2.bitwise_and(fg, self.inner_board_mask)

        meta = {
            "mean_val": mean_val,
            "max_val": max_val,
            "white_ratio": foreground_ratio,
            "threshold": None,  # aus Kompatibilitätsgründen vorhanden
            "color_diff_threshold": self.color_diff_threshold,
            "grad_diff_threshold": self.grad_diff_threshold,
        }

        return fg, norm_bgr, meta

    def detect_mask_candidates(self, warped_frame_bgr, gray=None):
        """
        Hauptmethode:
        - baut die Vordergrund-/Änderungsmaske
        - extrahiert Konturen
        - filtert sie geometrisch
        - erzeugt Kandidaten für mögliche Tip-Positionen
        - merged nahe Kandidaten
        - zeichnet die finalen Kandidaten ins Debug-Bild

        Rückgabe:
        {
            "mask": fg,
            "dbg": dbg,
            "candidates": merged,
            "reject_stats": reject_stats,
            "meta": meta,
        }
        """
        fg, norm_bgr, meta = self._build_foreground_mask(warped_frame_bgr, gray=gray)

        # Debug-Bild
        if self.use_virtual_greenscreen and fg is not None and norm_bgr is not None:
            dbg = self._build_virtual_greenscreen_debug(norm_bgr, fg)
        else:
            dbg = warped_frame_bgr.copy()

        reject_stats = {
            "area": 0,
            "points": 0,
            "length": 0,
            "width": 0,
            "slenderness": 0,
            "aspect": 0,
        }

        if fg is None:
            return {
                "mask": None,
                "dbg": dbg,
                "candidates": [],
                "reject_stats": reject_stats,
                "meta": meta,
            }

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw_candidates = []

        for cnt in contours:
            area = float(cv2.contourArea(cnt))

            if area < self.min_area or area > self.max_area:
                reject_stats["area"] += 1
                continue

            if len(cnt) < 5:
                reject_stats["points"] += 1
                continue

            pts = cnt.reshape(-1, 2).astype(np.float32)

            mean, eigenvecs = cv2.PCACompute(pts, mean=None)
            center = mean[0]
            axis = eigenvecs[0]

            proj = np.dot(pts - center, axis)

            i_min = int(np.argmin(proj))
            i_max = int(np.argmax(proj))

            p1 = pts[i_min]
            p2 = pts[i_max]

            length = float(np.linalg.norm(p2 - p1))
            if length < self.min_length:
                reject_stats["length"] += 1
                continue

            width = area / max(length, 1.0)
            width = max(width, 1.0)

            slenderness = length / width

            if width > self.max_width:
                reject_stats["width"] += 1
                continue

            if slenderness < self.min_slenderness:
                reject_stats["slenderness"] += 1
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = max(bw, bh) / max(1, min(bw, bh))
            if aspect < self.min_aspect:
                reject_stats["aspect"] += 1
                continue

            base_conf = length * slenderness * np.log(area + 1.0)

            idxs_min = np.argsort(proj)[:3]
            idxs_max = np.argsort(proj)[-3:]

            tip_a = np.mean(pts[idxs_min], axis=0)
            tip_b = np.mean(pts[idxs_max], axis=0)

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

        raw_candidates.sort(key=lambda o: o["confidence"], reverse=True)
        merged = []

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

        for o in merged[:20]:
            tx, ty = o["tip_board"]
            cv2.circle(dbg, (int(tx), int(ty)), 4, (0, 0, 255), -1)

        return {
            "mask": fg,
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
        "tip_board": (x, y),
        "per_cam_dist": [...],
        "used_cams": [...],
        "score": float,
        "max_dist": float
      }
    oder None
    """
    active = []

    for idx, m in enumerate(mask_list):
        if m is not None and cv2.countNonZero(m) > 0:
            active.append((idx, m))

    if len(active) < 2:
        return None

    dist_maps = []
    used_cams = []

    for idx, m in active:
        d = _build_distance_map_from_mask(m)
        if d is None:
            continue
        dist_maps.append(d)
        used_cams.append(idx)

    if len(dist_maps) < 2:
        return None

    union = np.zeros_like(board_mask, dtype=np.uint8)
    for _, m in active:
        union = cv2.bitwise_or(union, m)

    kernel = np.ones((9, 9), np.uint8)
    search_region = cv2.dilate(union, kernel, iterations=2)
    search_region = cv2.bitwise_and(search_region, board_mask)

    ys, xs = np.where(search_region > 0)
    if len(xs) == 0:
        return None

    stacked = np.stack(dist_maps, axis=0)
    vals = stacked[:, ys, xs]

    score = np.sum(vals, axis=0) + 0.75 * np.max(vals, axis=0)

    best_idx = int(np.argmin(score))

    bx = int(xs[best_idx])
    by = int(ys[best_idx])

    per_cam_dist = [float(stacked[i, by, bx]) for i in range(stacked.shape[0])]
    worst = max(per_cam_dist)

    if worst > max_dist:
        return None

    return {
        "tip_board": (float(bx), float(by)),
        "per_cam_dist": per_cam_dist,
        "used_cams": used_cams,
        "score": float(score[best_idx]),
        "max_dist": float(worst),
    }
