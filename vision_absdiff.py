import cv2
import numpy as np


class AbsDiffDetector:
    """
    AbsDiff Detector im Boardspace (warped 600x600).

    Hauptziel:
    - stabile Änderungsmaske im Warped-Bild erzeugen
    - optionale lokale Kandidaten für Debug ableiten

    detect_mask_candidates(...) -> {
        "mask": thr,
        "dbg": dbg,
        "candidates": [...],
        "reject_stats": {...},
        "meta": {...}
    }
    """

    def __init__(self, board_mask, freeze_mean=20, freeze_max=70):
        self.board_mask = board_mask
        self.reference_frame = None

        self.FREEZE_MEAN = float(freeze_mean)
        self.FREEZE_MAX = float(freeze_max)

        # etwas toleranter
        self.min_area = 160
        self.max_area = 18000
        self.min_length = 12
        self.merge_dist = 20

        self.min_diff_threshold = 18

        kernel_mask = np.ones((9, 9), np.uint8)
        self.inner_board_mask = cv2.erode(self.board_mask, kernel_mask, iterations=1)

        self.max_width = 52.0
        self.min_slenderness = 1.2
        self.min_aspect = 1.02

    def _prepare_gray(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        return gray

    def set_reference(self, frame_bgr):
        self.reference_frame = self._prepare_gray(frame_bgr)

    def detect(self, warped_frame_bgr, gray=None):
        """
        Kompatibilität zu altem Code.
        """
        result = self.detect_mask_candidates(warped_frame_bgr, gray=gray)
        return result["candidates"], result["dbg"]

    def _build_threshold_mask(self, warped_frame_bgr, gray=None):
        if self.reference_frame is None:
            return None, None, {"reason": "no_reference"}

        if gray is None:
            gray = self._prepare_gray(warped_frame_bgr)
        else:
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

        diff = cv2.absdiff(self.reference_frame, gray)
        diff_masked = cv2.bitwise_and(diff, self.inner_board_mask)

        mask_pixels = diff_masked[self.inner_board_mask > 0]
        if mask_pixels.size == 0:
            return None, None, {"reason": "empty_mask"}

        mean_val = float(np.mean(mask_pixels))
        max_val = float(np.max(mask_pixels))

        # globales leichtes Wackeln ignorieren
        if mean_val > self.FREEZE_MEAN and max_val < self.FREEZE_MAX:
            return None, None, {
                "reason": "freeze",
                "mean_val": mean_val,
                "max_val": max_val,
            }

        diff_blur = cv2.GaussianBlur(diff_masked, (5, 5), 0)

        otsu_thr, _ = cv2.threshold(
            diff_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        final_thr = max(self.min_diff_threshold, int(otsu_thr))

        _, thr = cv2.threshold(diff_blur, final_thr, 255, cv2.THRESH_BINARY)

        mask_nonzero = float(cv2.countNonZero(self.inner_board_mask))
        if mask_nonzero <= 0:
            return None, None, {"reason": "mask_zero"}

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

        kernel_open = np.ones((3, 3), np.uint8)
        kernel_close = np.ones((5, 5), np.uint8)

        thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel_open, iterations=1)
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        # minimal erweitern, damit dünne Dart-Anteile nicht abbrechen
        kernel_dilate = np.ones((3, 3), np.uint8)
        thr = cv2.dilate(thr, kernel_dilate, iterations=1)

        thr = cv2.bitwise_and(thr, self.inner_board_mask)

        meta = {
            "mean_val": mean_val,
            "max_val": max_val,
            "white_ratio": white_ratio,
            "threshold": final_thr,
        }
        return thr, diff_masked, meta

    def detect_mask_candidates(self, warped_frame_bgr, gray=None):
        thr, _diff_masked, meta = self._build_threshold_mask(warped_frame_bgr, gray=gray)
        dbg = warped_frame_bgr.copy()

        reject_stats = {
            "area": 0,
            "points": 0,
            "length": 0,
            "width": 0,
            "slenderness": 0,
            "aspect": 0,
        }

        if thr is None:
            return {
                "mask": None,
                "dbg": dbg,
                "candidates": [],
                "reject_stats": reject_stats,
                "meta": meta,
            }

        contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
            "mask": thr,
            "dbg": dbg,
            "candidates": merged,
            "reject_stats": reject_stats,
            "meta": meta,
        }


def _build_distance_map_from_mask(mask):
    """
    Abstand jedes Pixels zum nächsten aktiven Maskenpixel.
    mask: 255 = Änderung, 0 = kein Treffer
    """
    if mask is None or cv2.countNonZero(mask) == 0:
        return None

    inv = np.where(mask > 0, 0, 255).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    return dist


def fuse_warped_masks(mask_list, board_mask, max_dist=22.0):
    """
    Sucht direkt im Warped-Bild den Punkt, der zu den Änderungsmasken
    aller verfügbaren Kameras den kleinsten Abstand hat.

    Das ist die finale Tip-Bestimmung.

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

    # Suchregion = Vereinigung der Änderungen, leicht erweitert
    union = np.zeros_like(board_mask, dtype=np.uint8)
    for _, m in active:
        union = cv2.bitwise_or(union, m)

    kernel = np.ones((9, 9), np.uint8)
    search_region = cv2.dilate(union, kernel, iterations=2)
    search_region = cv2.bitwise_and(search_region, board_mask)

    ys, xs = np.where(search_region > 0)
    if len(xs) == 0:
        return None

    stacked = np.stack(dist_maps, axis=0)  # [N, H, W]
    vals = stacked[:, ys, xs]              # [N, P]

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