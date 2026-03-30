import cv2
import numpy as np
from itertools import product


class AbsDiffDetector:
    """
    AbsDiff Detector im Boardspace (warped 600x600):
    Liefert pro Kamera mehrere plausible Tip-Kandidaten.

    detect_candidates(warped_bgr, gray_optional) ->
        candidates, dbg, reject_stats

    candidate:
        {
          "tip_board": (x, y),
          "confidence": float,
          "contour": cnt,
          "extra": {...}
        }
    """

    def __init__(self, board_mask, freeze_mean=20, freeze_max=70):
        self.board_mask = board_mask
        self.reference_frame = None  # gray

        self.FREEZE_MEAN = float(freeze_mean)
        self.FREEZE_MAX = float(freeze_max)

        # Toleranter als vorher, damit Cam 2 eher Kandidaten liefert
        self.min_area = 200
        self.max_area = 16000
        self.min_length = 18
        self.merge_dist = 22

        self.min_diff_threshold = 18

        kernel_mask = np.ones((9, 9), np.uint8)
        self.inner_board_mask = cv2.erode(self.board_mask, kernel_mask, iterations=1)

        # gelockerte Formfilter
        self.max_width = 40.0
        self.min_slenderness = 1.6
        self.min_aspect = 1.15

    def _prepare_gray(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        return gray

    def set_reference(self, frame_bgr):
        self.reference_frame = self._prepare_gray(frame_bgr)

    def detect(self, warped_frame_bgr, gray=None):
        """
        Kompatibilität für alten Code.
        """
        candidates, dbg, _reject_stats = self.detect_candidates(warped_frame_bgr, gray=gray)
        return candidates, dbg

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

        if white_ratio < 0.00015:
            return None, None, {
                "reason": "too_small",
                "white_ratio": white_ratio,
            }

        if white_ratio > 0.08:
            return None, None, {
                "reason": "too_large",
                "white_ratio": white_ratio,
            }

        kernel_open = np.ones((3, 3), np.uint8)
        kernel_close = np.ones((5, 5), np.uint8)

        thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel_open, iterations=1)
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        thr = cv2.bitwise_and(thr, self.inner_board_mask)

        meta = {
            "mean_val": mean_val,
            "max_val": max_val,
            "white_ratio": white_ratio,
            "threshold": final_thr,
        }
        return thr, diff_masked, meta

    def detect_candidates(self, warped_frame_bgr, gray=None):
        thr, _, meta = self._build_threshold_mask(warped_frame_bgr, gray=gray)
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
            return [], dbg, reject_stats

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

            dist_a = float(np.linalg.norm(tip_a - center))
            dist_b = float(np.linalg.norm(tip_b - center))

            conf_a = float(base_conf * (1.0 + 0.02 * dist_a))
            conf_b = float(base_conf * (1.0 + 0.02 * dist_b))

            extra_common = {
                "area": area,
                "length": length,
                "width": width,
                "slenderness": slenderness,
                "aspect": aspect,
                "threshold": meta["threshold"],
                "white_ratio": meta["white_ratio"],
                "center": (float(center[0]), float(center[1])),
            }

            raw_candidates.append({
                "tip_board": (float(tip_a[0]), float(tip_a[1])),
                "confidence": conf_a,
                "contour": cnt,
                "extra": {
                    **extra_common,
                    "endpoint_side": "min_proj",
                }
            })

            raw_candidates.append({
                "tip_board": (float(tip_b[0]), float(tip_b[1])),
                "confidence": conf_b,
                "contour": cnt,
                "extra": {
                    **extra_common,
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

        return merged, dbg, reject_stats


def _pairwise_cluster_score(points):
    d01 = float(np.linalg.norm(points[0] - points[1]))
    d02 = float(np.linalg.norm(points[0] - points[2]))
    d12 = float(np.linalg.norm(points[1] - points[2]))
    return d01 + d02 + d12 + max(d01, d02, d12), (d01, d02, d12)


def fuse_three_cameras(cands_cam0, cands_cam1, cands_cam2, max_pair_dist=35.0):
    if not cands_cam0 or not cands_cam1 or not cands_cam2:
        return None

    best = None
    best_score = float("inf")

    top0 = sorted(cands_cam0, key=lambda c: c["confidence"], reverse=True)[:8]
    top1 = sorted(cands_cam1, key=lambda c: c["confidence"], reverse=True)[:8]
    top2 = sorted(cands_cam2, key=lambda c: c["confidence"], reverse=True)[:8]

    for c0, c1, c2 in product(top0, top1, top2):
        p0 = np.array(c0["tip_board"], dtype=np.float32)
        p1 = np.array(c1["tip_board"], dtype=np.float32)
        p2 = np.array(c2["tip_board"], dtype=np.float32)

        score, dists = _pairwise_cluster_score([p0, p1, p2])
        worst_dist = max(dists)

        if worst_dist > max_pair_dist:
            continue

        conf_sum = c0["confidence"] + c1["confidence"] + c2["confidence"]
        final_score = score - 0.002 * conf_sum

        if final_score < best_score:
            best_score = final_score
            fused = (p0 + p1 + p2) / 3.0

            best = {
                "tip_board": (float(fused[0]), float(fused[1])),
                "per_camera": [c0, c1, c2],
                "cluster_score": float(score),
                "max_pair_dist": float(worst_dist),
                "confidence": float(conf_sum),
            }

    return best