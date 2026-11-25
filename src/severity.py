"""
severity.py  --  PROMETHEUS-X severity estimation

Takes the anomaly heatmap produced by the detector (PatchCore) and a decision
threshold, and answers: "how serious is this defect?"

Severity combines two things:
    1. AREA      - how much of the image is defective (defect area %)
    2. STRENGTH  - peak and mean anomaly relative to the pixel threshold
    3. GEOMETRY  - connected-region count and their spatial extent

Output level: None / Low / Medium / High

Dependencies: numpy (required), scipy (optional, used for region counting
and bounding boxes; the module still works without it).
"""

from dataclasses import dataclass, field, asdict
from typing import List, Tuple

import numpy as np

try:
    from scipy import ndimage
    _HAS_SCIPY = True
except ImportError:  # graceful fallback
    _HAS_SCIPY = False


# --------------------------------------------------------------------------- #
# Tunable settings (explain these to the judges: they are the "severity policy")
# --------------------------------------------------------------------------- #
AREA_CAP_PCT = 5.0       # defect covering >= 5% of image => area component = 1.0
RATIO_CAP = 2.0          # peak score >= 2x threshold    => strength component = 1.0
W_AREA = 0.35            # weight of affected area
W_STRENGTH = 0.30        # weight of peak anomaly strength
W_MEAN_STRENGTH = 0.15   # weight of mean anomaly strength in regions
W_REGION_COUNT = 0.10    # weight of connected-region count
W_SPATIAL_EXTENT = 0.10  # weight of the total defect bounding extent
REGION_COUNT_CAP = 5     # region component saturates at this many blobs
SPATIAL_EXTENT_CAP_PCT = 50.0
LOW_MAX = 0.33           # index < 0.33  => Low
MEDIUM_MAX = 0.66        # index < 0.66  => Medium, otherwise High
MIN_REGION_PIXELS = 1    # ignore connected regions smaller than this


@dataclass
class SeverityResult:
    level: str                       # "None", "Low", "Medium", "High"
    index: float                     # 0..1 combined severity score
    area_pct: float                  # % of image pixels above threshold
    peak_score: float                # max anomaly value in the map
    mean_defect_score: float         # mean anomaly value inside defect pixels
    strength_ratio: float            # peak_score / threshold
    num_regions: int                 # number of separate defect blobs
    bboxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    # each bbox = (x_min, y_min, x_max, y_max)
    region_sizes_pct: List[float] = field(default_factory=list)
    region_centroids: List[Tuple[float, float]] = field(default_factory=list)
    spatial_extent_pct: float = 0.0
    mask: np.ndarray = None          # boolean defect mask, same shape as map

    def to_dict(self):
        """JSON-friendly dict (mask excluded) for the UI / report."""
        d = asdict(self)
        d.pop("mask", None)
        return d


def _validate_map(anomaly_map) -> np.ndarray:
    arr = np.asarray(anomaly_map, dtype=np.float64)
    # accept (H,W), (1,H,W), (H,W,1)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(
            f"anomaly_map must be 2-D after squeeze, got shape {arr.shape}"
        )
    if arr.size == 0:
        raise ValueError("anomaly_map is empty")
    if not np.all(np.isfinite(arr)):
        arr = np.nan_to_num(arr, nan=0.0, posinf=np.nanmax(arr[np.isfinite(arr)])
                            if np.any(np.isfinite(arr)) else 0.0, neginf=0.0)
    return arr


def _find_regions(mask: np.ndarray):
    """Return count, boxes, pixel areas, centroids, and cleaned mask."""
    if not mask.any():
        return 0, [], [], [], mask

    if _HAS_SCIPY:
        labeled, _ = ndimage.label(mask)
        bboxes, region_sizes, centroids = [], [], []
        keep = np.zeros_like(mask)
        slices = ndimage.find_objects(labeled)
        for i, sl in enumerate(slices, start=1):
            region = labeled[sl] == i
            if region.sum() < MIN_REGION_PIXELS:
                continue
            keep[sl] |= region
            y0, y1 = sl[0].start, sl[0].stop - 1
            x0, x1 = sl[1].start, sl[1].stop - 1
            bboxes.append((int(x0), int(y0), int(x1), int(y1)))
            local_y, local_x = np.where(region)
            region_sizes.append(int(region.sum()))
            centroids.append((
                round(float(x0 + local_x.mean()), 2),
                round(float(y0 + local_y.mean()), 2),
            ))
        return len(bboxes), bboxes, region_sizes, centroids, keep

    # fallback: one box around all defect pixels
    ys, xs = np.where(mask)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    centroid = (round(float(xs.mean()), 2), round(float(ys.mean()), 2))
    return 1, [bbox], [int(mask.sum())], [centroid], mask


def estimate_severity(anomaly_map, threshold: float) -> SeverityResult:
    """
    Estimate defect severity from an anomaly heatmap.

    Parameters
    ----------
    anomaly_map : array-like, shape (H, W)
        Pixel-level anomaly scores (higher = more abnormal).
    threshold : float
        Pixel score above which a pixel is considered defective.
        Must be > 0 (use the value chosen on normal validation images).

    Returns
    -------
    SeverityResult
    """
    if threshold is None or not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be a positive finite number")

    amap = _validate_map(anomaly_map)
    peak = float(amap.max())

    raw_mask = amap > threshold
    num_regions, bboxes, region_sizes, centroids, mask = _find_regions(raw_mask)

    # nothing above threshold -> no defect
    if num_regions == 0:
        return SeverityResult(
            level="None", index=0.0, area_pct=0.0, peak_score=peak,
            mean_defect_score=0.0, strength_ratio=peak / threshold,
            num_regions=0,
            bboxes=[],
            region_sizes_pct=[],
            region_centroids=[],
            spatial_extent_pct=0.0,
            mask=np.zeros_like(raw_mask),
        )

    area_pct = 100.0 * float(mask.sum()) / mask.size
    mean_defect = float(amap[mask].mean())
    ratio = peak / threshold
    mean_ratio = mean_defect / threshold

    ys, xs = np.where(mask)
    extent_pixels = (int(xs.max()) - int(xs.min()) + 1) * (int(ys.max()) - int(ys.min()) + 1)
    spatial_extent_pct = 100.0 * extent_pixels / mask.size

    # components, each clipped to [0, 1]
    area_component = min(area_pct / AREA_CAP_PCT, 1.0)
    strength_component = float(np.clip((ratio - 1.0) / (RATIO_CAP - 1.0), 0.0, 1.0))
    mean_component = float(np.clip((mean_ratio - 1.0) / (RATIO_CAP - 1.0), 0.0, 1.0))
    region_component = min(num_regions / REGION_COUNT_CAP, 1.0)
    extent_component = min(spatial_extent_pct / SPATIAL_EXTENT_CAP_PCT, 1.0)

    index = (
        W_AREA * area_component
        + W_STRENGTH * strength_component
        + W_MEAN_STRENGTH * mean_component
        + W_REGION_COUNT * region_component
        + W_SPATIAL_EXTENT * extent_component
    )
    index = float(np.clip(index, 0.0, 1.0))

    if index < LOW_MAX:
        level = "Low"
    elif index < MEDIUM_MAX:
        level = "Medium"
    else:
        level = "High"

    return SeverityResult(
        level=level,
        index=round(index, 4),
        area_pct=round(area_pct, 4),
        peak_score=round(peak, 6),
        mean_defect_score=round(mean_defect, 6),
        strength_ratio=round(ratio, 4),
        num_regions=num_regions,
        bboxes=bboxes,
        region_sizes_pct=[round(100.0 * size / mask.size, 4) for size in region_sizes],
        region_centroids=centroids,
        spatial_extent_pct=round(spatial_extent_pct, 4),
        mask=mask,
    )


# --------------------------------------------------------------------------- #
# Self-test:  python src/severity.py
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    H = W = 256
    thr = 0.5

    def blob(amap, cy, cx, r, val):
        yy, xx = np.ogrid[:H, :W]
        amap[(yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2] = val
        return amap

    rng = np.random.default_rng(0)
    base = rng.uniform(0.0, 0.3, (H, W))          # normal background

    tests = {
        "clean image":        base.copy(),
        "tiny weak scratch":  blob(base.copy(), 100, 100, 3, 0.55),
        "medium defect":      blob(base.copy(), 128, 128, 12, 0.8),
        "large strong crack": blob(base.copy(), 128, 128, 40, 1.4),
        "two separate spots": blob(blob(base.copy(), 50, 50, 8, 0.9), 200, 200, 8, 0.9),
    }

    for name, amap in tests.items():
        r = estimate_severity(amap, thr)
        print(f"{name:22s} -> {r.level:6s} idx={r.index:.2f} "
              f"area={r.area_pct:5.2f}% peak={r.peak_score:.2f} "
              f"regions={r.num_regions}")

    print("\nJSON for UI:", estimate_severity(tests['medium defect'], thr).to_dict())