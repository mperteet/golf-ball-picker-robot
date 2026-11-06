"""Core white-golf-ball detection.

This module is intentionally camera-free and I/O-free: it operates on a single
BGR frame (a NumPy array) so it can be unit-tested on still images and dropped
into the robot's autonomous state machine as an isolated "scan" node. The live
camera loop, the tuner, and the still-image tester all call into here.

Detection is classical CV (no ML): white golf balls are bright, low-saturation,
round blobs. We threshold for white in HSV, clean the mask morphologically, then
keep only contours that are the right size AND actually round. This is cheap
enough to run on the Raspberry Pi 5 CPU with no accelerator.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields

import cv2
import numpy as np


@dataclass
class DetectorParams:
    """All tunable knobs for the detector, in one place.

    `calibrate.py` and `tune.py` save these to a JSON file that the other scripts
    load with --params; edit the defaults here only to change the baseline. Keeping every threshold here means the live loop, the
    tuner, and the still-image tester never drift apart.
    """

    # --- "White" color gate (HSV). Hue spans full range; white = low S, high V.
    hue_min: int = 0
    hue_max: int = 180
    sat_max: int = 60          # reject anything too colorful to be a white ball
    val_min: int = 180         # reject anything too dark to be a white ball

    # --- Pre-processing
    blur_ksize: int = 5        # Gaussian blur kernel (odd). 0/1 disables blur.
    downscale: float = 1.0     # <1.0 shrinks the frame before detection (faster)

    # --- Morphology (elliptical kernel) to de-speckle and fill the ball
    morph_ksize: int = 5
    morph_iters: int = 1

    # --- Shape gates. Areas are ORIGINAL-frame px^2 whatever `downscale` is,
    # so calibrated values stay valid and downscale only trades accuracy for speed.
    min_area: int = 120        # px^2; drops tiny bright specks
    max_area: int = 100_000    # px^2; drops huge bright regions (walls, sky)
    min_circularity: float = 0.60   # 4*pi*area / perimeter^2 ; 1.0 == perfect circle
    max_circularity: float = 1.30   # allow a little slack above 1 for pixelation
    min_fill_ratio: float = 0.65    # contour area / minEnclosingCircle area

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DetectorParams":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path: str) -> None:
        """Write params to a JSON file (used by calibrate.py / tune.py)."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "DetectorParams":
        """Load params from a JSON file written by save()."""
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


@dataclass
class Detection:
    """One detected ball, in ORIGINAL-frame pixel coordinates.

    Shared by every detection backend (see backends.py) so that navigation and
    the autonomous state machine never learn which detector produced a box.

    The two score fields are backend-specific and only one is ever meaningful:
      * `circularity` is a classical-CV shape measurement; YOLO leaves it None.
      * `confidence` is a model score; the classical path leaves it 1.0, since
        it makes a hard accept/reject decision on each contour.
    """

    x: int
    y: int
    w: int
    h: int
    circularity: float | None = None
    confidence: float = 1.0
    label: str = "ball"

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)


def build_white_mask(frame_bgr: np.ndarray, params: DetectorParams) -> np.ndarray:
    """Return a binary mask (uint8, 0/255) of white-ish regions in the frame.

    Exposed separately so the tuner can display the mask alongside the result.
    """
    work = frame_bgr
    if params.blur_ksize and params.blur_ksize > 1:
        k = params.blur_ksize | 1  # force odd
        work = cv2.GaussianBlur(work, (k, k), 0)

    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    lower = np.array([params.hue_min, 0, params.val_min], dtype=np.uint8)
    upper = np.array([params.hue_max, params.sat_max, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    if params.morph_ksize and params.morph_ksize > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (params.morph_ksize, params.morph_ksize)
        )
        # open removes speckle, close fills holes inside the ball
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=params.morph_iters)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=params.morph_iters)

    return mask


def detect_golf_balls(frame_bgr: np.ndarray, params: DetectorParams | None = None) -> list[Detection]:
    """Detect white golf balls in a BGR frame.

    Pure function: no camera, no window, no disk. Returns bounding boxes in the
    coordinate system of the input frame (scaling is undone internally when
    `params.downscale < 1.0`).
    """
    if params is None:
        params = DetectorParams()

    scale = params.downscale if 0.0 < params.downscale < 1.0 else 1.0
    if scale != 1.0:
        small = cv2.resize(frame_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = frame_bgr

    mask = build_white_mask(small, params)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections: list[Detection] = []
    inv_scale = 1.0 / scale
    for contour in contours:
        area = cv2.contourArea(contour)
        # Gate in original-frame units: calibrate.py measures balls at full
        # resolution, so comparing the shrunk contour would reject real balls.
        if not params.min_area <= area * inv_scale * inv_scale <= params.max_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        if circularity < params.min_circularity or circularity > params.max_circularity:
            continue

        # Reject bright non-round blobs: a real ball fills most of its enclosing circle.
        (_cx, _cy), radius = cv2.minEnclosingCircle(contour)
        circle_area = math.pi * radius * radius
        if circle_area <= 0 or (area / circle_area) < params.min_fill_ratio:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        detections.append(
            Detection(
                x=int(round(x * inv_scale)),
                y=int(round(y * inv_scale)),
                w=int(round(w * inv_scale)),
                h=int(round(h * inv_scale)),
                circularity=float(circularity),
            )
        )

    return detections


def derive_params_from_rois(
    frame_bgr: np.ndarray,
    rois: list[tuple[int, int, int, int]],
    base: DetectorParams | None = None,
) -> DetectorParams:
    """Auto-tune thresholds from example golf balls the user boxed.

    `rois` is a list of (x, y, w, h) rectangles drawn over real golf balls in
    `frame_bgr`. We sample the pixels inside an inscribed circle of each box (to
    avoid the background in the box corners), measure their saturation/value
    distribution, and set the color gate to comfortably include them:

      * val_min = a low percentile of the balls' brightness (minus a margin)
      * sat_max = a high percentile of the balls' saturation (plus a margin)

    Area limits are set from the sizes of the boxed balls, in original-frame
    pixels (the unit detect_golf_balls gates on). Hue stays full-range
    because white is achromatic (its hue is meaningless/noisy).

    Pure function: no camera, no window. Returns a new DetectorParams.
    """
    if base is None:
        base = DetectorParams()
    if not rois:
        raise ValueError("need at least one ROI (box a golf ball) to calibrate")

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    fh, fw = hsv.shape[:2]

    sat_samples: list[np.ndarray] = []
    val_samples: list[np.ndarray] = []
    ball_areas: list[float] = []

    for (x, y, w, h) in rois:
        # clamp the box to the frame
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(fw, x + w), min(fh, y + h)
        if x1 <= x0 or y1 <= y0:
            continue
        patch = hsv[y0:y1, x0:x1]
        ph, pw = patch.shape[:2]

        # inscribed circle mask -> mostly ball pixels, not the corner background
        mask = np.zeros((ph, pw), dtype=np.uint8)
        radius = int(0.9 * min(pw, ph) / 2)
        cv2.circle(mask, (pw // 2, ph // 2), max(1, radius), 255, -1)
        sel = mask > 0

        sat_samples.append(patch[..., 1][sel])
        val_samples.append(patch[..., 2][sel])
        ball_areas.append(math.pi * (min(pw, ph) / 2.0) ** 2)

    if not sat_samples:
        raise ValueError("ROIs did not overlap the frame; nothing to sample")

    sat = np.concatenate(sat_samples)
    val = np.concatenate(val_samples)

    # Percentiles with margins so normal lighting variation still passes.
    sat_max = int(min(255, np.percentile(sat, 95) + 15))
    val_min = int(max(0, np.percentile(val, 5) - 20))

    min_area = int(max(20, 0.35 * min(ball_areas)))
    max_area = int(3.0 * max(ball_areas))

    return DetectorParams(
        hue_min=base.hue_min,
        hue_max=base.hue_max,
        sat_max=sat_max,
        val_min=val_min,
        blur_ksize=base.blur_ksize,
        downscale=base.downscale,
        morph_ksize=base.morph_ksize,
        morph_iters=base.morph_iters,
        min_area=min_area,
        max_area=max_area,
        min_circularity=base.min_circularity,
        max_circularity=base.max_circularity,
        min_fill_ratio=base.min_fill_ratio,
    )


def draw_detections(frame_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw bounding boxes + labels onto a COPY of the frame and return it."""
    out = frame_bgr.copy()
    for det in detections:
        cv2.rectangle(out, (det.x, det.y), (det.x + det.w, det.y + det.h), (0, 255, 0), 2)
        # Show whichever score the producing backend actually filled in.
        score = det.circularity if det.circularity is not None else det.confidence
        label = f"{det.label} {score:.2f}"
        y_text = det.y - 8 if det.y - 8 > 10 else det.y + det.h + 18
        cv2.putText(
            out, label, (det.x, y_text),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
        )
    cv2.putText(
        out, f"balls: {len(detections)}", (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
    )
    return out
