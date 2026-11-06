"""Interactive tuner for the white-golf-ball detector.

Opens the live camera with trackbars for every threshold. Adjust them until only
the golf balls are boxed under YOUR lighting/green, then press 'p' (or 'q' to
quit). With --out the values are saved to a JSON file that the other scripts load
with --params; they are also printed, in case you want to change the
DetectorParams defaults in detector.py instead.

The window shows the annotated frame and the binary white-mask side by side so
you can see exactly what the color gate is selecting.

Usage:
    python tune.py --camera 1
    python tune.py --camera 1 --params params.json --out params.json
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import cv2
import numpy as np

from camera import add_camera_args, open_camera
from detector import DetectorParams, build_white_mask, detect_golf_balls, draw_detections

WINDOW = "tuner (p=print params, q=quit)"

# Trackbar spec: name -> (max value, DetectorParams attribute, scale).
# The stored param value = trackbar position / scale.
_TRACKBARS = [
    ("hue_min", 180, "hue_min", 1),
    ("hue_max", 180, "hue_max", 1),
    ("sat_max", 255, "sat_max", 1),
    ("val_min", 255, "val_min", 1),
    ("blur_ksize", 31, "blur_ksize", 1),
    ("morph_ksize", 31, "morph_ksize", 1),
    ("min_area", 5000, "min_area", 1),
    ("max_area/1000", 500, "max_area", 0.001),   # position 100 -> 100_000 px^2
    ("min_circ x100", 130, "min_circularity", 100),
    ("min_fill x100", 100, "min_fill_ratio", 100),
]


def _noop(_value: int) -> None:
    pass


def _build_window(params: DetectorParams) -> None:
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    for name, max_val, attr, scale in _TRACKBARS:
        init = int(round(getattr(params, attr) * scale))
        cv2.createTrackbar(name, WINDOW, min(init, max_val), max_val, _noop)


def _read_params(base: DetectorParams) -> DetectorParams:
    """Overlay the trackbar positions on `base`.

    Starting from `base` rather than the defaults keeps the params that have no
    trackbar (downscale, morph_iters, max_circularity) as loaded from --params,
    so saving with --out doesn't silently reset them.
    """
    params = replace(base)
    defaults = DetectorParams()
    for name, _max_val, attr, scale in _TRACKBARS:
        pos = cv2.getTrackbarPos(name, WINDOW)
        value = pos / scale
        # int-typed params stay int; float params (circularity/fill) become float
        if isinstance(getattr(defaults, attr), int):
            value = int(round(value))
        setattr(params, attr, value)
    return params


def _print_params(params: DetectorParams) -> None:
    print("\n# --- DetectorParams (save with --out, or paste into detector.py defaults) ---")
    print(f"hue_min={params.hue_min}, hue_max={params.hue_max},")
    print(f"sat_max={params.sat_max}, val_min={params.val_min},")
    print(f"blur_ksize={params.blur_ksize}, morph_ksize={params.morph_ksize},")
    print(f"min_area={params.min_area}, max_area={params.max_area},")
    print(f"min_circularity={params.min_circularity:.2f}, min_fill_ratio={params.min_fill_ratio:.2f}")
    print("# ------------------------------------------------------------------------\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_camera_args(parser)
    parser.add_argument("--params", metavar="PATH", default=None,
                        help="start from tuned params in this JSON file (see calibrate.py)")
    parser.add_argument("--out", metavar="PATH", default=None,
                        help="also save params to this JSON file on 'p'/quit")
    args = parser.parse_args()

    start = DetectorParams.load(args.params) if args.params else DetectorParams()
    cap = open_camera(args.camera, args.width, args.height)
    _build_window(start)
    print("Adjust trackbars. Press 'p' to print params, 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            params = _read_params(start)
            detections = detect_golf_balls(frame, params)
            annotated = draw_detections(frame, detections)

            # Show the mask next to the result (mask -> 3-channel to hstack).
            mask = build_white_mask(frame, params)
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            combo = np.hstack([annotated, mask_bgr])
            cv2.imshow(WINDOW, combo)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                _print_params(params)
                if args.out:
                    params.save(args.out)
                    print(f"Saved params to {args.out}")
                break
            if key == ord("p"):
                _print_params(params)
                if args.out:
                    params.save(args.out)
                    print(f"Saved params to {args.out}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
