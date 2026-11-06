"""Run a detection backend on a still image (no camera needed).

Great for offline testing and regression checks: save a photo (or a frame grab)
of golf balls on the green, run it through here, and confirm the boxes land on
the balls. Both backends are exercised through the same interface the live loop
uses, so what you see here is what the robot sees.

Usage:
    python test_image.py balls.jpg                        # classical CV (default)
    python test_image.py balls.jpg --params params.json   # classical, tuned
    python test_image.py balls.jpg --backend yolo         # zero-shot YOLO11n
    python test_image.py balls.jpg -o out.jpg             # also save the result
    python test_image.py balls.jpg --no-show -o out.jpg   # headless: save only
"""

from __future__ import annotations

import argparse
import sys

import cv2

from backends import add_detector_args, detector_from_args
from detector import draw_detections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="path to an image file")
    parser.add_argument("-o", "--out", default=None, help="save the annotated image here")
    parser.add_argument("--no-show", dest="show", action="store_false",
                        help="don't open a window (headless)")
    add_detector_args(parser)
    args = parser.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"Could not read image: {args.image}", file=sys.stderr)
        sys.exit(1)

    detect = detector_from_args(args)
    detections = detect(frame)
    annotated = draw_detections(frame, detections)

    print(f"Detected {len(detections)} ball(s):")
    for i, d in enumerate(detections, 1):
        # Only one of the two scores is meaningful, depending on the backend.
        score = (f"circularity={d.circularity:.2f}" if d.circularity is not None
                 else f"confidence={d.confidence:.2f}")
        print(f"  {i}: {d.label} box=({d.x},{d.y},{d.w},{d.h}) "
              f"center={d.center} {score}")

    if args.out:
        cv2.imwrite(args.out, annotated)
        print(f"Saved annotated image to {args.out}")

    if args.show:
        cv2.imshow("test_image (press any key to close)", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
