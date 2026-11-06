"""Compare detection backends over a folder of images.

The point of the YOLO work is better RECALL: the classical detector misses balls
in shade and at distance. This script is the evidence for whether that actually
improved. It runs both backends over the same images, reports how many balls each
found, and writes side-by-side annotated pairs so you can confirm by eye that the
extra boxes are real balls and not new false positives.

This is deliberately NOT mAP: that needs labelled ground truth. It is a
count-and-look comparison, which is the honest measurement available before the
dataset exists, and it is what tells you whether labelling is worth doing.

Pass your best --params calibration: the comparison is only fair against a
properly tuned classical baseline.

Usage:
    python benchmark.py photos/
    python benchmark.py photos/ --params params.json --out comparison/
    python benchmark.py photos/ --imgsz 960          # help small/distant balls
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2

from backends import add_detector_args, make_detector
from detector import DetectorParams, draw_detections

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def find_images(path: str) -> list[str]:
    """Accept either a directory of images or a single image file."""
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        return []
    return sorted(
        os.path.join(path, name)
        for name in os.listdir(path)
        if name.lower().endswith(IMAGE_EXTS)
    )


def _banner(image, text: str):
    """Label a panel so the two halves of a comparison can't be confused."""
    out = image.copy()
    height = out.shape[0]
    cv2.rectangle(out, (0, height - 34), (out.shape[1], height), (0, 0, 0), -1)
    cv2.putText(out, text, (10, height - 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", help="directory of images, or a single image")
    add_detector_args(parser, select_backend=False)
    parser.add_argument("--out", metavar="DIR", default=None,
                        help="write side-by-side annotated comparisons here")
    args = parser.parse_args()

    paths = find_images(args.images)
    if not paths:
        print(f"No images found in {args.images!r} "
              f"(looked for {', '.join(IMAGE_EXTS)})", file=sys.stderr)
        sys.exit(1)

    params = DetectorParams.load(args.params) if args.params else DetectorParams()
    if args.params:
        print(f"Classical params from {args.params}")

    classical = make_detector("classical", params=params)
    yolo = make_detector("yolo", model=args.model, imgsz=args.imgsz,
                         conf=args.conf, iou=args.iou, coco_class=args.coco_class)

    if args.out:
        os.makedirs(args.out, exist_ok=True)

    print(f"\n{len(paths)} image(s)\n")
    header = f"{'image':<34} {'classical':>9} {'yolo':>6} {'delta':>6}"
    print(header)
    print("-" * len(header))

    totals = {"classical": 0, "yolo": 0}
    times = {"classical": 0.0, "yolo": 0.0}
    better = worse = 0

    for path in paths:
        frame = cv2.imread(path)
        if frame is None:
            print(f"{os.path.basename(path):<34}  (unreadable, skipped)")
            continue

        t0 = time.perf_counter()
        classical_dets = classical(frame)
        times["classical"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        yolo_dets = yolo(frame)
        times["yolo"] += time.perf_counter() - t0

        n_classical, n_yolo = len(classical_dets), len(yolo_dets)
        totals["classical"] += n_classical
        totals["yolo"] += n_yolo
        delta = n_yolo - n_classical
        if delta > 0:
            better += 1
        elif delta < 0:
            worse += 1

        name = os.path.basename(path)
        print(f"{name[:33]:<34} {n_classical:>9} {n_yolo:>6} {delta:>+6}")

        if args.out:
            pair = cv2.hconcat([
                _banner(draw_detections(frame, classical_dets), f"classical: {n_classical}"),
                _banner(draw_detections(frame, yolo_dets), f"yolo: {n_yolo}"),
            ])
            cv2.imwrite(os.path.join(args.out, f"cmp_{name}"), pair)

    count = len(paths)
    print("-" * len(header))
    print(f"{'TOTAL balls found':<34} {totals['classical']:>9} {totals['yolo']:>6} "
          f"{totals['yolo'] - totals['classical']:>+6}")
    print(f"\nmean latency   classical {times['classical'] / count * 1000:7.1f} ms"
          f"   yolo {times['yolo'] / count * 1000:7.1f} ms")
    print(f"images where yolo found more: {better}    fewer: {worse}")
    if args.out:
        print(f"\nSide-by-side comparisons written to {args.out}/")
    print("\nCounts alone don't prove correctness -- open the comparisons and "
          "confirm the extra boxes are real balls.")


if __name__ == "__main__":
    main()
