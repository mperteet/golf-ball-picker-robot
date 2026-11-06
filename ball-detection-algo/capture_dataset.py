"""Capture training images from the camera.

Milestone 2 needs a few hundred photos of real golf balls before a custom model
can be trained. This grabs frames to a folder with sequential filenames, either
on a keypress or automatically at a fixed interval while you walk the camera
around the green.

WHAT TO CAPTURE
    Shoot the conditions that currently FAIL, because that is what the model has
    to learn and what the classical thresholds cannot handle:
      * balls in shade, and half-shadowed balls
      * overcast light and low-angle evening sun
      * balls at range, near the far edge of the green
      * balls against fringe, collar, and non-uniform grass
    Include some frames with NO balls at all: negatives teach the model what
    isn't a ball, which is what keeps false positives down.

    Capture at the camera height the robot will actually use, and with the
    robot's own camera (--camera csi) wherever possible. The Camera Module 3
    Wide's 120-degree lens bends and shrinks balls differently from a laptop
    webcam, and a dataset shot from another lens or from standing height teaches
    the wrong view.

MUST BE RUN IN YOUR OWN TERMINAL -- it opens an interactive OpenCV window, so
prefix the command with '!' rather than letting an agent background it.

Usage:
    python capture_dataset.py --camera 1 --out datasets/raw
    python capture_dataset.py --camera 1 --out datasets/raw --interval 2.0
    python capture_dataset.py --camera csi --out datasets/raw   # on the robot (preferred)
"""

from __future__ import annotations

import argparse
import os
import time

import cv2

from camera import add_camera_args, open_camera


def next_index(out_dir: str, prefix: str) -> int:
    """Resume numbering so a second session doesn't overwrite the first."""
    existing = [f for f in os.listdir(out_dir)
                if f.startswith(prefix) and f.endswith(".jpg")]
    highest = 0
    for name in existing:
        stem = name[len(prefix):-len(".jpg")].lstrip("_")
        if stem.isdigit():
            highest = max(highest, int(stem))
    return highest + 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_camera_args(parser)
    parser.add_argument("--out", default="datasets/raw",
                        help="output folder (default: datasets/raw)")
    parser.add_argument("--prefix", default="green",
                        help="filename prefix; use a different one per session "
                             "or location (default: green)")
    parser.add_argument("--interval", type=float, default=0.0, metavar="SEC",
                        help="auto-capture every SEC seconds (0 = manual only)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    index = next_index(args.out, args.prefix)

    cap = open_camera(args.camera, args.width, args.height)
    print(f"Saving to {args.out}/ starting at #{index}")
    print("SPACE or 'c' = capture     q = quit")
    if args.interval > 0:
        print(f"Auto-capturing every {args.interval}s")

    saved = 0
    last_auto = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Dropped frame; retrying...")
                continue

            now = time.time()
            take = args.interval > 0 and (now - last_auto) >= args.interval

            preview = frame.copy()
            cv2.putText(preview, f"saved: {saved}   next: #{index}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("capture (SPACE=save, q=quit)", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in (ord(" "), ord("c")):
                take = True

            if take:
                # Write the CLEAN frame, never the annotated preview.
                path = os.path.join(args.out, f"{args.prefix}_{index:04d}.jpg")
                cv2.imwrite(path, frame)
                print(f"saved {path}")
                index += 1
                saved += 1
                last_auto = now
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\nCaptured {saved} image(s) into {args.out}/")
    print("Next: label them (Roboflow or CVAT) in YOLO format, single class 'ball'.")


if __name__ == "__main__":
    main()
