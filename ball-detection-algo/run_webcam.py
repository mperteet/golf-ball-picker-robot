"""Live golf-ball detector on the camera feed.

Opens the camera you point it at, runs the selected detection backend on each
frame, and draws a green box around every golf ball. This is the entry point that
will run on the Raspberry Pi 5 (add --no-display when the robot is headless).

Usage:
    python run_webcam.py --camera 1                        # classical CV (default)
    python run_webcam.py --camera 1 --params params.json   # classical, tuned
    python run_webcam.py --camera 1 --backend yolo         # zero-shot YOLO11n
    python run_webcam.py --camera csi --no-display         # headless robot, Pi camera
    python run_webcam.py --camera 1 --save last.jpg        # save latest annotated frame
Press 'q' in the window to quit.

The backend is chosen at startup and then called identically every frame, so
detection stays a swappable node for the autonomous state machine.
"""

from __future__ import annotations

import argparse
import time

import cv2

from backends import add_detector_args, detector_from_args
from camera import add_camera_args, open_camera
from detector import draw_detections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_camera_args(parser)
    parser.add_argument("--no-display", dest="display", action="store_false",
                        help="don't open a window (headless robot)")
    parser.add_argument("--save", metavar="PATH", default=None,
                        help="write the latest annotated frame to this path each loop")
    add_detector_args(parser)
    args = parser.parse_args()

    detect = detector_from_args(args)
    cap = open_camera(args.camera, args.width, args.height)

    print(f"Camera {args.camera} opened. Press 'q' to quit." if args.display
          else f"Camera {args.camera} opened (headless). Ctrl-C to quit.")

    fps_t0 = time.time()
    frames = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Dropped frame; retrying...")
                continue

            detections = detect(frame)
            annotated = draw_detections(frame, detections)

            frames += 1
            if frames % 30 == 0:
                dt = time.time() - fps_t0
                print(f"{frames / dt:.1f} FPS, {len(detections)} ball(s) this frame")
                fps_t0, frames = time.time(), 0

            if args.save:
                cv2.imwrite(args.save, annotated)

            if args.display:
                cv2.imshow("golf ball detector (press q to quit)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
