"""Auto-tune the detector by showing it real golf balls.

Instead of guessing thresholds with sliders, you box a few actual golf balls in
one frame and the tool measures their color/brightness and computes the settings
for you. It saves them to a JSON file the other scripts can load with --params.

Workflow:
    1. A live camera view opens (or a still image with --image).
    2. Press SPACE to freeze the frame you want to calibrate on.
    3. Drag a box around each golf ball. Press ENTER after each; press ENTER on
       an empty selection when you're done. (ESC cancels.)
    4. The tuned params are computed, previewed, and saved to --out.

Usage:
    python calibrate.py --camera 1                    # from a USB webcam
    python calibrate.py --camera csi                  # from the robot's Pi camera
    python calibrate.py --image green.jpg             # from a saved photo
    python calibrate.py --camera 1 --out params.json  # choose output file

Then run the detector with those settings:
    python run_webcam.py --camera 1 --params params.json
"""

from __future__ import annotations

import argparse
import sys

import cv2

from camera import camera_source, open_camera
from detector import (
    DetectorParams,
    derive_params_from_rois,
    detect_golf_balls,
    draw_detections,
)


def _grab_frame_from_camera(source: int | str, width: int, height: int):
    cap = open_camera(source, width, height)
    print("Live view: press SPACE to freeze a frame, or 'q' to abort.")
    frame = None
    try:
        while True:
            ok, f = cap.read()
            if not ok or f is None:
                continue
            preview = f.copy()
            cv2.putText(preview, "SPACE = freeze this frame,  q = quit",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("calibrate: pick a frame", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                frame = f.copy()
                break
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyWindow("calibrate: pick a frame")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--camera", type=camera_source, metavar="SOURCE",
                     help="USB webcam index (see list_cameras.py), or 'csi' for the Pi camera")
    src.add_argument("--image", help="calibrate from a saved image instead of the camera")
    parser.add_argument("--out", default="params.json", help="where to save tuned params")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Could not read image: {args.image}", file=sys.stderr)
            sys.exit(1)
    else:
        frame = _grab_frame_from_camera(args.camera, args.width, args.height)
        if frame is None:
            print("No frame captured; aborting.", file=sys.stderr)
            sys.exit(1)

    print("\nDraw a box around each golf ball. ENTER after each; "
          "ENTER on an empty box when done. ESC to cancel.")
    rois = cv2.selectROIs("calibrate: box the golf balls", frame, showCrosshair=False)
    cv2.destroyWindow("calibrate: box the golf balls")

    # selectROIs returns an (N,4) array of (x,y,w,h); filter out empty boxes.
    rois = [tuple(int(v) for v in r) for r in rois if r[2] > 0 and r[3] > 0]
    if not rois:
        print("No golf balls boxed; nothing to calibrate.", file=sys.stderr)
        sys.exit(1)
    print(f"Sampled {len(rois)} golf ball(s).")

    params = derive_params_from_rois(frame, rois)
    print("\nTuned parameters:")
    print(f"  sat_max={params.sat_max}  val_min={params.val_min}")
    print(f"  min_area={params.min_area}  max_area={params.max_area}")

    params.save(args.out)
    print(f"\nSaved to {args.out}")
    print(f"Run it with:  python run_webcam.py --camera <N|csi> --params {args.out}")

    # Preview the result on the calibration frame.
    dets = detect_golf_balls(frame, params)
    print(f"Detector found {len(dets)} ball(s) in the calibration frame.")
    annotated = draw_detections(frame, dets)
    cv2.imshow("calibrate: result (press any key to close)", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
