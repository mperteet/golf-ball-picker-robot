#!/usr/bin/env python3
"""Drive the VEX prototype to golf balls using the camera.

Each frame: read the camera -> detect balls (either backend) -> ChaseController
picks the nearest ball and decides wheel speeds -> TankDrive sends them to the
V5 Brain. The robot searches, turns to face a ball, drives up to it and stops;
once the ball is removed it looks for the next one.

Run from ball-detection-algo/ so relative paths (params.json, yolo11n.pt, the
Hailo model dir) resolve exactly as they do for run_webcam.py:

    cd ball-detection-algo
    python ../navigation/chase.py --camera 1 --dry-run                 # laptop, no Brain
    python ../navigation/chase.py --camera csi --params params.json    # robot, classical
    python ../navigation/chase.py --camera csi --backend yolo --model yolo11n_hailo_model --no-display
    python ../navigation/chase.py --camera csi --max-pct 20 --left 1,2 --right -3,-4

Every ChaseConfig field is a flag (--kp-turn, --arrive-frac, ...); see --help.
Press 'q' in the window or Ctrl-C to stop. Motors stop on any exit.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import fields

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "ball-detection-algo"))
sys.path.insert(0, _HERE)

import cv2  # noqa: E402

from backends import add_detector_args, detector_from_args  # noqa: E402
from camera import add_camera_args, open_camera  # noqa: E402
from controller import ChaseConfig, ChaseController, DriveCommand  # noqa: E402
from detector import draw_detections  # noqa: E402
from drive import add_drive_args, open_drive  # noqa: E402


def add_config_args(parser: argparse.ArgumentParser) -> None:
    """One flag per ChaseConfig field, so the CLI can never drift from the dataclass."""
    group = parser.add_argument_group("chase behaviour")
    defaults = ChaseConfig()
    for f in fields(ChaseConfig):
        group.add_argument("--" + f.name.replace("_", "-"), dest=f.name, type=float,
                           default=getattr(defaults, f.name),
                           help=f"(default: {getattr(defaults, f.name)})")


def config_from_args(args: argparse.Namespace) -> ChaseConfig:
    return ChaseConfig(**{f.name: getattr(args, f.name) for f in fields(ChaseConfig)})


def draw_overlay(frame, detections, cmd: DriveCommand, cfg: ChaseConfig):
    out = draw_detections(frame, detections)
    h, w = out.shape[:2]
    cv2.line(out, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)
    for frac, color in ((cfg.blind_frac, (0, 165, 255)), (cfg.arrive_frac, (0, 0, 255))):
        y = int(frac * h)
        cv2.line(out, (0, y), (w, y), color, 1)
    if cmd.target is not None:
        t = cmd.target
        cv2.rectangle(out, (t.x, t.y), (t.x + t.w, t.y + t.h), (0, 255, 255), 3)
        cv2.line(out, (w // 2, h), t.center, (0, 255, 255), 2)
    cv2.putText(out, f"{cmd.state.value}  L={cmd.left_pct:+.0f}%  R={cmd.right_pct:+.0f}%",
                (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_camera_args(parser)
    parser.add_argument("--no-display", dest="display", action="store_false",
                        help="don't open a window (headless robot)")
    parser.add_argument("--stale-s", type=float, default=0.5,
                        help="stop the motors if one camera+detect cycle takes longer "
                             "than this; the frame is too old to steer by (default: 0.5)")
    add_drive_args(parser)
    add_detector_args(parser)
    add_config_args(parser)
    args = parser.parse_args()

    cfg = config_from_args(args)
    detect = detector_from_args(args)
    cap = open_camera(args.camera, args.width, args.height)
    controller = ChaseController(cfg)

    print(f"Camera {args.camera} opened. " + ("Press 'q' to stop." if args.display else "Ctrl-C to stop."))
    last_state = None
    fps_t0, frames = time.monotonic(), 0
    try:
        with open_drive(args) as drive:
            while True:
                t0 = time.monotonic()
                ok, frame = cap.read()
                if not ok or frame is None:
                    drive.stop()
                    print("Dropped frame; motors stopped, retrying...")
                    continue
                detections = detect(frame)
                now = time.monotonic()
                if now - t0 > args.stale_s:
                    drive.stop()
                    print(f"Slow cycle ({now - t0:.2f} s > --stale-s {args.stale_s}); motors stopped")
                    continue

                h, w = frame.shape[:2]
                cmd = controller.update(detections, w, h, now)
                if cmd.state != last_state:
                    print(f"-> {cmd.state.value}")
                    last_state = cmd.state
                drive.set(cmd.left_pct, cmd.right_pct)
                frames += 1
                if frames % 30 == 0:
                    print(f"{frames / (now - fps_t0):.1f} FPS, {len(detections)} ball(s), "
                          f"{cmd.state.value} L={cmd.left_pct:+.0f}% R={cmd.right_pct:+.0f}%")
                    fps_t0, frames = now, 0

                if args.display:
                    cv2.imshow("chase (press q to stop)", draw_overlay(frame, detections, cmd, cfg))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print("Stopped.")


if __name__ == "__main__":
    main()
