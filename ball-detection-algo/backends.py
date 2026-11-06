"""Detector backend selection.

The single place that knows every detection implementation. Entry-point scripts
(and eventually the robot's autonomous state machine) depend on this factory
instead of importing a detector directly, so swapping classical CV for YOLO is a
flag rather than a code change.

Every backend satisfies one contract:

    detector(frame_bgr) -> list[Detection]

This is the ONLY module that imports both detectors. That is deliberate: it lets
detector.py stay free of any ML dependency, so the classical path remains a real
fallback on a machine with nothing but opencv and numpy installed.
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable

import numpy as np

from detector import Detection, DetectorParams, detect_golf_balls

BACKENDS = ("classical", "yolo")

#: Every backend is just something callable with this shape.
BallDetector = Callable[[np.ndarray], "list[Detection]"]


class ClassicalBackend:
    """Adapter giving the pure classical function the callable-object shape.

    Holds the tuned DetectorParams so callers don't have to thread them through
    every frame, which is what makes it interchangeable with YoloDetector.
    """

    def __init__(self, params: DetectorParams | None = None) -> None:
        self.params = params or DetectorParams()

    def __call__(self, frame_bgr: np.ndarray) -> list[Detection]:
        return detect_golf_balls(frame_bgr, self.params)

    def __repr__(self) -> str:
        return f"ClassicalBackend(params={self.params})"


def make_detector(
    backend: str = "classical",
    *,
    params: DetectorParams | None = None,
    model: str | None = None,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.45,
    coco_class: int | None = None,
) -> BallDetector:
    """Build a detector for `backend` (see BACKENDS).

    `coco_class` selects the YOLO class filter: None keeps the default
    sports-ball filter for stock COCO weights, a negative value disables
    filtering (correct for our own fine-tuned single-class model), and any other
    value filters to exactly that class id.
    """
    if backend == "classical":
        return ClassicalBackend(params)

    if backend == "yolo":
        # Imported lazily: 'import ultralytics' drags in torch and costs seconds.
        # The classical path must never pay that.
        from yolo_detector import COCO_SPORTS_BALL, DEFAULT_MODEL, YoloDetector

        if coco_class is None:
            class_filter = COCO_SPORTS_BALL
        elif coco_class < 0:
            class_filter = None
        else:
            class_filter = coco_class

        return YoloDetector(
            model_path=model or DEFAULT_MODEL,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            class_filter=class_filter,
        )

    raise ValueError(f"Unknown backend {backend!r}; expected one of {BACKENDS}")


def add_detector_args(parser: argparse.ArgumentParser, *, select_backend: bool = True) -> None:
    """Add the shared backend-selection flags to a script's parser.

    Centralised so run_webcam.py, test_image.py and benchmark.py cannot drift
    apart on flag names or defaults. benchmark.py runs every backend, so it
    passes `select_backend=False` to omit --backend.
    """
    group = parser.add_argument_group("detector")
    if select_backend:
        group.add_argument(
            "--backend", choices=BACKENDS, default="classical",
            help="detection backend (default: classical)",
        )
    group.add_argument(
        "--params", metavar="PATH", default=None,
        help="classical backend: tuned thresholds JSON (see calibrate.py)",
    )
    group.add_argument(
        "--model", metavar="PATH", default=None,
        help="yolo backend: .pt weights, .onnx, an exported NCNN dir, or a "
             "Hailo export dir for the AI HAT+ 2 (see export_hailo.py) "
             "(default: yolo11n.pt, downloaded on first use)",
    )
    group.add_argument(
        "--imgsz", type=int, default=640,
        help="yolo backend: inference size; raise to 960 to help small/distant "
             "balls at the cost of speed (default: 640)",
    )
    group.add_argument(
        "--conf", type=float, default=0.25,
        help="yolo backend: confidence threshold (default: 0.25)",
    )
    group.add_argument(
        "--iou", type=float, default=0.45,
        help="yolo backend: NMS IoU threshold (default: 0.45)",
    )
    group.add_argument(
        "--coco-class", type=int, default=None, metavar="ID",
        help="yolo backend: class id to keep. Default filters to COCO 'sports "
             "ball' (32) for stock weights; pass -1 for our fine-tuned "
             "single-class model",
    )


def detector_from_args(args: argparse.Namespace) -> BallDetector:
    """Build the detector described by flags added via add_detector_args().

    A missing optional dependency exits with the install hint rather than a
    traceback: "you didn't pip install this" is a user error, not a crash.
    """
    params = None
    if getattr(args, "params", None):
        params = DetectorParams.load(args.params)
        print(f"Loaded params from {args.params}")

    try:
        detector = make_detector(
            args.backend,
            params=params,
            model=args.model,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            coco_class=args.coco_class,
        )
    except ImportError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        raise SystemExit(1) from None

    print(f"Backend: {detector!r}")
    return detector
