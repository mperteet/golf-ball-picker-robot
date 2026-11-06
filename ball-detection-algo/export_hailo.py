"""Compile a YOLO model to a Hailo HEF for the Raspberry Pi AI HAT+ 2.

Runs on the DEV LAPTOP, never on the Pi: Hailo's Dataflow Compiler (DFC) only
exists for Linux x86_64. The output is a directory (`<weights>_hailo_model/`)
holding the .hef plus the metadata.yaml Ultralytics needs to load it. Copy the
WHOLE directory to the Pi and pass it to --model; nothing else changes.

A HEF is compiled for one chip. The AI HAT+ 2 is a Hailo-10H (the default here);
the older AI HAT+ / AI Kit are hailo8 / hailo8l and need a separate export.
Hailo-10H needs DFC 5.x; hailo8/hailo8l need DFC 3.x.

Usage (from the DFC venv, see README):
    python export_hailo.py                                    # zero-shot yolo11n, COCO calibration
    python export_hailo.py --weights best.pt --data data.yaml  # our fine-tuned golf-ball model

The input size, and the NMS confidence/IoU floors, are baked in at export. At
runtime --conf can only be raised above the exported value, never lowered.
"""

from __future__ import annotations

import argparse
import platform
import sys

HAILO_ARCHS = ("hailo10h", "hailo8", "hailo8l")

_DFC_HELP = (
    "Hailo export needs the Hailo Dataflow Compiler (the 'hailo_sdk_client' "
    "package), which is not on PyPI.\n"
    "    1. Download the DFC wheel from the Hailo Developer Zone (free account):\n"
    "       https://hailo.ai/developer-zone/software-downloads/\n"
    "       DFC 5.x for hailo10h (AI HAT+ 2); DFC 3.x for hailo8 / hailo8l.\n"
    "    2. Install it into a venv whose Python version matches the wheel's\n"
    "       cpXY tag, alongside CPU torch and requirements-yolo.txt.\n"
    "See README.md, 'Hailo AI HAT+ 2'."
)


def _preflight() -> None:
    """Fail early with a readable hint instead of a DFC traceback mid-export."""
    if not (sys.platform.startswith("linux") and platform.machine() == "x86_64"):
        sys.exit("Hailo export only runs on Linux x86_64 (the dev laptop), "
                 f"not {sys.platform}/{platform.machine()}.")
    try:
        import hailo_sdk_client  # noqa: F401
    except ImportError:
        sys.exit(_DFC_HELP)
    try:
        from ultralytics import YOLO  # noqa: F401
    except ImportError:
        sys.exit("Export needs ultralytics too:\n"
                 "    pip install -r requirements-yolo.txt")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default="yolo11n.pt",
                        help="YOLOv8/YOLO11 detection weights (default: yolo11n.pt)")
    parser.add_argument("--arch", choices=HAILO_ARCHS, default="hailo10h",
                        help="target chip; AI HAT+ 2 is hailo10h (default)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="fixed input size compiled into the HEF (default: 640)")
    parser.add_argument("--data", default=None,
                        help="dataset YAML for INT8 calibration. Omit for stock "
                             "weights (uses COCO); pass our golf-ball data.yaml "
                             "for a fine-tuned model")
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="fraction of the dataset used for calibration (default: 1.0)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="NMS score floor baked into the HEF (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IoU threshold baked into the HEF (default: 0.45, "
                             "matching run_webcam.py)")
    args = parser.parse_args()

    _preflight()
    from ultralytics import YOLO

    out_dir = YOLO(args.weights).export(
        format="hailo",
        name=args.arch,
        imgsz=args.imgsz,
        data=args.data,
        fraction=args.fraction,
        conf=args.conf,
        iou=args.iou,
    )

    print(f"\nExported {out_dir}\n"
          "Copy the whole directory to the Pi (metadata.yaml must stay next to the .hef):\n"
          f"    scp -r {out_dir} <user>@<pi>:<repo>/ball-detection-algo/\n"
          "Then on the Pi:\n"
          f"    .venv/bin/python run_webcam.py --camera csi --backend yolo --model {out_dir}")
    if args.data:
        print("    (add --coco-class -1 for a single-class golf-ball model)")


if __name__ == "__main__":
    main()
