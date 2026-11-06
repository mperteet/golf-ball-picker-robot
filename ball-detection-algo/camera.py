"""Camera access: USB webcam (dev laptop) or the Pi CSI camera (robot).

The robot's camera is a **Raspberry Pi Camera Module 3 Wide** (IMX708, 120°
diagonal, autofocus) on the Pi 5's CSI ribbon connector. The Pi 5 exposes CSI
cameras only through libcamera, so `cv2.VideoCapture` cannot read it; we go
through Picamera2 instead. The dev laptop has no CSI port and keeps using a USB
webcam through `cv2.VideoCapture`.

Both sources are opened by `open_camera()` and return an object with the same
two methods the scripts use -- `read() -> (ok, frame_bgr)` and `release()` -- so
nothing downstream (and neither detector) knows which camera is attached.

Select the source with `--camera`:
    --camera 1       USB webcam at OpenCV index 1 (find it with list_cameras.py)
    --camera csi     Pi CSI camera on port 0 (csi1 for the second port)
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

_PICAMERA2_HELP = (
    "The CSI camera needs Picamera2, which comes from apt, not pip:\n"
    "    sudo apt install python3-picamera2\n"
    "and a venv created with --system-site-packages so it can see it "
    "(the Hailo runtime needs the same). See README, 'Running on the Raspberry Pi 5'."
)


def camera_source(value: str) -> int | str:
    """argparse `type=` for --camera: a USB index, or 'csi' / 'csi0' / 'csi1'."""
    text = value.strip().lower()
    if text.isdigit():
        return int(text)
    if text == "csi":
        return "csi0"
    if text in ("csi0", "csi1"):
        return text
    raise argparse.ArgumentTypeError(
        f"expected a USB camera index (e.g. 1) or 'csi'/'csi1', got {value!r}")


def add_camera_args(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    """Add --camera/--width/--height, shared by every script that opens a camera."""
    parser.add_argument(
        "--camera", type=camera_source, required=required, metavar="SOURCE",
        help="USB webcam index (find it with list_cameras.py), or 'csi' for the "
             "robot's Pi Camera Module 3 Wide ('csi1' for the second CSI port)",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)


def _preferred_backend() -> int:
    """Pick the most reliable capture backend for the current OS.

    On Windows we use DirectShow (DSHOW): for USB (UVC) webcams it opens faster
    and, unlike MSMF, doesn't hang when probing a camera index that isn't there.
    """
    if sys.platform.startswith("win"):
        return cv2.CAP_DSHOW
    if sys.platform.startswith("linux"):
        # V4L2 is what the Raspberry Pi 5 uses for USB webcams.
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


class PiCamera:
    """Picamera2 wrapped in the slice of the cv2.VideoCapture API we use.

    Frames come out as BGR uint8, ready for OpenCV and both detectors.
    Picamera2's "RGB888" format is, despite the name, stored B,G,R in memory
    (libcamera names formats by packed word order), which is OpenCV's order.
    """

    def __init__(self, port: int = 0, width: int = 1280, height: int = 720) -> None:
        try:
            from picamera2 import Picamera2
        except ImportError as exc:  # Pi-only, apt-installed; explain, don't traceback
            raise RuntimeError(_PICAMERA2_HELP) from exc

        self.port = port
        self._cam = Picamera2(port)
        config = self._cam.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"})
        self._cam.configure(config)
        self._cam.start()
        # Continuous autofocus: balls range from just ahead of the robot to the
        # far side of the green, so any fixed focus would blur one end of that.
        try:
            from libcamera import controls
            self._cam.set_controls({"AfMode": controls.AfModeEnum.Continuous})
        except (ImportError, RuntimeError):
            pass  # a fixed-focus module has no AF control; keep its default
        self._open = True

    def isOpened(self) -> bool:  # noqa: N802 - mirrors cv2.VideoCapture
        return self._open

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self._open:
            return False, None
        frame = self._cam.capture_array("main")
        return frame is not None, frame

    def release(self) -> None:
        if self._open:
            self._cam.stop()
            self._cam.close()
            self._open = False


def open_camera(source: int | str, width: int = 1280, height: int = 720,
                fourcc: str = "MJPG"):
    """Open a camera and configure it for a fast stream.

    `source` is a USB index (cv2.VideoCapture) or 'csi0'/'csi1' (Picamera2); see
    `camera_source()`. Either way the result has `read()` and `release()`.

    For USB, requests MJPG so webcams can deliver higher resolutions at usable
    frame rates (raw YUYV is bandwidth-limited over USB), and falls back to
    CAP_ANY if the preferred backend can't open the device.

    Raises RuntimeError if the camera cannot be opened.
    """
    if isinstance(source, str):
        if not source.startswith("csi"):
            raise ValueError(f"unknown camera source {source!r}")
        return PiCamera(int(source[3:] or 0), width, height)

    backend = _preferred_backend()
    cap = cv2.VideoCapture(source, backend)
    if not cap.isOpened():
        # Retry with the platform-agnostic backend before giving up.
        cap.release()
        cap = cv2.VideoCapture(source, cv2.CAP_ANY)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {source}. "
            f"Run 'python list_cameras.py' to find the USB webcam's index."
        )

    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap
