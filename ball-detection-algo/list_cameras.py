"""Find the cameras: the Pi CSI camera (robot) and which index is the USB webcam.

OpenCV addresses cameras by number, and the USB cam's index varies per machine
(often 0 or 1 on a laptop; typically 0 on the RP5 when it's the only camera).

This shows a LIVE preview of each working camera, one at a time. Look at the
video, and when you see the USB webcam's view, note the index shown in the
window title/overlay. Then pass it as `--camera N` to the other scripts.

CSI cameras (the robot's Pi Camera Module 3 Wide) are checked first. They are
not OpenCV indices: they are listed through Picamera2 and previewed the same
way, and the flag to use is printed (`--camera csi`). On the laptop, which has
no Picamera2, the CSI check is skipped with a note. If the Pi reports no CSI
camera, `rpicam-hello --list-cameras` tells you whether libcamera sees it at all
(if not, it's the ribbon cable or OS, not this code).

Controls (while a preview is showing):
    n  -> next camera
    q  -> quit

Usage:
    python list_cameras.py            # live-preview CSI cameras, then USB indices 0..5
    python list_cameras.py --max 8    # probe USB indices 0..8
    python list_cameras.py --no-show  # just print what opens (headless)
"""

from __future__ import annotations

import os

# Silence OpenCV's videoio WARN spam: probing camera indices that don't exist is
# expected here, and each miss otherwise prints a scary-looking backend warning.
# Must be set before cv2 is imported (including via `camera` below).
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import argparse

import cv2

from camera import _preferred_backend, open_camera

WINDOW = "camera preview  (n = next, q = quit)"


def csi_flag(num: int) -> str:
    """The --camera value for CSI camera number `num` (see camera.camera_source)."""
    return "csi" if num == 0 else f"csi{num}"


def find_csi_cameras() -> tuple[list[dict], str | None]:
    """List CSI cameras via Picamera2 without opening them.

    Returns (cameras, note): `cameras` are Picamera2.global_camera_info() entries
    ("Model", "Num", ...), with USB webcams that libcamera also enumerates
    filtered out; `note` explains why the check was skipped, else None.
    """
    try:
        from picamera2 import Picamera2
    except ImportError:  # the laptop, or a Pi venv without --system-site-packages
        return [], ("CSI: skipped, picamera2 not importable (expected on the laptop; on "
                    "the Pi run 'sudo apt install python3-picamera2' and use a venv "
                    "created with --system-site-packages)")
    cameras = [info for info in Picamera2.global_camera_info()
               if "usb" not in str(info.get("Id", "")).lower()]
    return cameras, None


def _open_csi(info: dict, width: int = 1280, height: int = 720):
    """Open a listed CSI camera and confirm it delivers a frame, else None."""
    try:
        cap = open_camera(f"csi{info['Num']}", width, height)
    except RuntimeError as exc:  # e.g. busy: another process holds the camera
        print(f"{csi_flag(info['Num'])} ({info.get('Model', '?')}): failed to open: {exc}")
        return None
    ok, _ = cap.read()
    if ok:
        return cap
    cap.release()
    return None


def _preview(cap, label: str) -> bool:
    """Show a live preview until n or q. Returns True if the user quit."""
    print(f"{label}: live preview (press n for next, q to quit)")
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            return False
        h, w = frame.shape[:2]
        cv2.putText(frame, f"{label}   {w}x{h}   n=next  q=quit",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("n"):
            return False
        if key == ord("q"):
            return True


def _open(index: int, backend: int) -> cv2.VideoCapture | None:
    """Open an index and confirm it actually delivers a frame, else None."""
    cap = cv2.VideoCapture(index, backend)
    if cap.isOpened():
        ok, _ = cap.read()
        if ok:
            return cap
    cap.release()
    return None


def preview_each(max_index: int, csi: list[dict]) -> tuple[list[str], list[int]]:
    """Live-preview each working camera in turn, CSI first.

    Returns (working CSI flags, working USB indices).
    """
    backend = _preferred_backend()
    working_csi: list[str] = []
    working: list[int] = []
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    quit_all = False
    # CSI before USB: probing the Pi's V4L2 nodes must not race Picamera2 for them.
    for info in csi:
        cap = _open_csi(info)
        if cap is None:
            continue
        flag = csi_flag(info["Num"])
        working_csi.append(flag)
        quit_all = _preview(cap, f"--camera {flag} ({info.get('Model', '?')})")
        cap.release()
        if quit_all:
            break

    consecutive_misses = 0
    for index in range(max_index + 1):
        if quit_all:
            break
        cap = _open(index, backend)
        if cap is None:
            consecutive_misses += 1
            # Camera indices are contiguous from 0; a run of misses means we're done.
            if consecutive_misses >= 3 and working:
                break
            continue
        consecutive_misses = 0
        working.append(index)
        quit_all = _preview(cap, f"index {index}")
        cap.release()

    cv2.destroyAllWindows()
    for _ in range(4):  # pump the GUI so the window actually closes on Windows
        cv2.waitKey(1)
    return working_csi, working


def list_only(max_index: int, csi: list[dict]) -> tuple[list[str], list[int]]:
    """Headless: open each camera, report which deliver frames, no windows."""
    working_csi: list[str] = []
    for info in csi:
        flag = csi_flag(info["Num"])
        cap = _open_csi(info)
        if cap is None:
            print(f"{flag} ({info.get('Model', '?')}): listed but delivered no frame")
            continue
        _, frame = cap.read()
        h, w = frame.shape[:2]
        print(f"{flag} ({info.get('Model', '?')}): OPEN, frame {w}x{h}")
        working_csi.append(flag)
        cap.release()

    backend = _preferred_backend()
    working: list[int] = []
    consecutive_misses = 0
    for index in range(max_index + 1):
        cap = _open(index, backend)
        if cap is None:
            consecutive_misses += 1
            if consecutive_misses >= 3 and working:
                break
            print(f"index {index}: not available")
            continue
        consecutive_misses = 0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"index {index}: OPEN, frame {w}x{h}")
        working.append(index)
        cap.release()
    return working_csi, working


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max", type=int, default=5, help="highest index to probe")
    parser.add_argument("--no-show", dest="show", action="store_false",
                        help="don't open preview windows (just list working indices)")
    args = parser.parse_args()

    csi, note = find_csi_cameras()
    if note:
        print(note)
    elif not csi:
        print("CSI: no camera found. Check the ribbon cable and run "
              "'rpicam-hello --list-cameras'.")

    if args.show:
        working_csi, working = preview_each(args.max, csi)
    else:
        working_csi, working = list_only(args.max, csi)

    if working_csi:
        print(f"\nWorking CSI cameras: {working_csi}")
        print("This is the robot camera:")
        print(f"    python run_webcam.py --camera {working_csi[0]}")
    if working:
        print(f"\nWorking USB camera indices: {working}")
        print("Use the index whose preview showed the USB webcam:")
        print("    python run_webcam.py --camera <N>")
    if not working_csi and not working:
        print("\nNo cameras opened. Is the camera connected and not in use by another app?")


if __name__ == "__main__":
    main()
