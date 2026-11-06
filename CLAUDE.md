# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

Senior capstone for an **Autonomous Golf Ball Picker Robot** — a periodically-operating robot that scans a putting/chipping green (~every 30 min), collects golf balls, returns to a base station, and unloads them into a ball-stacking system. See `capstone_robot_system_plan.md` for the authoritative, up-to-date system concept, decision records, bill of materials, and open questions. **Read that document before making design or component recommendations** — it records decisions already made and their rationale, so proposing alternatives to settled choices (e.g. the controller) needs to engage with the reasoning there.

This is an **early-stage repository**: planning docs, the first computer-vision code (`ball-detection-algo/`, see below), and the Pi-to-VEX motor-control code. CAD lives on disk but is git-ignored. There is no repo-wide build system or CI; the only runnable code lives in the CV module and is driven by the scripts documented there. Its tests run with `python -m pytest tests` from `ball-detection-algo/` (needs `requirements-dev.txt`; no camera or ML install).

## Repository layout

- `capstone_robot_system_plan.md` — the single source of truth for the project. Update it when design decisions change rather than duplicating its content elsewhere.
- `ball_holder_model/` — Blender sources (`.blend`) and exported meshes (`.stl`) for the physical ball-holder fixture, iterated by version (`v1`, `v1.1`, `v1.2`). `.blend1` files are Blender's automatic backups. When adding an iteration, bump the version suffix and export a matching `.stl` alongside the `.blend`. **These files are git-ignored** (large binaries, no useful diff) — they exist on disk and are shared out-of-band, so don't assume a clone has them.
- `ball-detection-algo/` — the computer-vision ball-detection code (Python + OpenCV). See the dedicated section below.
- `rpi-vex-protocol/` — Raspberry Pi ↔ VEX V5 Brain motor control over USB serial: `brain/main.py` (VEXcode Python command server with a 500 ms watchdog) and `pi/vex_motor.py` (Pi-side pyserial client/CLI).

## Key settled decisions (from the plan doc)

- **Controller: Raspberry Pi 5**, chosen over the ESP32-P4 primarily for its Linux/Python/OpenCV vision ecosystem, built-in Wi-Fi, and lower project risk. The RP5 is the high-level brain; a **custom KiCad PCB** handles power distribution, motor-driver interfaces, and charging/docking — this is where the embedded-systems learning goal lives.
- **Drivetrain:** planned four-wheel drive with 4+ motor drivers. Keep **motor power and logic/computer power on separate rails**.
- **Camera:** Raspberry Pi Camera Module 3 Wide (IMX708, 120°, autofocus) on the RP5's CSI
  port. Mounting height/tilt is still open.
- **Vision:** YOLO11n neural detection is the primary path, with classical CV retained as a
  selectable fallback (see section 5 of the plan doc for the decision record). Still a
  high-value but *not top-priority* capability (priority #4) — hardware reliability
  (drivetrain, collection, return-to-dock) comes first.

## Computer-vision module (`ball-detection-algo/`)

Detects golf balls in a camera feed and draws a bounding box around each. The robot's camera
is a **Raspberry Pi Camera Module 3 Wide** (IMX708, 120° diagonal, autofocus, CSI ribbon),
selected with `--camera csi`; the dev laptop uses a USB webcam selected by index.
Written to run **unchanged on the RP5**; the laptop is just the dev box.
See `ball-detection-algo/README.md` for full usage.

**Design (deliberate, don't flatten it):** there are **two interchangeable backends**
behind one contract, `detector(frame) -> list[Detection]`, selected with `--backend`:

- **`classical`** — HSV white-thresholding + morphology + contour shape-gating (area,
  circularity, fill-ratio). Lives in `detector.py` as a **pure
  `detect_golf_balls(frame, params)`** (no camera/window/disk). All thresholds sit in one
  `DetectorParams` dataclass serialized to `params.json`, which is git-ignored because it is
  per-camera, per-lighting output of `calibrate.py`.
- **`yolo`** — Ultralytics YOLO11n in `yolo_detector.py`. A **callable object, not a pure
  function**, because the model must be loaded once and reused across frames.

`backends.py` holds the `make_detector()` factory and the shared CLI flags, and is the only
module importing both. **`detector.py` must never import `ultralytics` or `torch`** — that
separation is what keeps the classical path runnable on a bare OpenCV install, which is the
fallback if RP5 inference is too slow. The ML import is deferred into `YoloDetector.__init__`.

Detection stays an isolated node in the autonomous state machine (…→ scan → navigate →…);
swapping backends does not touch navigation. Camera capture, tuning, and display live in
separate scripts: the detection runners (`run_webcam.py`, `test_image.py`, `benchmark.py`) go
through the factory, while the classical-only tools (`calibrate.py`, `tune.py`) call
`detector.py` directly.

**Environment:** a venv lives at `ball-detection-algo/.venv`. Run scripts with
`.venv\Scripts\python.exe` (Windows) / `.venv/bin/python` (RP5). `pip install -r
requirements.txt` pulls prebuilt wheels on any Python 3.9–3.14 (OpenCV ships an abi3 wheel);
no version pin is needed despite the local Python being 3.14. **`requirements-yolo.txt` is
separate and optional** — it adds `ultralytics` (and torch, which is large). Keep the base
requirements file free of ML dependencies.

**Typical workflow:** pick the camera source — on the robot it is `--camera csi`; on the
laptop run `list_cameras.py` to find the USB cam's index (it is *not* the laptop's built-in).
Then either `calibrate.py --camera <src>` → `run_webcam.py --camera <src> --params
params.json` for the classical path, or `run_webcam.py --camera <src> --backend yolo` for the
neural one. `tune.py` is the manual slider fallback; `test_image.py` runs either backend on a
still photo; `benchmark.py` compares both over a folder; `capture_dataset.py` collects
training images.

**YOLO status:** `--backend yolo` currently runs **zero-shot** on stock COCO weights,
filtered to class 32 (`sports ball`), which needs no labelled data. A custom single-class
model is milestone 2 and requires a dataset that does not exist yet — the repo contains no
golf-ball imagery at all. With a fine-tuned model, pass `--coco-class -1` to disable the
filter, since a single-class model has no class 32.

**Hailo AI HAT+ 2 (Hailo-10H NPU on the RP5):** not a separate backend. It's a model
format for `--backend yolo`. `export_hailo.py` compiles a `<name>_hailo_model/` directory
(`.hef` + `metadata.yaml`) on the laptop, and `--model <that dir>` makes `YoloDetector`
run it through `hailo_runtime.py` (HailoRT InferModel API; no ultralytics/torch needed)
instead of `YOLO()`.

**Gotchas (hard-won, keep):**
- **Windows capture backend must be DirectShow (`CAP_DSHOW`), not MSMF** — MSMF is slow to
  open and *hangs* when probing a camera index that doesn't exist. `camera.py` selects DSHOW
  on Windows, V4L2 on Linux/RP5.
- **The interactive GUI scripts (`list_cameras.py`, `calibrate.py`, `tune.py`,
  `capture_dataset.py`, and `run_webcam.py`/`test_image.py` unless given `--no-display`/`--no-show`)
  must be run in the user's own terminal** (suggest the `!` prefix), never launched by the agent in the
  background — a backgrounded process has no interactive desktop, so its OpenCV window never
  receives keystrokes and the tool aborts.
- White-ball detection is lighting-sensitive; prefer `calibrate.py` over hand-tuning, and
  re-calibrate when lighting changes. `min_area`/`max_area` are always full-frame px², so
  `downscale` is a pure speed knob and never needs re-tuning. Calibrate at the distance the
  robot actually sees balls from — boxing balls held close to the camera sets a `min_area` floor that silently rejects
  everything on the far half of the green.
- **Distant balls are hard for YOLO too**, not just for the classical gates: at `--imgsz 640`
  a 720p frame is halved, so a 15 px ball lands near the stride-8 head's limit. Raise
  `--imgsz`, or let the scan → navigate loop drive closer and re-scan (preferred; no model
  change).
- **Hailo:**
  - **The Pi package is `hailo-h10-all`**, not `hailo-all` (that one is Hailo-8/8L).
    The two can't be installed together, and a HEF only runs on the chip it was compiled
    for (`--arch hailo10h`).
  - **HEF export runs only on Linux x86_64** with the Hailo DFC 5.x wheel (Developer
    Zone, not PyPI), in its own `.venv-dfc`, because the wheel doesn't support the main
    venv's Python 3.14. Never try to export on the Pi.
  - **The Pi venv must be created with `--system-site-packages`*
  - **Input size and the NMS conf/IoU floors are baked in at export.** `--imgsz` is
    ignored, and `--conf` can only be raised.
- **CSI camera ⇒ Picamera2, not `cv2.VideoCapture`.** The Pi 5 exposes the Camera Module 3
  only through libcamera. `camera.py`'s `PiCamera` wraps Picamera2 in the `read()`/`release()`
  slice of the VideoCapture API, so scripts and detectors don't know which camera is attached.
  Picamera2 is apt-only (`python3-picamera2`), so it is imported lazily like `ultralytics`;
  never add it to requirements. Its `"RGB888"` format is already BGR in memory, so frames need
  no color conversion.
- **The 120° wide lens shrinks distant balls** (~10 px across at 3 m in a 1280 px frame) and
  distorts them toward the edges. Build the training dataset from this camera at robot height,
  not from a laptop webcam.

## Team context

Four-member team: Will (lead), Micah (computer/software — the primary user here), Tiffany (research/docs), Isaac (hardware). Software and CV work is Micah's lane; KiCad electronics is a shared learning effort.
