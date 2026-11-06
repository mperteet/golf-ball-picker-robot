# Golf Ball Detection

Golf-ball detector for the autonomous golf ball picker robot. Opens a camera,
finds golf balls in each frame, and draws a bounding box around each one.

Two interchangeable detection backends:

| Backend | How it works | Use it for |
|---|---|---|
| `classical` (CLI default) | HSV white-thresholding + morphology + contour shape gates. No ML, no dependencies beyond OpenCV. | Fast, predictable, controlled/indoor light. The fallback. |
| `yolo` | Ultralytics YOLO11n neural detector. | Shade, uneven outdoor light, and balls at distance, where fixed thresholds fail. |

Both satisfy the same contract, `detector(frame) -> list[Detection]`, so
navigation and the autonomous state machine never learn which one ran. Pick with
`--backend`. YOLO is the project's primary path (see the plan doc, section 5);
`classical` is only the command-line default because it runs on the base install.

This code is written to run **unchanged on the Raspberry Pi 5**; the laptop is
just the development environment. The robot's camera is a **Raspberry Pi Camera
Module 3 Wide** (IMX708, 120° diagonal, autofocus) on the Pi's CSI ribbon port,
selected with `--camera csi`. The laptop uses any USB webcam, selected by index.

## Files

| File | What it does |
|---|---|
| `detector.py` | Classical detection logic and the shared `Detection` type. `detect_golf_balls(frame, params)` is a pure function. Imports no ML libraries, by design. |
| `yolo_detector.py` | `YoloDetector`, the neural backend. A callable object, because a model must be loaded once and reused. |
| `backends.py` | `make_detector(...)` factory plus the shared CLI flags. The only module importing both detectors. |
| `camera.py` | Opens the camera: the Pi CSI camera via Picamera2 (`--camera csi`), or a USB webcam via OpenCV with the right backend per OS. |
| `list_cameras.py` | Probe tool: lists the Pi CSI camera (and its `--camera csi` flag) and finds which index is the USB webcam. |
| `calibrate.py` | **Classical auto-tune**: box a few real golf balls and it computes thresholds. |
| `tune.py` | Live trackbar tuner for hand-adjusting classical thresholds. |
| `run_webcam.py` | Main live detector on the camera feed. |
| `test_image.py` | Run detection on a still image (no camera). |
| `benchmark.py` | Run both backends over a folder and compare, side by side. |
| `capture_dataset.py` | Collect training images from the camera. |
| `export_hailo.py` | Compile a YOLO model to a Hailo HEF for the AI HAT+ 2. Runs on the laptop only. |
| `tests/` | pytest suite for the classical detector, backend factory, and camera selection. Needs no camera, window, or ML install. |

## Setup

Any **Python 3.9-3.14** works. A virtual environment is recommended.

```bash
python3 -m venv .venv           # venv is required on RP5 Bookworm (PEP 668)
source .venv/bin/activate       # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**On the Pi, create the venv with `python3 -m venv --system-site-packages .venv`
instead.** The CSI camera (Picamera2) and the Hailo runtime are both installed
by apt into the system Python, and a plain venv can't see them.

That is everything the classical backend needs. The YOLO backend is **optional**
and installed separately, because `ultralytics` pulls in torch.

**Install CPU-only torch first.** Installing `ultralytics` straight away drags in
torch's default CUDA build: the wheel, cuDNN, and the NVIDIA runtime libraries
total several GB. No machine on this project has an NVIDIA GPU, and the Raspberry
Pi 5 is ARM, so all of it is dead weight.

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements-yolo.txt
```

The second command sees torch already satisfied and leaves it alone. Train on a
CUDA machine or Colab instead, and bring back only the weights.

Keeping the two requirement files apart is deliberate: if inference is too slow on
the robot, the classical backend still runs on a machine with only OpenCV and
NumPy.

## Usage

```bash
# 1. Find the USB webcam's index (not the laptop's built-in):
python list_cameras.py

# 2a. Classical: auto-tune for your lighting, then run.
python calibrate.py --camera 1                              # writes params.json
python run_webcam.py --camera 1 --params params.json

# 2b. YOLO: no tuning step, works out of the box.
python run_webcam.py --camera 1 --backend yolo

# Offline: test either backend on a saved photo.
python test_image.py photo.jpg --backend yolo
```

Press `q` to quit any live window.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests
```

The suite runs on synthetic frames, so it needs only the base requirements plus
pytest. It also checks that the classical path never imports `ultralytics` or
`torch`.

## Choosing a backend

The classical detector fails in two specific ways, and both are inherent rather
than a tuning mistake:

- **Shade and uneven light.** It accepts a pixel only above a fixed brightness,
  so a ball in shadow drops out. Lowering the threshold to catch it starts
  boxing bright grass and concrete instead. There is no setting that survives a
  green with both sun and shade on it.
- **Distance.** Calibration sets an area floor from the balls you boxed. Balls
  further away fall below it and are discarded before the shape checks run.

YOLO keys on shape and context rather than absolute brightness, so it handles
both. It costs roughly an order of magnitude more compute per frame.

Run `benchmark.py` on your own photos rather than taking the above on faith:

```bash
python benchmark.py photos/ --params params.json --out comparison/
```

It reports per-image counts for both backends and writes side-by-side annotated
pairs. Counts alone don't prove correctness, so open the comparisons and confirm
the extra boxes are real balls.

## Tuning the classical backend

White-ball detection is lighting-sensitive; the same thresholds rarely work in
two different rooms.

### Auto-tune (recommended): `calibrate.py`
```bash
python calibrate.py --camera 1              # or: --image photo.jpg
```
Press SPACE to freeze a frame, drag a box around each golf ball (ENTER after
each; ENTER on an empty box when done). Re-calibrate whenever lighting changes.

**Calibrate at the distance the robot will actually see balls from.** Boxing
balls held close to the camera sets an area floor that rejects everything on the
far half of the green.

### Manual: `tune.py`
```bash
python tune.py --camera 1 --params params.json --out params.json
```
The right pane shows the raw white mask, so you can see what the color gate
selects before shape filtering.

### If detection is still off
- **Missing balls** → lower `val_min` (accept dimmer whites) or raise `sat_max`.
- **Missing distant balls** → lower `min_area`. Areas are always in full-frame
  pixels, so changing `downscale` never requires re-tuning them.
- **Boxing bright non-balls** → raise `min_circularity` / `min_fill_ratio`, or
  tighten `min_area`/`max_area`.

## Training a custom YOLO model

The default `--backend yolo` runs **zero-shot**: stock COCO weights filtered to
the `sports ball` class (id 32), which already finds golf balls with no labelling
at all. Use it to decide whether the ML path is worth pursuing before investing
in a dataset.

To go further, train a single-class model:

1. **Capture.** `python capture_dataset.py --camera csi --out datasets/raw` (on the Pi)
   Shoot the conditions that currently fail: shade, overcast, low sun, balls at
   range. Include frames with no balls. Shoot from the robot's camera height,
   ideally on the Pi with `--camera csi`: the wide lens distorts balls in ways
   a laptop webcam's images won't teach the model.
2. **Label.** Roboflow or CVAT, exported in YOLO format, one class: `ball`.
   200-500 images is a reasonable target for a single-class detector.
3. **Train.** Off this machine. It has no CUDA GPU, so a local run takes many
   hours against roughly half an hour on a free Colab T4.
   ```bash
   yolo detect train model=yolo11n.pt data=data.yaml epochs=100 imgsz=640
   ```
   Leave HSV and scale augmentation on; lighting invariance is the entire point.
4. **Use it.** Pass the weights and disable the COCO class filter, since a
   single-class model has no class 32:
   ```bash
   python run_webcam.py --camera 1 --backend yolo --model best.pt --coco-class -1
   ```

## Running on the Raspberry Pi 5

- **Camera:** the robot uses a **Pi Camera Module 3 Wide** on the CSI ribbon
  connector. The Pi 5 exposes CSI cameras only through libcamera, so frames come
  from **Picamera2**, not `cv2.VideoCapture`. Pass `--camera csi` (or `csi1` for
  the second port) to any script; nothing else changes. Picamera2 comes from apt
  and needs the same `--system-site-packages` venv as the Hailo runtime:
  ```bash
  sudo apt install python3-picamera2
  rpicam-hello --list-cameras        # should list an imx708_wide
  python run_webcam.py --camera csi --no-display
  ```
  Continuous autofocus is switched on at startup. The camera ships with a 15 cm
  15-to-22-pin FFC cable. The Pi 5's CSI ports are 22-pin, so it fits, but 15 cm
  constrains where the camera can mount relative to the Pi. A USB webcam still
  works on the Pi by index, as on the laptop.
- **Headless robot (no monitor):** add `--no-display` to `run_webcam.py`. Use
  `--save out.jpg` to write the latest annotated frame, or import the detector
  directly and feed the boxes to the navigation code.
- **YOLO on the Pi:** export to NCNN, which is meaningfully faster than PyTorch
  on ARM. `--model` loads the exported directory through the same code path.
  ```bash
  yolo export model=best.pt format=ncnn
  python run_webcam.py --camera csi --backend yolo --model best_ncnn_model --coco-class -1
  ```
  Expect a few frames per second on the CPU. That is adequate for a robot that
  stops, scans, then drives. With the AI HAT+ 2 attached, use a Hailo export
  instead (next section).
- **Small distant balls** are hard for YOLO too: at `--imgsz 640` a 720p frame is
  halved, so a 15 px ball becomes ~7 px. Raise `--imgsz` to 960 (slower), or let
  the robot drive closer and re-scan, which is the better robotics answer. The
  wide lens makes this worse: spread over ~102° horizontally, a 1280 px frame
  gives roughly 12 px per degree, so a ball 3 m away is only ~10 px across before
  any YOLO downscaling (less toward the edges, where the lens compresses the
  image). If range matters more than speed, raise `--imgsz` together with a
  higher capture `--width/--height`; either alone is capped by the other.

## Hailo AI HAT+ 2 (NPU)

The robot's Pi carries an **AI HAT+ 2 (Hailo-10H)**. YOLO runs on it through the
same `--backend yolo` path. There is no separate backend; only `--model` changes.
A Hailo export directory is run by `hailo_runtime.py` through HailoRT's InferModel
API, not by Ultralytics: Ultralytics' Hailo backend uses the legacy
configure/InferVStreams API, which the Hailo-10H rejects with
`HAILO_NOT_IMPLEMENTED` (just as `hailortcli run` is rejected in favor of `run2`).

A HEF is compiled ahead of time on the **laptop** (Linux x86_64 only), then
copied to the Pi.

### 1. Pi: install the Hailo runtime (once)

```bash
sudo apt update && sudo apt full-upgrade
sudo apt install dkms hailo-h10-all      # AI HAT+ 2. NOT hailo-all (that's Hailo-8/8L;
sudo reboot                              # the two packages can't coexist)
hailortcli fw-control identify           # should report the Hailo-10H
```

`hailo-h10-all` installs HailoRT's Python bindings into the **system** Python,
so rebuild the project venv so it can see them:

```bash
rm -rf .venv
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
.venv/bin/pip install -r requirements-yolo.txt
.venv/bin/python -c "import hailo_platform"   # must succeed
```

### 2. Laptop: compile the HEF

The Hailo Dataflow Compiler (DFC) is a wheel from the
[Hailo Developer Zone](https://hailo.ai/developer-zone/software-downloads/)
(free account). The Hailo-10H needs **DFC 5.x** (3.x is for Hailo-8/8L). The
wheel supports only certain Python versions, which don't include the 3.14 in
`.venv`, so give it its own venv using the Python version in the wheel's `cpXY`
tag:

```bash
python3.X -m venv .venv-dfc                 # X = the wheel's Python version
.venv-dfc/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
.venv-dfc/bin/pip install -r requirements-yolo.txt ~/Downloads/hailo_dataflow_compiler-*.whl
.venv-dfc/bin/python export_hailo.py        # -> yolo11n_hailo_model/
```

Compiling takes a while (INT8 quantization plus calibration). With no `--data`,
it calibrates on COCO, which is right for the stock zero-shot model.

### 3. Run it on the Pi

Copy the **whole** directory, because `metadata.yaml` must stay next to the `.hef`:

```bash
scp -r yolo11n_hailo_model <user>@<pi>:<repo>/ball-detection-algo/
# on the Pi:
.venv/bin/python run_webcam.py --camera csi --backend yolo --model yolo11n_hailo_model
```

The startup line prints `hailo=True` when the NPU is in use. Compare its FPS
against the CPU model with `--no-display`.

### Things that are fixed at export

- **Input size.** `--imgsz` is ignored for a HEF. To try 960 for distant balls,
  re-export with `export_hailo.py --imgsz 960`.
- **NMS floor.** `export_hailo.py --conf/--iou` are compiled in. At runtime,
  `--conf` can only be *raised* above the exported value.
- **Chip.** A `hailo10h` HEF won't run on a Hailo-8/8L, or the other way round.

### Custom golf-ball model

Train as in "Training a custom YOLO model", then calibrate on **our own**
images. Calibrating on COCO would tune the INT8 ranges for the wrong scenes.

```bash
.venv-dfc/bin/python export_hailo.py --weights best.pt --data data.yaml
.venv/bin/python run_webcam.py --camera csi --backend yolo --model best_hailo_model --coco-class -1
```

Hailo recommends about 1,000 calibration images. With fewer, expect some INT8
accuracy loss relative to `best.pt`; check it with `benchmark.py`.

## Not done yet (future work)

Distance/size estimation, tracking a ball across frames, and handing targets to
navigation. A raw `onnxruntime` backend, which would drop the torch dependency on
the Pi to about a tenth the size, if the install footprint becomes a problem.
