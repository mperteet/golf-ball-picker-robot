# Autonomous Golf Ball Picker Robot

Senior capstone project: a robot that periodically sweeps a putting/chipping green, collects golf balls, returns to its base station, and unloads them into a ball-stacking system.

Instead of running continuously, the robot goes out about every 30 minutes and recharges at its dock in between.

> **Status:** early stage. The repo contains the system plan and the computer-vision ball detector. Drivetrain, collection, docking, and navigation are in design.

## How it works

The planned operating cycle:

1. **Idle / charge** at the base station
2. **Launch** on a schedule (~every 30 min)
3. **Scan** the green with the on-board camera
4. **Navigate** to detected balls
5. **Collect** them with the on-board mechanism
6. **Return** and **dock** to charge
7. **Unload** into the base station's golf-ball stacker

| Subsystem | Direction |
|---|---|
| Computer | Raspberry Pi 5 |
| Vision | Raspberry Pi Camera Module 3 Wide (CSI, 120°) + YOLO11n on a Raspberry Pi AI HAT+ 2 (Hailo-10H NPU), with a classical-CV fallback |
| Electronics | Custom KiCad PCB for power distribution, motor-driver interfaces, and charging/docking |
| Drive | Four-wheel drive, with motor power kept on a separate rail from logic power |
| Base station | Charging dock + golf-ball stacker |

The full concept, decision records (including why the Pi 5 was chosen over the ESP32-P4), bill of materials, and open questions are in **[`capstone_robot_system_plan.md`](capstone_robot_system_plan.md)**.

## Repository layout

| Path | Contents |
|---|---|
| [`capstone_robot_system_plan.md`](capstone_robot_system_plan.md) | The system plan and the source of truth for design decisions |
| [`ball-detection-algo/`](ball-detection-algo/) | Golf-ball detection in Python + OpenCV, covered below |
| [`rpi-vex-protocol/`](rpi-vex-protocol/) | Raspberry Pi to VEX V5 Brain motor control over USB serial |

CAD for the ball-holder fixture (Blender `.blend` / `.stl`) is kept out of git and shared separately.

## Ball detection

The robot's "scan" step. It finds golf balls in a camera frame and returns bounding boxes. Two interchangeable backends share one interface, `detector(frame) -> list[Detection]`:

- **`yolo`** (primary): Ultralytics YOLO11n. On the robot it runs on the Hailo NPU, compiled to a HEF by `export_hailo.py`.
- **`classical`** (fallback): HSV white-thresholding plus shape checks. It needs only OpenCV.

Quick start:

```bash
cd ball-detection-algo
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python list_cameras.py                                     # find the USB webcam index
python run_webcam.py --camera 1                            # classical backend, laptop USB webcam
python run_webcam.py --camera csi                          # on the robot: Pi Camera Module 3 Wide
```

For the YOLO backend, Pi/Hailo setup, calibration, and training a custom model, see **[`ball-detection-algo/README.md`](ball-detection-algo/README.md)**.

## Priorities

Hardware reliability comes before autonomy:

1. Four-wheel drivetrain and manual control
2. Ball collection mechanism
3. Return-to-base and docking
4. Camera pipeline and ball detection
5. Autonomous navigation and collection
6. Unloading, stacking, then optional cleaning and solar charging

## Team

| Member | Role |
|---|---|
| Will | Team lead |
| Micah | Computer / software |
| Tiffany | Research / documentation |
| Isaac | Hardware |
