# Autonomous Golf Ball Picker Robot

Senior capstone project: a robot that periodically sweeps a putting/chipping green, collects golf balls, returns to its base station, and unloads them into a ball-stacking system.

Instead of running continuously, the robot goes out about every 30 minutes and recharges at its dock in between.

> **Status:** early stage. The repo contains the system plan, the computer-vision ball detector, and Pi-to-VEX V5 motor control, and camera-guided driving to balls on the VEX prototype. The final drivetrain, collection, docking, and full navigation are in design.

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
| [`navigation/`](navigation/) | Camera-guided driving: turns detections into tank-drive commands for the VEX prototype |

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

For the YOLO backend, Pi/Hailo setup, calibration, and training a custom model, see **[`ball-detection-algo/README.md`](ball-detection-algo/README.md)**. Tests run with `python -m pytest tests` from `ball-detection-algo/` after `pip install -r requirements-dev.txt`; they need no camera or ML install.

## Motor control (VEX V5 prototype)

[`rpi-vex-protocol/`](rpi-vex-protocol/) lets the Pi drive VEX V5 motors through the Brain's micro-USB port, so drivetrain and mechanism prototypes can run from the same Pi that does vision.

- **[`brain/main.py`](rpi-vex-protocol/brain/main.py)** runs on the V5 Brain (VEXcode V5 Python). It is a line-based serial command server. Its first line, `# vex:disable=repl`, must stay first; without it the REPL takes over the USB port.
- **[`pi/vex_motor.py`](rpi-vex-protocol/pi/vex_motor.py)** runs on the Pi. It auto-detects the Brain's user serial port and works as a library (`VexBrain`) or as a CLI.

Protocol (ASCII, one command per line, replies `OK ...` or `ERR ...`):

| Command | Effect |
|---|---|
| `PING` | Heartbeat, replies `OK PONG` |
| `SPIN <port> <pct>` | Spin at -100..100 % velocity |
| `STOP <port\|ALL> [COAST\|BRAKE\|HOLD]` | Stop one motor or all |
| `MOVE <port> <deg> <pct>` | Go to an absolute encoder position (doesn't wait) |
| `RESET <port>` | Zero the encoder |
| `STATUS <port>` | Position, velocity, current, temperature, installed |

**Safety:** the Brain stops every motor if it hears nothing for 500 ms. `VexBrain` sends a heartbeat every 200 ms and stops all motors when it closes, so a crashed or unplugged Pi can't leave the robot driving.

```bash
# On the Pi: needs pyserial, and your user in the dialout group
sudo apt install python3-serial
python3 rpi-vex-protocol/pi/vex_motor.py list               # show VEX serial ports
python3 rpi-vex-protocol/pi/vex_motor.py spin 1 50 --seconds 2
python3 rpi-vex-protocol/pi/vex_motor.py move 1 360 40
python3 rpi-vex-protocol/pi/vex_motor.py status 1
```

```python
from vex_motor import VexBrain
with VexBrain() as brain:      # leaving the block stops every motor
    brain.spin(1, 50)
    print(brain.status(1))
```

`GEARING` in `brain/main.py` defaults to the green 18:1 cartridge. Change it if your motors use the red or blue cartridge.

## Driving to balls (VEX prototype)

[`navigation/`](navigation/) connects the two modules above. Each frame, the detector's boxes go to a pure controller that picks the closest ball (the one lowest in the frame). The controller turns to face it, drives up to it, and stops. A tank-drive layer then sends the wheel speeds to the Brain. Check motor directions with `drive.py` first, with the wheels off the ground:

```bash
cd ball-detection-algo
.venv/bin/python ../navigation/drive.py forward 30 --seconds 1    # every wheel forward?
.venv/bin/python ../navigation/chase.py --camera 1 --dry-run       # laptop: prints wheel commands only
.venv/bin/python ../navigation/chase.py --camera csi --max-pct 20  # robot
```

See [`navigation/README.md`](navigation/README.md) for the state machine, tuning flags and safety stops.

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
