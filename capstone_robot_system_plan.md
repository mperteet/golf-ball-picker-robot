# Autonomous Golf Ball Picker Robot

## Senior Capstone — Consolidated Concept & Preliminary System Plan

## 1. Project Overview

The team is developing an autonomous golf ball picker robot for putting and chipping greens. The robot will locate and retrieve golf balls and unload them into a golf-ball stacking system. The project is intended to provide a more convenient practice experience for individual golfers.

The current concept is a periodically operating robot rather than a machine that must run continuously. The robot will go out approximately every 30 minutes to scan the field, collect balls, and return to its base. While idle, it can recharge at the base station. Solar charging may be considered as a supplemental option.

## 2. Current System Concept

| Subsystem | Current direction |
|---|---|
| Central computer | Raspberry Pi 5 (RP5) — new team choice |
| Vision | Raspberry Pi Camera Module 3 Wide (CSI) on the RP5; YOLO11n neural detection with a classical-CV fallback (see section 5) |
| Drive | Potential four-wheel-drive drivetrain with at least four motor drivers |
| Ball collection | Mechanical collection system mounted on the robot |
| Ball cleaning | Potential cleaning mechanism |
| Ball handling | Robot returns collected balls to the base station |
| Stacking | Base station includes a golf-ball stacker to organize returned balls into stacks |
| Charging | Docking base with rechargeable-battery charging; solar may supplement charging |
| Communications | RP5 provides the main computing platform and can handle wireless communication |

## 3. Why We Chose the Raspberry Pi 5

- **Computer vision:** The RP5 gives the project a strong general-purpose computing environment for camera processing and computer-vision development.
- **Development speed:** A Linux-based computer makes it easier to use Python, OpenCV, debugging tools, and existing software libraries.
- **Central controller:** It can coordinate navigation, vision, ball collection, communication, and the overall autonomous state machine.
- **Wireless capability:** Built-in wireless networking reduces the need for a separate communications computer.
- **Project risk:** Compared with building the system around a newer embedded processor, the RP5 is a more familiar and accessible development platform.
- **Embedded learning remains possible:** The team can still design custom electronics and motor-driver interfaces in KiCad while keeping the high-level software on the RP5.

## 4. Raspberry Pi 5 vs. ESP32-P4 — Decision Record

| Category | Raspberry Pi 5 | ESP32-P4 |
|---|---|---|
| Vision / camera software | Excellent; straightforward Linux/Python ecosystem | Capable, but more embedded-oriented |
| Autonomous control | Excellent for high-level control | Excellent for embedded real-time control |
| Motor control | Needs external motor-driver hardware | Needs external motor-driver hardware |
| Wi-Fi / Bluetooth | Built in | Requires additional wireless solution |
| Development | Easier and faster for the team | More complex |
| Embedded-systems learning | Good with custom electronics | Excellent |
| Power | Higher | Lower |
| Project risk | Lower | Higher |
| Current decision | **CHOSEN** | Not selected as primary controller |

## 5. Vision Approach — Decision Record

Ball detection began as classical computer vision: HSV white-thresholding,
morphological cleanup, and contour gates on area, circularity, and fill ratio.
Field testing showed two failure modes that are inherent to that approach rather
than a tuning mistake:

- **Shade and uneven light.** A fixed brightness threshold drops any ball in
  shadow. Lowering it to catch shaded balls admits bright grass and concrete
  instead. No single setting survives a green carrying both sun and shade.
- **Distance.** Calibration derives a minimum blob area from the sample balls, so
  balls beyond that range are discarded before shape checks run.

| Category | Classical CV | YOLO11n |
|---|---|---|
| Robustness to lighting | Poor; fixed thresholds | Good; learns shape and context |
| Distant/small balls | Poor; hard area floor | Moderate; improves with input size |
| Compute cost on RP5 | Very low, 30+ FPS | A few FPS on CPU |
| Dependencies | OpenCV only | Adds ultralytics and torch |
| Tuning effort | Re-calibrate per lighting change | Train once on a labelled dataset |
| Current decision | **Retained as fallback** | **CHOSEN as primary** |

**Decision:** adopt an Ultralytics YOLO11n detector as the primary vision path
and keep the classical detector as a selectable fallback. Both sit behind one
interface, so the autonomous state machine is unaffected by the choice and the
two can be compared on real footage.

Keeping the classical path is a risk control, not indecision. If neural inference
proves too slow on the Raspberry Pi 5, the robot still has a working detector that
needs nothing beyond OpenCV.

Rollout is staged: zero-shot detection using stock COCO weights (which already
carry a `sports ball` class) validates the approach before any labelling effort;
a custom single-class model trained on our own green follows. On the robot, the
model runs on a **Raspberry Pi AI HAT+ 2 (Hailo-10H NPU)**: it is compiled to a
Hailo HEF on an x86 Linux dev machine and loaded through the same YOLO backend,
so neither the detector interface nor the state machine changes. A CPU/NCNN
export remains the fallback if the HAT is absent, and classical CV remains the
fallback beneath that.

**Camera (decided):** Arducam-sold Raspberry Pi Camera Module 3 Wide: Sony
IMX708, 12 MP, 120° diagonal field of view, autofocus, on the Pi 5's CSI port
with a 15 cm 15-to-22-pin FFC cable. The wide lens lets one scan cover more of the
green, and autofocus holds focus from near the robot out to range. The cost is
resolution per degree: in a 1280 px frame a ball 3 m away is only about 10 px
across, and less toward the edges, where the lens distorts. On the Pi 5 the camera
is read through libcamera/Picamera2 rather than OpenCV's capture API. Only the
capture code changed, not the detector interface.

Still open: the camera's mounting height and tilt, which together with the lens
set the practical detection distance. The 15 cm cable also limits how far the
camera can sit from the Pi. The dataset must be captured with this camera at the
robot's real height once that is fixed.

## 6. Proposed Autonomous Operating Cycle

1. **Idle / Charge:** Robot remains at the base station between runs and recharges its battery.
2. **Launch:** Robot leaves the docking station on a scheduled cycle, approximately every 30 minutes.
3. **Scan:** Camera and software scan the putting/chipping area for golf balls.
4. **Navigate:** The robot plans movement toward detected balls while controlling its drivetrain.
5. **Collect:** The collection mechanism captures balls and stores them on the robot.
6. **Optional Clean:** If the cleaning subsystem is implemented, collected balls can be cleaned before return.
7. **Return:** Robot navigates back to the base station.
8. **Dock:** Robot aligns with the charging station and connects to the charging system.
9. **Unload / Stack:** Collected balls are transferred to the base station and organized by the golf-ball stacker.
10. **Repeat:** Robot resumes charging/idle operation until the next scheduled collection cycle.

**Prototype status (scan → navigate):** until the final motors arrive, the loop runs on a
VEX V5 prototype (4-motor tank drive, no intake) driven from the Pi over USB serial. It
uses reactive visual servoing: aim at the ball lowest in the camera frame (closest, for a
fixed, downward-tilted camera), turn to face it, then drive at it and stop at the bottom of
the frame. There is no map or localization yet, so return-to-base is not covered. See
`navigation/`.

## 7. Preliminary Hardware Bill of Materials

| Component / subsystem | Quantity / notes | Preliminary budget |
|---|---|---:|
| Raspberry Pi 5 | 1 central computer | $80–$100 |
| Camera | 1; Raspberry Pi Camera Module 3 Wide (IMX708, 120°, autofocus), CSI, 15 cm FFC | Selected |
| Raspberry Pi AI HAT+ 2 | 1; Hailo-10H NPU (40 TOPS) for on-robot YOLO inference | Acquired |
| MicroSD / storage | 1 | $10–$20 |
| Power regulation | Logic rail + appropriate converters | $15–$30 |
| DC geared drive motors | 4 planned | $40–$100 |
| Motor drivers | At least 4 channels/drivers | $30–$80 |
| Wheels | 4 | $30–$50 |
| Rechargeable battery | Robot battery pack | $50–$100 |
| Battery management / protection | Sized to battery | $20–$40 |
| Chassis / frame | Custom or purchased | $40–$75 |
| Ball collection mechanism | Custom prototype | $30–$75 |
| Cleaning mechanism | Optional | $20–$60 |
| Sensors / navigation hardware | To be determined | $25–$75 |
| Wiring / connectors / misc. | Prototype allowance | $30–$50 |
| Base charging dock | Contacts + charging hardware | $50–$100 |
| Ball stacking mechanism | Custom base-station subsystem | $30–$100 |

> **Important:** These are planning estimates, not final quotations. Motors, motor drivers, battery capacity, charging hardware, and the stacking mechanism should be selected after the robot's weight, target speed, traction requirements, and operating time are established.

## 8. Power-System Direction

- **Robot battery:** Use a rechargeable battery sized for the four-motor drivetrain and RP5 electronics.
- **Separate rails:** Keep the motor power path separate from the regulated computer/sensor power path to reduce electrical noise and prevent motor loads from disrupting the computer.
- **Charging dock:** The base station should recharge the robot automatically when it returns.
- **24-hour requirement:** Continuous 24-hour operation is not required. The robot instead operates in periodic collection cycles, substantially reducing the energy requirement.
- **Solar:** Solar can be investigated as a supplemental source for the base station, but it should not be assumed to provide all required charging power until an energy budget is calculated.

## 9. Electronics / Embedded-System Plan

The RP5 is the high-level computer. A custom KiCad PCB can provide connectors, power distribution, motor-driver interfaces, sensor connections, charging/docking interfaces, and other supporting electronics. This preserves a meaningful embedded-systems and PCB-design component without requiring the entire project to be built around a more difficult embedded processor.

The exact motor drivers, motors, battery chemistry, charging architecture, sensors, and docking contacts remain to be selected (the camera is chosen; see section 5). These choices should be based on measured or estimated robot mass, motor current, wheel size, terrain, desired speed, battery capacity, and charging time.

## 10. Project Priorities

1. Reliable four-wheel drivetrain and basic manual control.
2. Reliable ball collection mechanism.
3. Reliable return-to-base and docking concept.
4. RP5 camera pipeline and basic ball detection.
5. Autonomous navigation and collection behavior.
6. Automatic unloading into the base station.
7. Golf-ball stacking mechanism.
8. Optional ball-cleaning mechanism.
9. Solar charging investigation and optimization.

## 11. Team / Project Context

The capstone team consists of four members: Will (team leader), Micah (computer/software), Tiffany (research/documentation/people skills), and Isaac (hardware). Micah has begun learning KiCad for the project's electronics work. The original project direction was changed from earlier concepts, including HALO and a rocket avionics/telemetry system, to the autonomous golf ball picker.

## 12. Decisions Still Needed

- Exact Raspberry Pi 5 model / RAM configuration.
- Camera mounting height, tilt, and position relative to the Pi (15 cm cable); the camera model is settled (see section 5).
- Motor voltage, torque, RPM, and current requirements.
- Four motor-driver models and whether each driver handles one or multiple motors.
- Battery voltage, capacity, chemistry, and connector.
- Charging voltage/current and automatic docking method.
- Navigation sensors and/or localization method.
- Ball collection geometry and storage capacity.
- Base-station unloading interface.
- Golf-ball stacking mechanism.
- Whether the cleaning mechanism is included in the first prototype.
- Whether solar charging is practical after an energy budget is calculated.
- On-robot YOLO throughput on the AI HAT+ 2 (Hailo-10H), measured on real footage; the accelerator question itself is settled (see section 5).

---

**Document status:** Preliminary system concept. Component prices and specifications are planning estimates and should be verified before purchasing or committing the PCB design.
