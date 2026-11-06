# navigation — drive the VEX prototype to golf balls

Closes the loop between the ball detector ([`../ball-detection-algo/`](../ball-detection-algo/)) and the VEX V5 motors ([`../rpi-vex-protocol/`](../rpi-vex-protocol/)). It finds a ball, turns to face it, drives up to it and stops. Once the ball is picked up, it looks for the next one. This is the scan → navigate step of the operating cycle, built for the prototype: a 4-motor tank drive with no intake and a forward-facing, down-tilted Pi Camera Module 3 Wide.

| File | Role |
|---|---|
| `controller.py` | **Pure logic.** `ChaseController.update(detections, w, h, now)` returns left/right wheel %. It does no I/O and takes the time as an argument, so it is fully unit-tested. Every knob is in `ChaseConfig`. |
| `drive.py` | `TankDrive` turns (left %, right %) into VEX `SPIN` commands. `DryRunDrive` prints the commands instead. Also a CLI for checking motor directions. |
| `chase.py` | The runnable loop: camera → detector (either backend) → controller → drive. |

## Behaviour

The camera is fixed and tilted at the ground, so **the lower a ball's bottom edge sits in the frame, the closer it is**. The controller always chases the ball lowest in the frame, and it keeps chasing that ball while it stays in view.

| State | When | Does |
|---|---|---|
| `SEARCH` | no ball in view for `lost_timeout_s` | turns in place toward where a ball was last seen, in pulses (`search_turn_s` on, `search_pause_s` still) so the detector gets sharp frames |
| `ALIGN` | ball off-center by more than `align_tol` | turns in place (`kp_turn` × error, at least `min_turn_pct`) until within `approach_tol` |
| `APPROACH` | ball roughly ahead | drives forward with steering, and slows from `approach_pct` to `min_approach_pct` once the ball passes `slow_frac` |
| `ARRIVED` | ball bottom ≥ `arrive_frac`, or lost while ≥ `blind_frac` (it slid under the camera) | stops and holds until no close ball has been seen for `cooldown_s`, then goes back to `SEARCH` |

`max_pct` (default 30) caps every wheel command. Both wheels scale together, so the cap never changes the turn rate.

## Running

Run from `ball-detection-algo/` so relative paths (`params.json`, `yolo11n.pt`, `yolo11n_hailo_model/`) resolve the same way they do for `run_webcam.py`. The detector flags are identical to `run_webcam.py`'s.

```bash
cd ball-detection-algo

# 1. Laptop, no Brain: watch the overlay and the printed wheel commands
.venv/bin/python ../navigation/chase.py --camera 1 --dry-run

# 2. Pi + Brain, WHEELS OFF THE GROUND: check every wheel rolls forward, then spins right
.venv/bin/python ../navigation/drive.py forward 30 --seconds 1
.venv/bin/python ../navigation/drive.py turn 30 --seconds 1
#    wrong direction or motor? fix it with the port flags (negative = reversed):
.venv/bin/python ../navigation/drive.py forward 30 --left 1,2 --right -3,-4

# 3. On the floor, slow first
.venv/bin/python ../navigation/chase.py --camera csi --params params.json --max-pct 20
.venv/bin/python ../navigation/chase.py --camera csi --backend yolo --model yolo11n_hailo_model --no-display
```

Every `ChaseConfig` field is a flag (`--kp-turn`, `--arrive-frac`, ...; see `--help`). The overlay draws the `blind_frac` line in orange and the `arrive_frac` line in red. Tune those two lines to the real camera tilt: `arrive_frac` should be where a ball sits when it is right at the robot's front.

Talking to the Brain needs pyserial (`sudo apt install python3-serial`; the Pi venv sees it through `--system-site-packages`). `--dry-run` needs nothing beyond the detector's requirements.

**Safety:** the motors stop on any exit (`q`, Ctrl-C, or a crash), on a dropped frame, and whenever one camera + detect cycle takes longer than `--stale-s` (0.5 s), because a frame that old is too stale to steer by. The Brain's 500 ms watchdog still covers an unplugged or frozen Pi.

## Tests

```bash
cd navigation && ../ball-detection-algo/.venv/bin/python -m pytest tests
```

No camera, Brain or pyserial needed.
