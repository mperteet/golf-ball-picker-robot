# vex:disable=repl
# ---------------------------------------------------------------------------- #
#   Module:       main.py  (runs ON the VEX V5 Brain, via VEXcode V5 Python)    #
#   Description:  Serial motor server. A Raspberry Pi plugged into the Brain's  #
#                 micro-USB port sends one text command per line; the Brain     #
#                 drives the motors and answers "OK ..." or "ERR ...".          #
# ---------------------------------------------------------------------------- #
#
# The "vex:disable=repl" line above MUST stay the first line: without it the
# MicroPython REPL owns the USB user port and eats the Pi's commands.
#
# Protocol (ASCII, newline-terminated, case-insensitive):
#   PING                          -> OK PONG
#   SPIN  <port> <pct>            -> OK          spin at -100..100 % velocity
#   STOP  <port|ALL> [COAST|BRAKE|HOLD]  -> OK
#   MOVE  <port> <deg> <pct>      -> OK          go to absolute position (no wait)
#   RESET <port>                  -> OK          zero the encoder
#   STATUS <port>                 -> OK <pos_deg> <vel_pct> <current_A> <temp_C> <installed>
#
# Safety watchdog: if no command arrives for WATCHDOG_MS, every motor is
# stopped. The Pi side sends PING heartbeats to keep motors alive.

from vex import *
import sys

WATCHDOG_MS = 500
GEARING = GearSetting.RATIO_18_1  # green cartridge; change to RATIO_36_1 / RATIO_6_1

brain = Brain()
motors = {}
last_cmd_ms = 0
watchdog_tripped = True  # start stopped until the Pi talks to us


def get_motor(port):
    port = int(port)
    if port < 1 or port > 21:
        raise ValueError("port must be 1-21")
    m = motors.get(port)
    if m is None:
        m = Motor(getattr(Ports, "PORT" + str(port)), GEARING, False)
        motors[port] = m
    return m


def stop_all(mode=BRAKE):
    for m in motors.values():
        m.stop(mode)


STOP_MODES = {"COAST": COAST, "BRAKE": BRAKE, "HOLD": HOLD}


def clamp_pct(v):
    v = float(v)
    return max(-100.0, min(100.0, v))


def handle(line):
    parts = line.strip().upper().split()
    if not parts:
        return None
    cmd, args = parts[0], parts[1:]

    if cmd == "PING":
        return "OK PONG"

    if cmd == "SPIN":
        m = get_motor(args[0])
        m.spin(FORWARD, clamp_pct(args[1]), PERCENT)
        return "OK"

    if cmd == "STOP":
        mode = STOP_MODES.get(args[1], BRAKE) if len(args) > 1 else BRAKE
        if args[0] == "ALL":
            stop_all(mode)
        else:
            get_motor(args[0]).stop(mode)
        return "OK"

    if cmd == "MOVE":
        m = get_motor(args[0])
        m.spin_to_position(float(args[1]), DEGREES, abs(clamp_pct(args[2])), PERCENT, False)
        return "OK"

    if cmd == "RESET":
        get_motor(args[0]).reset_position()
        return "OK"

    if cmd == "STATUS":
        m = get_motor(args[0])
        return "OK %.1f %.1f %.2f %.1f %d" % (
            m.position(DEGREES),
            m.velocity(PERCENT),
            m.current(CurrentUnits.AMP),
            m.temperature(TemperatureUnits.CELSIUS),
            1 if m.installed() else 0,
        )

    raise ValueError("unknown command " + cmd)


def reply(text):
    print(text)


def open_input():
    """The USB user port. /dev/serial1 reads raw bytes; stdin is the fallback."""
    try:
        return open("/dev/serial1", "rb")
    except Exception:
        return None


def read_line(stream):
    if stream is None:
        return sys.stdin.readline()
    buf = bytearray()
    while True:
        b = stream.read(1)
        if not b:
            continue
        if b == b"\n":
            return buf.decode()
        if b != b"\r":
            buf.extend(b)


def serial_thread():
    global last_cmd_ms, watchdog_tripped
    stream = open_input()
    while True:
        line = read_line(stream)
        last_cmd_ms = brain.timer.time(MSEC)
        watchdog_tripped = False
        try:
            out = handle(line)
        except (IndexError, ValueError) as e:
            out = "ERR " + str(e) if str(e) else "ERR bad arguments"
        except Exception as e:
            out = "ERR " + str(e)
        if out is not None:
            reply(out)


Thread(serial_thread)

brain.screen.print("Pi motor server ready")
while True:
    if not watchdog_tripped and brain.timer.time(MSEC) - last_cmd_ms > WATCHDOG_MS:
        stop_all()
        watchdog_tripped = True
    brain.screen.set_cursor(3, 1)
    brain.screen.clear_line(3)
    brain.screen.print("Link: " + ("LOST - motors stopped" if watchdog_tripped else "OK"))
    wait(50, MSEC)
