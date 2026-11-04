#!/usr/bin/env python3
"""Control VEX V5 motors from a Raspberry Pi over the Brain's USB port.

Pair with brain/main.py running on the V5 Brain. Use as a library:

    from vex_motor import VexBrain
    with VexBrain() as brain:
        brain.spin(1, 50)          # port 1 at 50 %
        time.sleep(2)
        print(brain.status(1))
    # leaving the block stops every motor

or from the shell:

    python3 vex_motor.py list
    python3 vex_motor.py spin 1 50 --seconds 2
    python3 vex_motor.py move 1 360 40
    python3 vex_motor.py status 1

Needs pyserial (`sudo apt install python3-serial` or `pip install pyserial`)
and the user in the `dialout` group.
"""

import argparse
import threading
import time
from dataclasses import dataclass

import serial
from serial.tools import list_ports

VEX_VID = 0x2888  # VEX Robotics USB vendor id


@dataclass
class MotorStatus:
    position_deg: float
    velocity_pct: float
    current_a: float
    temperature_c: float
    installed: bool


class VexError(RuntimeError):
    pass


def find_vex_ports():
    return sorted((p for p in list_ports.comports() if p.vid == VEX_VID), key=lambda p: p.device)


def find_user_port():
    """The Brain exposes two serial ports: system (interface 0, used by VEXcode
    downloads) and user (interface 2, where print()/stdin go). We want the user one."""
    ports = find_vex_ports()
    if not ports:
        raise VexError("No VEX Brain found over USB. Is it plugged in and powered on?")
    for p in ports:
        desc = (p.description or "").lower()
        if "user" in desc or (p.location or "").endswith(".2"):
            return p.device
    # Fallback: the user port is normally the second one (/dev/ttyACM1).
    return ports[-1].device


class VexBrain:
    def __init__(self, port=None, timeout=0.5, heartbeat_s=0.2):
        self.port = port or find_user_port()
        self._ser = serial.Serial(self.port, 115200, timeout=timeout)
        self._lock = threading.Lock()
        self._alive = True
        time.sleep(0.1)
        self._ser.reset_input_buffer()
        self.ping()
        # Brain stops all motors if it hears nothing for 500 ms; keep it fed.
        self._hb = threading.Thread(target=self._heartbeat, args=(heartbeat_s,), daemon=True)
        self._hb.start()

    # --- transport ---------------------------------------------------------
    def command(self, line):
        with self._lock:
            self._ser.write((line + "\n").encode())
            deadline = time.monotonic() + self._ser.timeout * 2
            while time.monotonic() < deadline:
                resp = self._ser.readline().decode(errors="replace").strip()
                if resp.startswith("OK"):
                    return resp[2:].strip()
                if resp.startswith("ERR"):
                    raise VexError(f"{line!r}: {resp[3:].strip()}")
                # anything else is a stray print() from the brain; skip it
            raise VexError(f"No reply to {line!r} on {self.port}. Is brain/main.py running?")

    def _heartbeat(self, period):
        while self._alive:
            time.sleep(period)
            try:
                self.ping()
            except (VexError, serial.SerialException):
                pass

    # --- motor API ---------------------------------------------------------
    def ping(self):
        return self.command("PING") == "PONG"

    def spin(self, port, pct):
        """Spin continuously at pct (-100..100) until stop() or the watchdog."""
        self.command(f"SPIN {port} {pct:.1f}")

    def stop(self, port="ALL", mode="BRAKE"):
        """mode: COAST, BRAKE or HOLD."""
        self.command(f"STOP {port} {mode}")

    def move_to(self, port, degrees, pct=50):
        """Go to an absolute encoder position; returns immediately."""
        self.command(f"MOVE {port} {degrees:.1f} {pct:.1f}")

    def reset_position(self, port):
        self.command(f"RESET {port}")

    def status(self, port):
        pos, vel, cur, temp, inst = self.command(f"STATUS {port}").split()
        return MotorStatus(float(pos), float(vel), float(cur), float(temp), inst == "1")

    def close(self):
        self._alive = False
        try:
            self.stop("ALL")
        finally:
            self._ser.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", help="serial device (default: auto-detect Brain user port)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="show VEX serial ports")
    s = sub.add_parser("spin", help="spin a motor for a while")
    s.add_argument("motor", type=int)
    s.add_argument("pct", type=float)
    s.add_argument("--seconds", type=float, default=2.0)
    m = sub.add_parser("move", help="move to an absolute position and wait")
    m.add_argument("motor", type=int)
    m.add_argument("degrees", type=float)
    m.add_argument("pct", type=float, nargs="?", default=50)
    st = sub.add_parser("status", help="print motor telemetry")
    st.add_argument("motor", type=int)
    r = sub.add_parser("reset", help="zero a motor's encoder")
    r.add_argument("motor", type=int)
    sub.add_parser("stop", help="stop all motors")
    args = ap.parse_args()

    if args.cmd == "list":
        for p in find_vex_ports():
            print(p.device, p.location, p.description)
        return

    with VexBrain(args.port) as brain:
        print(f"Connected on {brain.port}")
        if args.cmd == "spin":
            brain.spin(args.motor, args.pct)
            time.sleep(args.seconds)
        elif args.cmd == "move":
            brain.move_to(args.motor, args.degrees, args.pct)
            time.sleep(0.2)
            while abs(brain.status(args.motor).velocity_pct) > 1:
                time.sleep(0.05)
        elif args.cmd == "reset":
            brain.reset_position(args.motor)
        elif args.cmd == "stop":
            pass  # close() stops everything
        if args.cmd in ("spin", "move", "status"):
            print(brain.status(args.motor))


if __name__ == "__main__":
    main()
