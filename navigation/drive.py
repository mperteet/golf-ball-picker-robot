#!/usr/bin/env python3
"""Tank drive on the VEX V5 prototype, over the Pi-to-Brain serial link.

`TankDrive` maps (left %, right %) onto the drive motors with one SPIN per
motor. Motor ports are given VEX-style: a negative port number means that
motor is mounted mirrored and spins reversed, e.g. left 1,2 and right -3,-4.

`DryRunDrive` has the same interface but only prints, so chase.py can be tested
on a laptop with no Brain attached (and without pyserial installed).

Before running anything autonomous, check the port signs with the wheels off
the ground -- "forward" must roll every wheel forward, "turn" must spin right:

    python drive.py forward 30 --seconds 1
    python drive.py turn 30 --seconds 1
    python drive.py forward 30 --left 1,2 --right -3,-4
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_VEX_PI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rpi-vex-protocol", "pi")


def parse_ports(text: str) -> list[int]:
    """argparse `type=`: '1,2' or '-3,-4' -> [1, 2] / [-3, -4]. Sign = reversed."""
    try:
        ports = [int(p) for p in text.replace(" ", "").split(",") if p]
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated ports like 1,2 or -3,-4, got {text!r}")
    if not ports or any(not 1 <= abs(p) <= 21 for p in ports):
        raise argparse.ArgumentTypeError(f"ports must be 1-21 (negative = reversed), got {text!r}")
    return ports


class TankDrive:
    """Left/right wheel percentages -> SPIN commands on a VexBrain."""

    def __init__(self, brain, left_ports: list[int], right_ports: list[int]) -> None:
        overlap = {abs(p) for p in left_ports} & {abs(p) for p in right_ports}
        if overlap:
            raise ValueError(f"port(s) {sorted(overlap)} assigned to both sides")
        self.brain = brain
        self.left_ports = list(left_ports)
        self.right_ports = list(right_ports)
        self._last: tuple[float, float] | None = None

    def set(self, left_pct: float, right_pct: float) -> None:
        cmd = (round(left_pct, 1), round(right_pct, 1))
        # Re-sending an unchanged speed is pointless: motors keep spinning, and
        # VexBrain's heartbeat already feeds the Brain's 500 ms watchdog.
        if cmd == self._last:
            return
        if cmd == (0.0, 0.0):
            self.stop()
            return
        for ports, pct in ((self.left_ports, cmd[0]), (self.right_ports, cmd[1])):
            for p in ports:
                self.brain.spin(abs(p), pct if p > 0 else -pct)
        self._last = cmd  # only after every motor accepted it

    def stop(self) -> None:
        self.brain.stop("ALL", "BRAKE")
        self._last = (0.0, 0.0)

    def close(self) -> None:
        try:
            self.stop()
        finally:
            self.brain.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class DryRunDrive:
    """Prints wheel commands instead of sending them. Same interface as TankDrive."""

    def __init__(self, left_ports: list[int] | None = None, right_ports: list[int] | None = None) -> None:
        self.left_ports, self.right_ports = left_ports, right_ports
        self._last: tuple[float, float] | None = None

    def set(self, left_pct: float, right_pct: float) -> None:
        cmd = (round(left_pct, 1), round(right_pct, 1))
        if cmd != self._last:
            print(f"[dry-run] L={cmd[0]:+6.1f}%  R={cmd[1]:+6.1f}%")
            self._last = cmd

    def stop(self) -> None:
        self.set(0.0, 0.0)

    def close(self) -> None:
        self.stop()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def add_drive_args(parser: argparse.ArgumentParser) -> None:
    """Drive flags shared by drive.py and chase.py."""
    group = parser.add_argument_group("drive")
    group.add_argument("--left", type=parse_ports, default=[1, 2], metavar="PORTS",
                       help="left drive motor ports, negative = reversed (default: 1,2)")
    group.add_argument("--right", type=parse_ports, default=[-3, -4], metavar="PORTS",
                       help="right drive motor ports, negative = reversed (default: -3,-4)")
    group.add_argument("--port", default=None,
                       help="Brain serial device (default: auto-detect the user port)")
    group.add_argument("--dry-run", action="store_true",
                       help="print wheel commands instead of talking to the Brain")


def open_drive(args: argparse.Namespace):
    """Build a TankDrive on a real Brain, or a DryRunDrive with --dry-run."""
    if args.dry_run:
        return DryRunDrive(args.left, args.right)
    # Lazy: only a real Brain needs pyserial, so --dry-run works on a bare laptop.
    sys.path.insert(0, _VEX_PI_DIR)
    try:
        from vex_motor import VexBrain, VexError
    except ImportError as exc:
        raise SystemExit(
            f"{exc}\nTalking to the Brain needs pyserial "
            "(`sudo apt install python3-serial` on the Pi, or `pip install pyserial`). "
            "Use --dry-run to test without one."
        ) from None
    try:
        brain = VexBrain(args.port)
    except VexError as exc:
        raise SystemExit(str(exc)) from None
    print(f"Connected to VEX Brain on {brain.port}")
    return TankDrive(brain, args.left, args.right)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=("forward", "turn"),
                    help="forward: both sides +pct; turn: spin right in place (left +pct, right -pct)")
    ap.add_argument("pct", type=float, help="speed in percent; negative reverses")
    ap.add_argument("--seconds", type=float, default=1.0)
    add_drive_args(ap)
    args = ap.parse_args()

    with open_drive(args) as drive:
        if args.action == "forward":
            drive.set(args.pct, args.pct)
        else:
            drive.set(args.pct, -args.pct)
        time.sleep(args.seconds)


if __name__ == "__main__":
    main()
