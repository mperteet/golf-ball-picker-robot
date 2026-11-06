"""Turn ball detections into tank-drive wheel speeds ("chase the nearest ball").

Pure logic: no camera, no motors, no clock. `ChaseController.update()` takes the
detections from one frame plus the current time and returns the left/right wheel
percentages to send, so the whole behaviour is unit-testable on fake data.
chase.py does the I/O around it.

Assumes a camera fixed to the robot, facing forward and tilted down at the
ground. Under that assumption a ball's *bottom edge* in the image is a distance
proxy: the lower in the frame, the closer to the robot. `near` below is that
bottom edge as a fraction of frame height (0 = top, 1 = bottom).

States:
  SEARCH    no ball in view: turn in place toward where one was last seen, in
            short pulses with pauses so the detector gets sharp frames.
  ALIGN     ball well off-center: turn in place to face it.
  APPROACH  ball roughly ahead: drive at it, steering to keep it centered and
            slowing as it gets close.
  ARRIVED   ball reached the bottom of the frame (or slid out of view under the
            camera while close). Stop. The prototype has no intake, so we hold
            until the ball has been out of view for `cooldown_s` (i.e. someone
            picked it up), then search for the next one.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import asdict, dataclass, fields
from enum import Enum

# Detection lives in the detector module; see conftest.py / chase.py for why the
# ball-detection-algo directory is put on sys.path rather than imported as a package.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ball-detection-algo"))
from detector import Detection  # noqa: E402


class State(str, Enum):
    SEARCH = "SEARCH"
    ALIGN = "ALIGN"
    APPROACH = "APPROACH"
    ARRIVED = "ARRIVED"


@dataclass
class ChaseConfig:
    """Every tuning knob for the chase behaviour, in one place.

    Speeds are motor percent (-100..100). Errors are the ball's horizontal offset
    from frame center as a fraction of half the frame width (-1 left .. +1 right).
    """

    max_pct: float = 30.0          # hard cap on any wheel command

    # --- Turning (ALIGN, and steering during APPROACH)
    kp_turn: float = 40.0          # wheel % per unit of horizontal error
    min_turn_pct: float = 12.0     # in-place turns below this stall on friction
    align_tol: float = 0.25        # |error| above this -> stop and turn in place
    approach_tol: float = 0.12     # |error| below this -> start driving (hysteresis)

    # --- Driving at the ball (APPROACH)
    approach_pct: float = 30.0     # forward speed while the ball is far
    min_approach_pct: float = 12.0 # forward speed just before arriving
    slow_frac: float = 0.60        # start slowing once the ball's bottom passes this
    arrive_frac: float = 0.90      # ball bottom at/below this -> ARRIVED
    blind_frac: float = 0.75       # ball lost while at/below this -> it went under the camera

    # --- Losing and finding balls
    lost_timeout_s: float = 0.5    # keep the last command this long through dropouts
    match_frac: float = 0.15       # stick with the previous target if a ball is within
                                   # this fraction of frame width of where it was
    search_pct: float = 20.0       # in-place turn speed while searching
    search_turn_s: float = 0.3     # search pulse: turn this long...
    search_pause_s: float = 0.5    # ...then hold still this long
    cooldown_s: float = 2.0        # ARRIVED: ball must be out of view this long

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ChaseConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class DriveCommand:
    """What to do this frame, plus enough context to draw/log it."""

    left_pct: float
    right_pct: float
    state: State
    target: Detection | None = None
    error: float | None = None     # target's horizontal offset, -1..1
    near: float | None = None      # target's bottom edge, 0..1 of frame height


def _clamp(v: float, limit: float) -> float:
    return max(-limit, min(limit, v))


def _limit_pair(left: float, right: float, limit: float) -> tuple[float, float]:
    """Scale both wheels down together so neither exceeds `limit`.

    Clamping each wheel separately would change their ratio, i.e. the turn rate.
    """
    biggest = max(abs(left), abs(right))
    if biggest > limit > 0:
        k = limit / biggest
        return left * k, right * k
    return left, right


class ChaseController:
    def __init__(self, config: ChaseConfig | None = None) -> None:
        self.cfg = config or ChaseConfig()
        self.state = State.SEARCH
        self._state_since: float | None = None
        self._last_seen: float | None = None       # time a target was last in view
        self._last_center: tuple[float, float] | None = None
        self._last_error = 0.0
        self._last_near = 0.0
        self._last_cmd = (0.0, 0.0)

    # --- helpers -------------------------------------------------------------
    def _enter(self, state: State, now: float) -> None:
        if state != self.state:
            self.state = state
            self._state_since = now

    def _pick_target(self, detections: list[Detection], frame_w: int, frame_h: int,
                     now: float) -> Detection | None:
        if not detections:
            return None
        # Stick with the ball we were already chasing if it is still in view, so
        # two balls at similar distances don't make the robot flip between them.
        recent = self._last_seen is not None and now - self._last_seen <= self.cfg.lost_timeout_s
        if recent and self._last_center is not None:
            lx, ly = self._last_center
            nearest = min(detections, key=lambda d: math.dist(d.center, (lx, ly)))
            if math.dist(nearest.center, (lx, ly)) <= self.cfg.match_frac * frame_w:
                return nearest
        # Otherwise the closest ball: lowest bottom edge in the image.
        return max(detections, key=lambda d: d.y + d.h)

    def _cmd(self, left: float, right: float, **ctx) -> DriveCommand:
        left, right = _limit_pair(left, right, self.cfg.max_pct)
        self._last_cmd = (left, right)
        return DriveCommand(left, right, self.state, **ctx)

    def _turn_in_place(self, error: float) -> tuple[float, float]:
        turn = self.cfg.kp_turn * error
        if abs(turn) < self.cfg.min_turn_pct:
            turn = math.copysign(self.cfg.min_turn_pct, error or 1.0)
        turn = _clamp(turn, self.cfg.max_pct)
        return turn, -turn  # positive error = ball to the right = turn right

    def _approach(self, error: float, near: float) -> tuple[float, float]:
        c = self.cfg
        # Forward speed ramps from approach_pct at slow_frac down to
        # min_approach_pct at arrive_frac.
        span = max(c.arrive_frac - c.slow_frac, 1e-6)
        t = min(max((near - c.slow_frac) / span, 0.0), 1.0)
        fwd = c.approach_pct + (c.min_approach_pct - c.approach_pct) * t
        turn = c.kp_turn * error
        return fwd + turn, fwd - turn

    # --- main entry ----------------------------------------------------------
    def update(self, detections: list[Detection], frame_w: int, frame_h: int,
               now: float) -> DriveCommand:
        c = self.cfg
        if self._state_since is None:
            self._state_since = now

        target = self._pick_target(detections, frame_w, frame_h, now)
        error = near = None
        if target is not None:
            cx, _ = target.center
            error = _clamp((cx - frame_w / 2) / (frame_w / 2), 1.0)
            near = (target.y + target.h) / frame_h
            self._last_seen = now
            self._last_center = target.center
            self._last_error = error
            self._last_near = near
        ctx = {"target": target, "error": error, "near": near}

        # ARRIVED: hold still until no *close* ball has been seen for cooldown_s.
        # Far balls are ignored here -- we wait for this one to be removed first.
        if self.state == State.ARRIVED:
            if target is not None and near >= c.blind_frac:
                self._state_since = now  # still there; restart the cooldown
            elif now - self._state_since >= c.cooldown_s:
                self._enter(State.SEARCH, now)
                self._last_center = None  # don't re-lock onto the old position
            if self.state == State.ARRIVED:
                return self._cmd(0.0, 0.0, **ctx)

        if target is None:
            lost_for = None if self._last_seen is None else now - self._last_seen
            if self.state in (State.ALIGN, State.APPROACH):
                if lost_for is not None and lost_for <= c.lost_timeout_s:
                    # Brief dropout: keep doing what we were doing. Near the
                    # bottom of the frame this is also a short blind push onto
                    # a ball that has slid under the camera.
                    return self._cmd(*self._last_cmd, **ctx)
                if self._last_near >= c.blind_frac:
                    self._enter(State.ARRIVED, now)
                    return self._cmd(0.0, 0.0, **ctx)
            self._enter(State.SEARCH, now)
            phase = (now - self._state_since) % (c.search_turn_s + c.search_pause_s)
            if phase >= c.search_turn_s:
                return self._cmd(0.0, 0.0, **ctx)
            direction = 1.0 if self._last_error >= 0 else -1.0
            return self._cmd(direction * c.search_pct, -direction * c.search_pct, **ctx)

        if near >= c.arrive_frac:
            self._enter(State.ARRIVED, now)
            return self._cmd(0.0, 0.0, **ctx)

        if self.state == State.APPROACH:
            if abs(error) > c.align_tol:
                self._enter(State.ALIGN, now)
        elif abs(error) < c.approach_tol or (self.state == State.SEARCH and abs(error) <= c.align_tol):
            self._enter(State.APPROACH, now)
        else:
            self._enter(State.ALIGN, now)

        if self.state == State.ALIGN:
            return self._cmd(*self._turn_in_place(error), **ctx)
        return self._cmd(*self._approach(error, near), **ctx)
