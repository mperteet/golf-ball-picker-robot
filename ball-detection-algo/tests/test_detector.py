"""Tests for the classical detector and the backend factory.

Everything here runs on synthetic frames with only opencv + numpy installed:
no camera, no window, no ML dependencies.
"""

from __future__ import annotations

import subprocess
import sys

import cv2
import numpy as np
import pytest

from backends import ClassicalBackend, make_detector
from detector import (
    Detection,
    DetectorParams,
    derive_params_from_rois,
    detect_golf_balls,
    draw_detections,
)

GRASS_BGR = (40, 120, 40)
BALL_BGR = (245, 245, 245)
BALL_RADIUS = 20
BALL_CENTERS = [(100, 100), (300, 250), (500, 400)]


def _green(width: int = 640, height: int = 480) -> np.ndarray:
    return np.full((height, width, 3), GRASS_BGR, np.uint8)


def _balls_frame() -> np.ndarray:
    """Three white balls on grass, plus a bright bar that must not count."""
    frame = _green()
    for center in BALL_CENTERS:
        cv2.circle(frame, center, BALL_RADIUS, BALL_BGR, -1)
    cv2.rectangle(frame, (400, 50), (600, 70), (250, 250, 250), -1)
    return frame


def _ball_rois() -> list[tuple[int, int, int, int]]:
    """Tight boxes around each ball, as a user would draw in calibrate.py."""
    d = 2 * BALL_RADIUS
    return [(cx - BALL_RADIUS, cy - BALL_RADIUS, d, d) for cx, cy in BALL_CENTERS]


def _centers(detections: list[Detection]) -> list[tuple[int, int]]:
    return sorted(d.center for d in detections)


def _assert_finds_balls(detections: list[Detection], tol: int = 3) -> None:
    assert len(detections) == len(BALL_CENTERS)
    for (x, y), (ex, ey) in zip(_centers(detections), sorted(BALL_CENTERS)):
        assert abs(x - ex) <= tol and abs(y - ey) <= tol


# --- detect_golf_balls ------------------------------------------------------


def test_finds_balls_and_rejects_non_round_blob():
    detections = detect_golf_balls(_balls_frame())
    _assert_finds_balls(detections)
    assert all(d.circularity is not None for d in detections)


def test_empty_green_has_no_detections():
    assert detect_golf_balls(_green()) == []


def test_downscale_returns_original_frame_coordinates():
    full = detect_golf_balls(_balls_frame(), DetectorParams())
    half = detect_golf_balls(_balls_frame(), DetectorParams(downscale=0.5))
    _assert_finds_balls(half)
    for a, b in zip(sorted(full, key=lambda d: d.center), sorted(half, key=lambda d: d.center)):
        assert abs(a.w - b.w) <= 3 and abs(a.h - b.h) <= 3


@pytest.mark.parametrize("downscale", [1.0, 0.5, 0.25])
def test_calibrated_params_work_at_any_downscale(downscale):
    """Area gates are in original-frame pixels, so downscale only buys speed.

    Regression: calibrate.py measures ball areas at full resolution. When the
    gates were compared against the downscaled contour instead, downscale=0.5
    shrank every ball to a quarter of its calibrated area and all of them fell
    under min_area.
    """
    frame = _balls_frame()
    params = derive_params_from_rois(frame, _ball_rois(), DetectorParams(downscale=downscale))
    assert params.downscale == downscale
    _assert_finds_balls(detect_golf_balls(frame, params))


def test_max_area_rejects_oversized_blob_at_downscale():
    frame = _green()
    cv2.circle(frame, (320, 240), 100, BALL_BGR, -1)  # ~31,400 px^2
    params = DetectorParams(max_area=20_000)
    assert detect_golf_balls(frame, params) == []
    assert detect_golf_balls(frame, DetectorParams(max_area=20_000, downscale=0.5)) == []


# --- derive_params_from_rois ------------------------------------------------


def test_calibration_color_gate_brackets_the_balls():
    params = derive_params_from_rois(_balls_frame(), _ball_rois())
    ball_hsv = cv2.cvtColor(np.uint8([[BALL_BGR]]), cv2.COLOR_BGR2HSV)[0, 0]
    grass_hsv = cv2.cvtColor(np.uint8([[GRASS_BGR]]), cv2.COLOR_BGR2HSV)[0, 0]
    assert ball_hsv[1] <= params.sat_max and ball_hsv[2] >= params.val_min
    assert grass_hsv[2] < params.val_min or grass_hsv[1] > params.sat_max


def test_calibration_area_gates_bracket_the_balls():
    params = derive_params_from_rois(_balls_frame(), _ball_rois())
    ball_area = np.pi * BALL_RADIUS ** 2
    assert params.min_area < ball_area < params.max_area


def test_calibration_keeps_base_shape_params():
    base = DetectorParams(min_circularity=0.7, morph_iters=2)
    params = derive_params_from_rois(_balls_frame(), _ball_rois(), base)
    assert params.min_circularity == 0.7 and params.morph_iters == 2


def test_calibration_requires_rois():
    with pytest.raises(ValueError):
        derive_params_from_rois(_balls_frame(), [])


def test_calibration_rejects_rois_outside_frame():
    with pytest.raises(ValueError):
        derive_params_from_rois(_balls_frame(), [(5000, 5000, 40, 40)])


# --- DetectorParams ---------------------------------------------------------


def test_params_json_round_trip(tmp_path):
    params = DetectorParams(sat_max=42, val_min=199, downscale=0.5, min_fill_ratio=0.7)
    path = tmp_path / "params.json"
    params.save(str(path))
    assert DetectorParams.load(str(path)) == params


def test_params_from_dict_ignores_unknown_keys():
    assert DetectorParams.from_dict({"sat_max": 10, "retired_knob": 3}).sat_max == 10


# --- drawing and backends ---------------------------------------------------


def test_draw_detections_does_not_modify_input():
    frame = _balls_frame()
    before = frame.copy()
    out = draw_detections(frame, detect_golf_balls(frame))
    assert np.array_equal(frame, before)
    assert not np.array_equal(out, before)


def test_classical_backend_matches_pure_function():
    params = DetectorParams(val_min=170)
    detector = make_detector("classical", params=params)
    assert isinstance(detector, ClassicalBackend)
    assert detector(_balls_frame()) == detect_golf_balls(_balls_frame(), params)


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        make_detector("nope")


def test_classical_path_imports_no_ml_dependencies():
    """detector.py and backends.py must stay runnable on a bare OpenCV install."""
    code = (
        "import sys, detector, backends; "
        "bad = [m for m in ('ultralytics', 'torch') if m in sys.modules]; "
        "print(', '.join(bad)); sys.exit(1 if bad else 0)"
    )
    root = __file__.rsplit("tests", 1)[0]
    result = subprocess.run([sys.executable, "-c", code], cwd=root,
                            capture_output=True, text=True)
    assert result.returncode == 0, f"classical path imported: {result.stdout}{result.stderr}"
