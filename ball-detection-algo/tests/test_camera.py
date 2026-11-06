"""Tests for camera source selection and the Picamera2 wrapper.

Picamera2 only exists on the Pi, so the CSI path is exercised against a fake
module that records how it was driven.
"""

from __future__ import annotations

import argparse
import sys
import types

import numpy as np
import pytest

import camera
from camera import PiCamera, add_camera_args, camera_source, open_camera


@pytest.mark.parametrize("value, expected", [
    ("0", 0), ("1", 1), ("csi", "csi0"), ("CSI", "csi0"), ("csi0", "csi0"), ("csi1", "csi1"),
])
def test_camera_source_parses(value, expected):
    assert camera_source(value) == expected


@pytest.mark.parametrize("value", ["", "usb", "csi2", "-1", "1.5"])
def test_camera_source_rejects(value):
    with pytest.raises(argparse.ArgumentTypeError):
        camera_source(value)


def test_add_camera_args_defaults():
    parser = argparse.ArgumentParser()
    add_camera_args(parser)
    args = parser.parse_args(["--camera", "csi"])
    assert (args.camera, args.width, args.height) == ("csi0", 1280, 720)


class _FakePicamera2:
    instances: list["_FakePicamera2"] = []

    def __init__(self, port):
        self.port = port
        self.calls: list[str] = []
        self.controls: dict = {}
        _FakePicamera2.instances.append(self)

    def create_video_configuration(self, main):
        return {"main": main}

    def configure(self, config):
        self.config = config

    def start(self):
        self.calls.append("start")

    def set_controls(self, controls):
        self.controls.update(controls)

    def capture_array(self, stream):
        w, h = self.config["main"]["size"]
        return np.zeros((h, w, 3), np.uint8)

    def stop(self):
        self.calls.append("stop")

    def close(self):
        self.calls.append("close")


@pytest.fixture
def fake_picamera2(monkeypatch):
    _FakePicamera2.instances.clear()
    monkeypatch.setitem(sys.modules, "picamera2",
                        types.SimpleNamespace(Picamera2=_FakePicamera2))
    af = types.SimpleNamespace(AfModeEnum=types.SimpleNamespace(Continuous="continuous"))
    monkeypatch.setitem(sys.modules, "libcamera", types.SimpleNamespace(controls=af))
    return _FakePicamera2


def test_open_camera_csi_uses_picamera2(fake_picamera2):
    cap = open_camera("csi1", 640, 360)
    assert isinstance(cap, PiCamera) and cap.isOpened()
    cam = fake_picamera2.instances[-1]
    assert cam.port == 1
    assert cam.config["main"] == {"size": (640, 360), "format": "RGB888"}
    assert cam.controls == {"AfMode": "continuous"}

    ok, frame = cap.read()
    assert ok and frame.shape == (360, 640, 3) and frame.dtype == np.uint8

    cap.release()
    cap.release()  # idempotent, like cv2.VideoCapture
    assert cam.calls == ["start", "stop", "close"]
    assert cap.read() == (False, None)


def test_csi_without_picamera2_explains_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "picamera2", None)  # makes the import fail
    with pytest.raises(RuntimeError, match="python3-picamera2"):
        open_camera("csi0")


def test_camera_module_imports_without_picamera2():
    """The laptop has no Picamera2; importing camera.py must not need it."""
    assert "picamera2" not in vars(camera)
