"""Tests for list_cameras.py's CSI check, against a fake Picamera2."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

import list_cameras


class _FakePicamera2:
    camera_info = [
        {"Model": "imx708_wide", "Num": 0, "Id": "/base/axi/pcie@120000/rp1/i2c@88000/imx708@1a"},
        {"Model": "uvcvideo", "Num": 1, "Id": "/base/axi/pcie@120000/rp1/usb@200000-1:1.0-046d:0825"},
    ]

    def __init__(self, port):
        self.port = port

    @classmethod
    def global_camera_info(cls):
        return cls.camera_info

    def create_video_configuration(self, main):
        return {"main": main}

    def configure(self, config):
        self.config = config

    def start(self):
        pass

    def set_controls(self, controls):
        pass

    def capture_array(self, stream):
        w, h = self.config["main"]["size"]
        return np.zeros((h, w, 3), np.uint8)

    def stop(self):
        pass

    def close(self):
        pass


@pytest.fixture
def fake_picamera2(monkeypatch):
    monkeypatch.setitem(sys.modules, "picamera2",
                        types.SimpleNamespace(Picamera2=_FakePicamera2))
    monkeypatch.setitem(sys.modules, "libcamera", None)  # no AF control; must be tolerated
    return _FakePicamera2


def test_find_csi_cameras_skips_usb(fake_picamera2):
    cameras, note = list_cameras.find_csi_cameras()
    assert note is None
    assert [c["Model"] for c in cameras] == ["imx708_wide"]


def test_find_csi_cameras_without_picamera2(monkeypatch):
    monkeypatch.setitem(sys.modules, "picamera2", None)  # makes the import fail
    cameras, note = list_cameras.find_csi_cameras()
    assert cameras == [] and "python3-picamera2" in note


@pytest.mark.parametrize("num, flag", [(0, "csi"), (1, "csi1")])
def test_csi_flag(num, flag):
    assert list_cameras.csi_flag(num) == flag


def test_list_only_reports_csi(fake_picamera2, monkeypatch, capsys):
    monkeypatch.setattr(list_cameras, "_open", lambda index, backend: None)  # no USB cams
    cameras, _ = list_cameras.find_csi_cameras()
    working_csi, working = list_cameras.list_only(2, cameras)
    assert (working_csi, working) == (["csi"], [])
    assert "csi (imx708_wide): OPEN, frame 1280x720" in capsys.readouterr().out
