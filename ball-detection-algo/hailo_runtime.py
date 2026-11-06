"""Run a Hailo HEF on the AI HAT+ 2 (Hailo-10H) through HailoRT directly.

WHY NOT LET ULTRALYTICS DO IT
    Ultralytics' Hailo backend drives HailoRT through the legacy
    configure/activate/InferVStreams API. On the Hailo-10H, HailoRT 5.x only
    implements the newer InferModel API (`VDevice.create_infer_model`), so the
    legacy path fails with "HAILO_NOT_IMPLEMENTED" (the same reason
    `hailortcli run` is rejected there in favor of `hailortcli run2`). This module
    uses InferModel instead, and as a side effect the Pi no longer needs torch
    for the NPU path.

WHAT THE HEF GIVES BACK
    export_hailo.py compiles NMS onto the chip, so the single output is already
    per-class detections: for each class an array of rows
    [y_min, x_min, y_max, x_max, score], normalized to the letterboxed input.
    This module undoes the letterbox and returns original-frame pixels.

Only `hailo_platform` (from apt's hailo-h10-all) plus opencv/numpy are needed.
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np

_MISSING_DEP_HELP = (
    "Running a Hailo export needs HailoRT's Python bindings (hailo_platform).\n"
    "    sudo apt install hailo-h10-all\n"
    "and a venv created with --system-site-packages so it can see them. "
    "See README.md, 'Hailo AI HAT+ 2'."
)

# Ultralytics pads letterboxes with this gray; matching it keeps the INT8 model
# seeing the same borders it was calibrated on.
_PAD_VALUE = 114
_TIMEOUT_MS = 10_000


class HailoModel:
    """Callable HEF runner: `model(frame_bgr) -> [(x1, y1, x2, y2, score, cls)]`."""

    def __init__(self, hef_path: str | Path) -> None:
        try:
            from hailo_platform import FormatType, HailoSchedulingAlgorithm, VDevice
        except ImportError as exc:
            raise ImportError(_MISSING_DEP_HELP) from exc

        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

        with ExitStack() as stack:
            vdevice = stack.enter_context(VDevice(params))
            infer_model = vdevice.create_infer_model(str(hef_path))
            infer_model.set_batch_size(1)
            infer_model.input().set_format_type(FormatType.UINT8)
            infer_model.output().set_format_type(FormatType.FLOAT32)
            self._configured = stack.enter_context(infer_model.configure())
            self._bindings = self._configured.create_bindings(
                output_buffers={
                    infer_model.output().name:
                        np.empty(infer_model.output().shape, dtype=np.float32)
                }
            )
            self.input_h, self.input_w = infer_model.input().shape[:2]
            self._stack = stack.pop_all()

    def close(self) -> None:
        """Release the NPU. Safe to call twice."""
        if stack := getattr(self, "_stack", None):
            self._stack = None
            stack.close()

    __del__ = close

    def __call__(self, frame_bgr: np.ndarray) -> list[tuple]:
        image, ratio, pad_x, pad_y = self._letterbox(frame_bgr)
        self._bindings.input().set_buffer(image)
        self._configured.run([self._bindings], _TIMEOUT_MS)
        per_class = self._bindings.output().get_buffer()
        # Some HailoRT versions keep a batch dimension around the class list.
        if len(per_class) == 1 and isinstance(per_class[0], (list, tuple)):
            per_class = per_class[0]

        scale = np.array([self.input_h, self.input_w, self.input_h, self.input_w],
                         dtype=np.float32)
        rows = []
        for cls, dets in enumerate(per_class):
            for y1, x1, y2, x2, score in np.asarray(dets, dtype=np.float32).reshape(-1, 5):
                y1, x1, y2, x2 = np.array([y1, x1, y2, x2]) * scale
                rows.append(((x1 - pad_x) / ratio, (y1 - pad_y) / ratio,
                             (x2 - pad_x) / ratio, (y2 - pad_y) / ratio,
                             float(score), cls))
        return rows

    def _letterbox(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        """Resize keeping aspect ratio, pad to the HEF's input, convert to RGB."""
        h, w = frame_bgr.shape[:2]
        ratio = min(self.input_h / h, self.input_w / w)
        new_w, new_h = round(w * ratio), round(h * ratio)
        pad_x = (self.input_w - new_w) // 2
        pad_y = (self.input_h - new_h) // 2

        canvas = np.full((self.input_h, self.input_w, 3), _PAD_VALUE, dtype=np.uint8)
        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        return np.ascontiguousarray(canvas[..., ::-1]), ratio, pad_x, pad_y
