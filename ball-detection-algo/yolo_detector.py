"""YOLO detection backend (Ultralytics YOLO11).

WHY THIS IS A CLASS AND NOT A PURE FUNCTION
    The classical detector in detector.py is a pure function because its entire
    "model" is a dozen numbers in a dataclass, cheap to pass per frame. A neural
    net is megabytes of weights that must be loaded ONCE and reused, so this
    backend is a callable object holding the loaded model. It exposes the same
    call signature as the classical adapter -- detector(frame) -> list[Detection]
    -- so run_webcam.py, test_image.py, and the future state machine cannot tell
    the two apart.

WHY THE IMPORT IS DEFERRED
    'ultralytics' pulls in torch: hundreds of megabytes, and seconds to import.
    It is an OPTIONAL dependency (requirements-yolo.txt). Importing this module
    must never require it, so the import lives inside __init__. That is what
    keeps --backend classical working on a bare opencv + numpy install, which is
    the fallback if inference turns out too slow on the Raspberry Pi 5.

MODEL FORMATS
    YOLO() loads .pt weights, an exported .onnx, or an exported NCNN directory
    through the identical call, so deploying to the RP5 CPU changes the --model
    path and nothing in this file.

    A Hailo export directory (a .hef plus metadata.yaml, built by
    export_hailo.py) is the one exception: it is run by hailo_runtime.py, not by
    YOLO(), because Ultralytics' Hailo backend uses a HailoRT API the Hailo-10H
    does not implement (see that module). It is still selected purely by
    --model, needs no ultralytics/torch, and returns the same detections.

    A Hailo HEF is compiled for one fixed input size, so --imgsz is ignored and
    self.imgsz reports the compiled size instead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from detector import Detection

# COCO class 32 is "sports ball". Stock YOLO weights already find golf balls
# under that label, which is what lets us evaluate the ML path before labelling
# a single image (milestone 1). A model fine-tuned on our own golf balls is
# single-class and has no class 32, so the filter must then be disabled with
# --coco-class -1.
COCO_SPORTS_BALL = 32

DEFAULT_MODEL = "yolo11n.pt"

_MISSING_DEP_HELP = (
    "The YOLO backend requires the 'ultralytics' package, which is an optional "
    "dependency of this project.\n"
    "    pip install -r requirements-yolo.txt\n"
    "The classical backend (--backend classical) needs nothing beyond "
    "requirements.txt."
)


def _find_hef(model_path: str) -> Path | None:
    """Return the .hef inside `model_path` if it is a Hailo export dir, else None."""
    path = Path(model_path)
    return next(path.glob("*.hef"), None) if path.is_dir() else None


def _hailo_metadata(hef: Path) -> dict:
    """Read the Ultralytics metadata.yaml that export_hailo.py writes beside the HEF."""
    import yaml  # PyYAML: an ultralytics dependency, and python3-yaml on Pi OS

    return yaml.safe_load((hef.parent / "metadata.yaml").read_text()) or {}


class YoloDetector:
    """Callable YOLO backend: `detector(frame_bgr) -> list[Detection]`."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        class_filter: int | None = COCO_SPORTS_BALL,
        label: str = "ball",
    ) -> None:
        self.model_path = model_path
        hef = _find_hef(model_path)
        self.hailo = hef is not None
        if self.hailo:
            from hailo_runtime import HailoModel

            meta = _hailo_metadata(hef)
            self.model = HailoModel(hef)
            if self.model.input_w != imgsz:
                print(f"Hailo HEF is compiled at imgsz={self.model.input_w}; "
                      f"ignoring --imgsz {imgsz} (re-export to change it).")
                imgsz = self.model.input_w
            names = meta.get("names") or {}
        else:
            try:
                from ultralytics import YOLO
            except ImportError as exc:  # optional dependency; explain, don't traceback
                raise ImportError(_MISSING_DEP_HELP) from exc

            self.model = YOLO(model_path)
            names = getattr(self.model, "names", None) or {}
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.label = label

        # Only apply the class filter if the loaded model actually HAS that class.
        # Our own fine-tuned single-class model would otherwise match nothing and
        # silently report zero balls, which is a miserable bug to chase.
        self.class_filter = [class_filter] if class_filter in names else None
        self.class_names = names

    def __call__(self, frame_bgr: np.ndarray) -> list[Detection]:
        frame_h, frame_w = frame_bgr.shape[:2]
        detections: list[Detection] = []
        for x1, y1, x2, y2, score, cls in self._predict(frame_bgr):
            # Both paths already map boxes back to original-frame coordinates
            # (undoing the letterbox), so we only clamp against off-by-one
            # overruns at the frame edges.
            x = max(0, int(round(x1)))
            y = max(0, int(round(y1)))
            w = min(frame_w, int(round(x2))) - x
            h = min(frame_h, int(round(y2))) - y
            if w <= 0 or h <= 0:
                continue
            detections.append(
                Detection(x=x, y=y, w=w, h=h, confidence=float(score),
                          label=self._label_for(int(cls)))
            )
        return detections

    def _predict(self, frame_bgr: np.ndarray) -> list[tuple]:
        """Raw (x1, y1, x2, y2, score, cls) rows in original-frame pixels."""
        if self.hailo:
            # NMS already ran on the chip with the floors baked in at export;
            # --conf and the class filter can only narrow that further.
            return [r for r in self.model(frame_bgr)
                    if r[4] >= self.conf
                    and (self.class_filter is None or r[5] in self.class_filter)]

        results = self.model.predict(
            frame_bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            classes=self.class_filter,
            verbose=False,
        )
        boxes = results[0].boxes if results else None
        if boxes is None or len(boxes) == 0:
            return []
        return [(*xyxy, score, int(cls)) for xyxy, score, cls in zip(
            boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist())]

    def _label_for(self, class_id: int) -> str:
        """Name a detection honestly.

        When the class filter is on we asked for one specific class and are
        treating it as a golf ball, so `label` (default "ball") is what we mean.
        With filtering off, report the model's own class name instead -- that way
        running stock COCO weights unfiltered says "bus" and "person" rather than
        claiming everything on screen is a ball.
        """
        if self.class_filter is not None:
            return self.label
        return self.class_names.get(class_id, self.label)

    def __repr__(self) -> str:
        return (
            f"YoloDetector(model={self.model_path!r}, imgsz={self.imgsz}, "
            f"conf={self.conf}, iou={self.iou}, class_filter={self.class_filter}, "
            f"hailo={self.hailo})"
        )
