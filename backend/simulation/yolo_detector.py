"""
YOLO11 beacon detector.

Drop-in alternative to the Classical CV detector
(simulation/detector.py::BeaconDetector).

Interface contract (same as Classical):
    detect(frame) -> {"x": float, "y": float, "confidence": float,
                      "bbox": [x1, y1, x2, y2]} | None

- frame: BGR numpy array (same as VirtualCamera.render output, 1280x720x3).
- Returns None when nothing detected or model unavailable.
- Classical CV is unchanged; this module only adds a second detector.
- PAT, tracker, gimbal and physics are untouched; they only consume
  x/y/confidence (bbox is optional extra).

Model: backend/models/best.pt (YOLO11n trained on synthetic beacons).
If best.pt is missing but best.pt.zip exists (torch zip renamed),
it is reconstructed automatically.
If ultralytics/torch is missing, the detector reports unavailable
instead of crashing so Classical keeps working.
"""

import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Resolve default model path relative to this file:
# backend/simulation/yolo_detector.py -> backend/models/best.pt
_DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best.pt"
_DEFAULT_MODEL_ZIP = Path(__file__).resolve().parent.parent / "models" / "best.pt.zip"

AVAILABLE_DETECTORS = ("classical", "yolo")


def _reconstruct_pt_from_zip(zip_path: Path, pt_path: Path) -> bool:
    """Ensure best.pt exists, deriving from best.pt.zip if needed.

    Torch .pt files are zip archives whose members live under a prefix
    matching the filename stem (e.g. best.pt -> best/data.pkl).
    best.pt.zip already has that layout (best/...), so a byte copy is
    sufficient. If layouts differ, re-zip with the correct prefix.
    """
    try:
        import zipfile

        if pt_path.exists():
            return True
        if not zip_path.exists():
            return False
        expected_prefix = pt_path.stem + "/"
        with zipfile.ZipFile(zip_path, "r") as zin:
            names = zin.namelist()
            if f"{expected_prefix}data.pkl" in names or "data.pkl" in names and expected_prefix == "best/":
                # Covers: zip already has best/ prefix for best.pt target.
                # Also covers exact match. Byte copy preserves torch format.
                if any(n.startswith(expected_prefix) for n in names):
                    pt_path.write_bytes(zip_path.read_bytes())
                    logger.info(f"Copied {zip_path} -> {pt_path} (torch format preserved)")
                    return True
            # Fallback: re-zip contents under expected prefix
            import tempfile, os

            with tempfile.TemporaryDirectory() as tmpdir:
                zin.extractall(tmpdir)
                # Find source root: prefer folder matching expected stem,
                # else first dir, else tmpdir itself.
                candidate = os.path.join(tmpdir, pt_path.stem)
                if not os.path.isdir(candidate):
                    subdirs = [
                        os.path.join(tmpdir, d)
                        for d in os.listdir(tmpdir)
                        if os.path.isdir(os.path.join(tmpdir, d))
                    ]
                    src_root = subdirs[0] if len(subdirs) == 1 else tmpdir
                    src_prefix = "" if src_root == tmpdir else os.path.basename(src_root) + "/"
                else:
                    src_root = candidate
                    src_prefix = pt_path.stem + "/"
                with zipfile.ZipFile(pt_path, "w", compression=zipfile.ZIP_STORED) as zout:
                    for root, _, files in os.walk(src_root):
                        for f in files:
                            full = os.path.join(root, f)
                            rel = os.path.relpath(full, src_root)
                            # Strip old prefix, add expected one
                            if src_prefix and rel.startswith(src_prefix):
                                rel = rel[len(src_prefix):]
                            # rel is now prefix-free; add expected prefix
                            if rel.startswith(expected_prefix):
                                arc = rel
                            else:
                                arc = expected_prefix + rel.replace(os.sep, "/")
                            zout.write(full, arc)
        logger.info(f"Reconstructed {pt_path} from {zip_path}")
        return pt_path.exists()
    except Exception as e:
        logger.warning(f"Failed to reconstruct {pt_path} from {zip_path}: {e}")
        return False


def resolve_model_path(explicit: Optional[str] = None) -> Optional[Path]:
    """Return usable best.pt path, reconstructing from .zip if needed."""
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        return None
    if _DEFAULT_MODEL_PATH.exists():
        return _DEFAULT_MODEL_PATH
    # Try auto-reconstruction
    if _DEFAULT_MODEL_ZIP.exists():
        if _reconstruct_pt_from_zip(_DEFAULT_MODEL_ZIP, _DEFAULT_MODEL_PATH):
            return _DEFAULT_MODEL_PATH
    return None


class YOLO11Detector:
    """YOLO11n beacon detector with Classical-compatible interface."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf: float = 0.25,
        # Inference resolution. Profiling shows the torch forward pass is
        # ~90% of per-tick cost, so this dominates live cadence (118 ms at
        # 640 vs 76 ms at 416 steady-state on the reference box, ~1.5x).
        # 416 was chosen over 640 by measurement, not tuning: same-seed
        # closed-loop runs at 640 vs 416 both lock on moving and disturbed
        # targets with means within 0.2 px and 100% detection, while 320
        # degrades static center error to ~6.5 px and was rejected. Training
        # resolution is untouched; this is inference-only. stride=32
        # compatible (13*32). Explicit imgsz=640 still available per call.
        imgsz: int = 416,
        device: str = "cpu",
        defer_load: bool = False,
    ):
        self.requested_path = model_path
        self.model_path: Optional[Path] = resolve_model_path(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.model = None
        self.load_error: Optional[str] = None
        self._warmed = False
        # Serializes model load / warmup / inference across threads. The
        # only possible concurrency is a background warmup overlapping one
        # live tick (ticks themselves are strictly sequential), so this
        # lock is uncontended in steady state. Reentrant because warmup()
        # calls _load() while holding it.
        self._lock = threading.RLock()

        if self.model_path is None:
            self.load_error = (
                f"Model file not found: {_DEFAULT_MODEL_PATH} "
                f"(also checked {_DEFAULT_MODEL_ZIP})"
            )
            logger.warning(self.load_error)
        elif not defer_load:
            self._load()

    def _load(self) -> bool:
        """Lazy-load ultralytics YOLO model. Returns True on success."""
        if self.model is not None:
            return True
        with self._lock:
            if self.model is not None:
                return True
            if self.model_path is None or not self.model_path.exists():
                self.load_error = f"Model file missing: {self.model_path}"
                return False
            try:
                from ultralytics import YOLO

                self.model = YOLO(str(self.model_path))
                # Fuse/conftest not needed; keep defaults. Verbose off at predict time.
                logger.info(f"YOLO11Detector loaded {self.model_path} (conf={self.conf})")
                self.load_error = None
                return True
            except ImportError as e:
                self.load_error = (
                    f"ultralytics not installed (pip install ultralytics): {e}"
                )
                logger.warning(self.load_error)
                return False
            except Exception as e:
                self.load_error = f"Failed to load YOLO model {self.model_path}: {e}"
                logger.warning(self.load_error, exc_info=True)
                self.model = None
                return False

    def warmup(self) -> bool:
        """Load the model and run one inference pass in the calling thread.

        Idempotent and safe to call concurrently: concurrent callers share
        the single loaded model, and a second call after success is a fast
        no-op. Never raises; returns False when the model cannot be made
        ready. The frame is synthetic (camera-sized black image) and its
        result is discarded, so detection outputs are unaffected.
        """
        if self._warmed and self.model is not None:
            return True
        try:
            with self._lock:
                if not self._load():
                    return False
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                self.model.predict(
                    frame, conf=self.conf, imgsz=self.imgsz,
                    device=self.device, verbose=False,
                )
                self._warmed = True
                logger.info("YOLO11Detector warmup complete")
                return True
        except Exception as e:
            logger.warning(f"YOLO warmup failed: {e}")
            return False

    @property
    def is_available(self) -> bool:
        """True if model is loaded and ready for inference."""
        if self.model is not None:
            return True
        return self._load()

    def get_info(self) -> Dict:
        return {
            "type": "yolo",
            "available": self.model is not None,
            "model_path": str(self.model_path) if self.model_path else None,
            "conf": self.conf,
            "imgsz": self.imgsz,
            "device": self.device,
            "error": self.load_error,
        }

    def detect(self, frame) -> Optional[Dict]:
        """Detect beacon in BGR frame.

        Returns:
            {"x": cx, "y": cy, "confidence": conf,
             "bbox": [x1, y1, x2, y2]} or None.
        """
        if frame is None:
            return None
        if not self.is_available:
            return None
        try:
            # Ultralytics handles BGR numpy directly; verbose=False keeps logs clean.
            # Locked: a background warmup may otherwise run first-inference
            # autotuning on the same model concurrently.
            with self._lock:
                results = self.model.predict(
                    frame, conf=self.conf, imgsz=self.imgsz,
                    device=self.device, verbose=False,
                )
            if not results:
                return None
            r = results[0]
            if r.boxes is None or len(r.boxes) == 0:
                return None
            # Pick highest-confidence box (torch is already a hard
            # dependency via ultralytics, so no local import is needed).
            confs = r.boxes.conf.cpu().tolist()
            idx = max(range(len(confs)), key=lambda i: confs[i])
            box = r.boxes.xyxy.cpu().tolist()[idx]  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = (float(v) for v in box)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            confidence = float(confs[idx])
            bbox: List[float] = [x1, y1, x2, y2]
            return {
                "x": cx,
                "y": cy,
                "confidence": confidence,
                "bbox": bbox,
            }
        except Exception as e:
            logger.warning(f"YOLO inference failed: {e}", exc_info=True)
            return None


def create_detector(detector_type: str = "classical", **kwargs):
    """Factory reusing existing interfaces.

    Args:
        detector_type: "classical" | "yolo"
    """
    if detector_type == "yolo":
        return YOLO11Detector(**kwargs)
    elif detector_type == "classical":
        from .detector import BeaconDetector

        return BeaconDetector()
    raise ValueError(f"Unknown detector_type: {detector_type}. Use {AVAILABLE_DETECTORS}")
