from pathlib import Path
from typing import Optional

from .config import Paths, TrainingConfig, default_paths, default_training
from .predictors import get_detector
from .predictors.base import EvalResult, TrainResult


def train_model(
    paths: Optional[Paths] = None,
    cfg: Optional[TrainingConfig] = None,
    overrides: Optional[dict] = None,
) -> TrainResult:
    """Backward-compatible YOLO training wrapper using the detector interface."""
    detector = get_detector(
        "yolo",
        paths=paths or default_paths(),
        train_cfg=cfg or default_training(),
        overrides=overrides,
    )
    return detector.train()


def evaluate_model(
    weights: Path,
    paths: Optional[Paths] = None,
    imgsz: int = 640,
    batch: int = 8,
    conf: float = 0.25,
    device: Optional[str] = None,
) -> EvalResult:
    """Backward-compatible YOLO evaluation wrapper using the detector interface."""
    detector = get_detector("yolo", paths=paths or default_paths())
    return detector.evaluate(weights=weights, imgsz=imgsz, batch=batch, conf=conf, device=device)


def train_detector(
    backend: str = "yolo",
    paths: Optional[Paths] = None,
    train_cfg=None,
    overrides: Optional[dict] = None,
):
    """Backend-agnostic train entrypoint using the detector registry."""
    detector = get_detector(backend, paths=paths or default_paths(), train_cfg=train_cfg, overrides=overrides)
    return detector.train()
