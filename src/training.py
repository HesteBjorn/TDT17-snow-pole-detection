from pathlib import Path
from typing import Dict, Optional

from ultralytics import YOLO

from .config import Paths, TrainingConfig, default_paths, default_training


def train_model(
    paths: Optional[Paths] = None,
    cfg: Optional[TrainingConfig] = None,
    overrides: Optional[Dict] = None,
):
    """Run a YOLO training job with lightweight defaults."""
    paths = paths or default_paths()
    cfg = cfg or default_training()

    train_args = cfg.to_ultralytics(paths)
    if overrides:
        train_args.update(overrides)

    model = YOLO(cfg.model)
    results = model.train(**train_args)
    return results


def evaluate_model(
    weights: Path,
    paths: Optional[Paths] = None,
    imgsz: int = 640,
    batch: int = 8,
    conf: float = 0.25,
    device: Optional[str] = None,
):
    """Evaluate a trained checkpoint; returns Ultralytics metrics object."""
    paths = paths or default_paths()
    model = YOLO(str(weights))

    val_args = {
        "data": str(paths.data_config),
        "imgsz": imgsz,
        "batch": batch,
        "conf": conf,
    }
    if device:
        val_args["device"] = device

    metrics = model.val(**val_args)
    return metrics
