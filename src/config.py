from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class Paths:
    """Common filesystem locations."""

    dataset_root: Path = Path("Poles2025/roadpoles_v1")
    runs_dir: Path = Path("runs")
    artifacts_dir: Path = Path("artifacts")
    data_yaml: Optional[Path] = None

    @property
    def data_config(self) -> Path:
        """Path to the YOLO data.yaml file."""
        if self.data_yaml:
            return self.data_yaml
        return self.dataset_root / "data.yaml"


@dataclass
class TrainingConfig:
    """Training hyperparameters and paths."""

    model: str = "yolov8n.pt"
    epochs: int = 50
    imgsz: int = 640
    batch: int = 8
    workers: int = 4
    lr0: float = 0.01
    device: Optional[str] = None  # e.g. '0' for first GPU, 'cpu' to force CPU
    seed: int = 42
    patience: int = 50

    def to_ultralytics(self, paths: Paths) -> dict:
        """Return kwargs for YOLO train/val calls."""
        cfg = {
            "data": str(paths.data_config),
            "epochs": self.epochs,
            "imgsz": self.imgsz,
            "batch": self.batch,
            "workers": self.workers,
            "lr0": self.lr0,
            "seed": self.seed,
            "patience": self.patience,
            "project": str(paths.runs_dir),
        }
        if self.device:
            cfg["device"] = self.device
        return cfg

    def asdict(self):
        return asdict(self)


def default_paths() -> Paths:
    return Paths()


def default_training() -> TrainingConfig:
    return TrainingConfig()
