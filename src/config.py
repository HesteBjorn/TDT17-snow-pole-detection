from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from .data import resolve_dataset

@dataclass
class Paths:
    """Common filesystem locations."""

    dataset_root: Path = Path("Poles2025/Road_poles_iPhone")
    dataset_key: str = "iphone"  # 'v1' (folder splits) or 'iphone' (filelists)
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


@dataclass
class TorchvisionDetectionConfig:
    """Simple config for torchvision Faster R-CNN fine-tuning."""

    backbone: str = "resnet50_fpn"
    weights: str = "DEFAULT"  # torchvision enum name
    epochs: int = 30
    batch_size: int = 4
    num_workers: int = 4
    lr: float = 0.005
    weight_decay: float = 1e-4
    momentum: float = 0.9
    device: Optional[str] = None  # e.g. "cuda", "cpu"
    print_every: int = 10
    save_last: bool = True

    def asdict(self):
        return asdict(self)


def select_paths(dataset: str) -> Paths:
    """
    Build a Paths object for a named dataset ('v1' or 'iphone').
    """
    key, root, data_yaml = resolve_dataset(dataset)
    repo_root = Path(__file__).resolve().parent.parent
    base_root = root if root.is_absolute() else (repo_root / root)
    data_yaml_abs = None
    if data_yaml:
        data_yaml_abs = data_yaml if data_yaml.is_absolute() else (base_root / data_yaml)
    return Paths(dataset_root=base_root, data_yaml=data_yaml_abs, dataset_key=key)
