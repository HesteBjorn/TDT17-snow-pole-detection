from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import Paths, default_paths


@dataclass
class TrainResult:
    last_checkpoint: Optional[Path]
    train_time_seconds: float
    history: Dict[str, Any]
    save_dir: Optional[Path] = None
    raw: Optional[Any] = None


@dataclass
class EvalResult:
    results_dict: Dict[str, float]
    eval_time_seconds: Optional[float] = None
    raw: Optional[Any] = None


class Detector(ABC):
    """Common detector interface."""

    name: str

    def __init__(self, paths: Optional[Paths] = None):
        self.paths = paths or default_paths()

    @abstractmethod
    def train(self) -> TrainResult:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, weights: Path, **kwargs) -> EvalResult:
        raise NotImplementedError
