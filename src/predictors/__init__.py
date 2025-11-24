from typing import Dict, Type

from .base import Detector
from .fasterrcnn import FasterRCNNDetector
from .yolo import YoloDetector

_REGISTRY: Dict[str, Type[Detector]] = {
    "yolo": YoloDetector,
    "fasterrcnn": FasterRCNNDetector,
}


def get_detector(name: str, **kwargs) -> Detector:
    try:
        cls = _REGISTRY[name.lower()]
    except KeyError:
        raise ValueError(f"Unknown detector backend: {name}") from None
    return cls(**kwargs)


__all__ = ["Detector", "YoloDetector", "FasterRCNNDetector", "get_detector"]
