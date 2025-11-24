import time
from pathlib import Path
from typing import Dict, Optional

from ultralytics import YOLO
import yaml

from ..config import TrainingConfig, default_training
from .base import Detector, EvalResult, TrainResult


class YoloDetector(Detector):
    """Ultralytics YOLO wrapper."""

    name = "yolo"

    def __init__(
        self,
        paths=None,
        train_cfg: Optional[TrainingConfig] = None,
        overrides: Optional[Dict] = None,
    ):
        super().__init__(paths=paths)
        self.cfg = train_cfg or default_training()
        self.overrides = overrides or {}

    def _prepare_data_yaml(self) -> Path:
        """Ultralytics expects train/val keys; map from Train/Validation if needed."""
        orig = self.paths.data_config
        with open(orig, "r", encoding="utf-8") as f:
            data_cfg = yaml.safe_load(f)

        if "train" in data_cfg and "val" in data_cfg:
            return orig

        # Map capitalized keys to expected ones and resolve to absolute paths.
        def _resolve_path(value: str) -> Path:
            p = Path(value)
            if not p.is_absolute():
                p = self.paths.dataset_root / p
            return p

        artifacts_root = (
            self.paths.artifacts_dir
            if self.paths.artifacts_dir.is_absolute()
            else (self.paths.dataset_root.parent / self.paths.artifacts_dir)
        ).resolve()
        artifacts_root.mkdir(parents=True, exist_ok=True)

        def _rewrite_filelist(list_path: Path, out_name: str) -> Path:
            """Make a new filelist with absolute paths, fixing any leading 'data/'."""
            lines = []
            with open(list_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("data/"):
                        line = line[len("data/") :]
                    p = self.paths.dataset_root / line
                    lines.append(str(p.resolve()))
            out_path = (artifacts_root / out_name).resolve()
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            return out_path

        train = data_cfg.get("Train") or data_cfg.get("train")
        val = data_cfg.get("Validation") or data_cfg.get("val")
        test = data_cfg.get("Test") or data_cfg.get("test")
        names = data_cfg.get("names")

        mapped = {}
        if train:
            train_path = _resolve_path(train)
            if train_path.suffix.lower() in {".txt"}:
                train_path = _rewrite_filelist(train_path, "train_list.txt")
            mapped["train"] = str(train_path)
        if val:
            val_path = _resolve_path(val)
            if val_path.suffix.lower() in {".txt"}:
                val_path = _rewrite_filelist(val_path, "val_list.txt")
            mapped["val"] = str(val_path)
        if test:
            test_path = _resolve_path(test)
            if test_path.suffix.lower() in {".txt"}:
                test_path = _rewrite_filelist(test_path, "test_list.txt")
            mapped["test"] = str(test_path)
        if names:
            mapped["names"] = names

        new_yaml = (artifacts_root / "data_resolved.yaml").resolve()
        with open(new_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(mapped, f)
        return new_yaml

    def train(self) -> TrainResult:
        data_yaml = self._prepare_data_yaml()
        args = self.cfg.to_ultralytics(self.paths)
        args["data"] = str(data_yaml)
        args.update(self.overrides)

        model = YOLO(self.cfg.model)
        start = time.perf_counter()
        results = model.train(**args)
        duration = time.perf_counter() - start

        last_checkpoint = None
        best_checkpoint = None
        save_dir = getattr(results, "save_dir", None)
        if save_dir:
            candidate_last = Path(save_dir) / "weights" / "last.pt"
            if candidate_last.exists():
                last_checkpoint = candidate_last
            candidate_best = Path(save_dir) / "weights" / "best.pt"
            if candidate_best.exists():
                best_checkpoint = candidate_best

        print(f"[yolo][train] time: {duration:.2f}s ({duration/60:.2f} min)")
        return TrainResult(
            last_checkpoint=best_checkpoint or last_checkpoint,
            train_time_seconds=duration,
            history={},
            save_dir=Path(save_dir) if save_dir else None,
            raw=results,
        )

    def evaluate(
        self,
        weights: Path,
        imgsz: int = 640,
        batch: int = 8,
        conf: float = 0.25,
        device: Optional[str] = None,
        **_,
    ) -> EvalResult:
        model = YOLO(str(weights))

        data_yaml = self._prepare_data_yaml()
        val_args = {
            "data": str(data_yaml),
            "imgsz": imgsz,
            "batch": batch,
            "conf": conf,
        }
        if device:
            val_args["device"] = device

        start = time.perf_counter()
        metrics = model.val(**val_args)
        duration = time.perf_counter() - start
        metrics.eval_time_seconds = duration
        print(f"[yolo][eval] time: {duration:.2f}s ({duration/60:.2f} min)")

        results_dict = getattr(metrics, "results_dict", None) or {}
        return EvalResult(
            results_dict=results_dict,
            eval_time_seconds=duration,
            raw=metrics,
        )
