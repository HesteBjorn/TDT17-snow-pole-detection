import argparse
from pathlib import Path

from src.config import Paths, TorchvisionDetectionConfig, TrainingConfig, select_paths
from src.predictors import get_detector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a detector for snow pole detection.")
    parser.add_argument("--backend", choices=["yolo", "fasterrcnn"], default="yolo", help="Training backend.")
    parser.add_argument("--dataset", choices=["v1", "iphone"], default="iphone", help="Dataset preset to use.")
    parser.add_argument("--data-root", type=Path, default=None, help="Optional dataset root override.")
    parser.add_argument("--data-yaml", type=Path, default=None, help="Override data.yaml location.")
    parser.add_argument("--device", type=str, default=None, help="Device string (e.g. '0', 'cuda', or 'cpu').")

    # YOLO-specific
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model checkpoint or config.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", type=str, default=None, help="Optional YOLO run name.")

    # Torchvision-specific
    parser.add_argument("--tv-epochs", type=int, default=30, help="Faster R-CNN epochs.")
    parser.add_argument("--tv-batch", type=int, default=4, help="Faster R-CNN batch size.")
    parser.add_argument("--tv-lr", type=float, default=0.005, help="Faster R-CNN learning rate.")
    parser.add_argument("--tv-workers", type=int, default=4, help="Faster R-CNN dataloader workers.")

    return parser.parse_args()


def main():
    args = parse_args()
    base_paths = select_paths(args.dataset)
    if args.data_root or args.data_yaml:
        base_paths = Paths(
            dataset_root=args.data_root or base_paths.dataset_root,
            data_yaml=args.data_yaml or base_paths.data_yaml,
            runs_dir=base_paths.runs_dir,
            artifacts_dir=base_paths.artifacts_dir,
        )
    paths = base_paths

    cfg_builders = {
        "yolo": lambda a: TrainingConfig(
            model=a.model,
            epochs=a.epochs,
            imgsz=a.imgsz,
            batch=a.batch,
            workers=a.workers,
            lr0=a.lr0,
            device=a.device,
            seed=a.seed,
            patience=a.patience,
        ),
        "fasterrcnn": lambda a: TorchvisionDetectionConfig(
            epochs=a.tv_epochs,
            batch_size=a.tv_batch,
            lr=a.tv_lr,
            num_workers=a.tv_workers,
            device=a.device,
        ),
    }

    cfg = cfg_builders[args.backend](args)
    overrides = {"name": args.name} if args.backend == "yolo" and args.name else None

    detector = get_detector(args.backend, paths=paths, train_cfg=cfg, overrides=overrides)
    detector.train()


if __name__ == "__main__":
    main()
