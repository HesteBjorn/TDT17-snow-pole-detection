import argparse
from pathlib import Path

from src.config import Paths, TrainingConfig
from src.training import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO model for snow pole detection.")
    parser.add_argument("--data-root", type=Path, default=Path("Poles2025/roadpoles_v1"), help="Dataset root.")
    parser.add_argument("--data-yaml", type=Path, default=None, help="Override data.yaml location.")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Model checkpoint or config.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--device", type=str, default=None, help="Device string for Ultralytics (e.g. '0' or 'cpu').")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", type=str, default=None, help="Optional run name.")
    return parser.parse_args()


def main():
    args = parse_args()
    paths = Paths(dataset_root=args.data_root, data_yaml=args.data_yaml)
    cfg = TrainingConfig(
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        lr0=args.lr0,
        device=args.device,
        seed=args.seed,
        patience=args.patience,
    )

    overrides = {}
    if args.name:
        overrides["name"] = args.name

    train_model(paths=paths, cfg=cfg, overrides=overrides)


if __name__ == "__main__":
    main()
