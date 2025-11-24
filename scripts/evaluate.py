import argparse
from pathlib import Path

from src.config import Paths
from src.predictors import get_detector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--backend", choices=["yolo", "fasterrcnn"], default="yolo")
    parser.add_argument("--weights", type=Path, required=True, help="Path to trained weights (e.g., best.pt or .pth).")
    parser.add_argument("--data-root", type=Path, default=Path("Poles2025/Road_poles_iPhone"))
    parser.add_argument("--data-yaml", type=Path, default=None)
    parser.add_argument("--imgsz", type=int, default=640, help="Only used by YOLO.")
    parser.add_argument("--batch", type=int, default=8, help="Only used by YOLO.")
    parser.add_argument("--conf", type=float, default=0.25, help="Only used by YOLO.")
    parser.add_argument("--device", type=str, default=None, help="Device string (e.g. '0', 'cuda', or 'cpu').")
    return parser.parse_args()


def main():
    args = parse_args()
    paths = Paths(dataset_root=args.data_root, data_yaml=args.data_yaml)
    detector = get_detector(args.backend, paths=paths)
    metrics = detector.evaluate(
        weights=args.weights,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        device=args.device,
    )

    results = getattr(metrics, "results_dict", None) or {}
    for key in ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"):
        if key in results:
            print(f"{key}: {results[key]:.4f}")
    duration = getattr(metrics, "eval_time_seconds", None)
    if duration:
        print(f"eval_time_seconds: {duration:.2f}")
        print(f"eval_time_minutes: {duration/60:.2f}")


if __name__ == "__main__":
    main()
