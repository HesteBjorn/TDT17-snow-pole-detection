# Snow Pole Detection (TDT17)

Lightweight project shell for experimenting with snow pole detection using the provided Poles2025 dataset (YOLO-format labels, single `snow_pole` class). The goal is quick iteration on data exploration, training, and evaluation of small object-detection models (e.g., YOLO variants).

## Layout
- `Poles2025/` — dataset root (train/valid/test in YOLO format, plus RoadPoles-MSJ for extra qualitative use)
- `src/` — minimal, hackable Python modules
  - `config.py` — central paths/hyperparameters
  - `data.py` — dataset helpers, label parsing, small stats utilities
  - `training.py` — YOLO training/eval wrapper for fast experiments
- `scripts/` — thin CLIs for common tasks
  - `train.py` — run a training job from the CLI
  - `evaluate.py` — run evaluation on a saved checkpoint
- `notebooks/` — ad-hoc exploration notebooks

## Quickstart
1) Create an environment and install deps:
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2) (Optional) Adjust defaults in `src/config.py` if your dataset path differs.

3) Train a small model (Ultralytics YOLO):
```bash
python scripts/train.py --model yolov8n.pt --epochs 50 --imgsz 640
```

4) Evaluate a trained checkpoint:
```bash
python scripts/evaluate.py --weights runs/detect/train/weights/best.pt
```

5) Explore data:
- Open `notebooks/01_exploration.ipynb` for quick label counts, sample visuals, and split inspection.

## Notes
- The repo ignores `Poles2025/` to avoid accidental commits of the dataset.
- Metrics of interest: Precision, Recall, mAP@50, mAP@0.5:0.95 (Ultralytics reports these by default).
- Keep iterations short: start with nano/tiny models, low epochs, then scale up as needed.
