from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import yaml

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def load_data_yaml(path: Path) -> Dict:
    """Read YOLO data.yaml."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _find_image(stem: str, images_dir: Path) -> Optional[Path]:
    for ext in IMG_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def load_annotations(split: str, dataset_root: Path) -> pd.DataFrame:
    """
    Load YOLO annotations for a split into a tidy DataFrame.

    Columns: image_path, label_path, class_id, cx, cy, w, h
    """
    labels_dir = dataset_root / split / "labels"
    images_dir = dataset_root / split / "images"
    rows: List[dict] = []

    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    for label_file in sorted(labels_dir.glob("*.txt")):
        image_path = _find_image(label_file.stem, images_dir)
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                if image_path is None:
                    # Skip if corresponding image is missing
                    continue
                class_id, cx, cy, w, h = map(float, parts)
                rows.append(
                    {
                        "image_path": image_path,
                        "label_path": label_file,
                        "class_id": int(class_id),
                        "cx": cx,
                        "cy": cy,
                        "w": w,
                        "h": h,
                    }
                )
    return pd.DataFrame(rows)


def _normalize_rel_path(path_str: str) -> Path:
    """Strip common prefixes like 'data/' used in some filelists."""
    p = Path(path_str.strip())
    if p.parts and p.parts[0] == "data":
        p = Path(*p.parts[1:])
    return p


def _label_path_for_image(image_path: Path, dataset_root: Path, images_dirname: str = "images", labels_dirname: str = "labels") -> Path:
    """Infer label path from an image path using parallel images/labels structure."""
    try:
        rel = image_path.relative_to(dataset_root)
    except ValueError:
        rel = image_path
    parts = list(rel.parts)
    if parts and parts[0] == images_dirname:
        parts[0] = labels_dirname
    return dataset_root / Path(*parts).with_suffix(".txt")


def load_annotations_from_filelist(
    list_file: Path,
    dataset_root: Path,
    images_dirname: str = "images",
    labels_dirname: str = "labels",
) -> pd.DataFrame:
    """
    Load YOLO annotations using a file list (as in Road_poles_iPhone).

    The file list contains image paths; labels are inferred by swapping the images/ root to labels/.
    """
    rows: List[dict] = []
    with open(list_file, "r", encoding="utf-8") as f:
        image_paths = [dataset_root / _normalize_rel_path(line) for line in f if line.strip()]

    for image_path in image_paths:
        label_path = _label_path_for_image(image_path, dataset_root, images_dirname, labels_dirname)
        if not label_path.exists():
            continue
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id, cx, cy, w, h = map(float, parts)
                rows.append(
                    {
                        "image_path": image_path,
                        "label_path": label_path,
                        "class_id": int(class_id),
                        "cx": cx,
                        "cy": cy,
                        "w": w,
                        "h": h,
                    }
                )
    return pd.DataFrame(rows)


def split_summary(dataset_root: Path) -> pd.DataFrame:
    """Return a small summary of annotation counts per split."""
    records = []
    for split in ("train", "valid", "test"):
        labels_dir = dataset_root / split / "labels"
        images_dir = dataset_root / split / "images"
        label_files = list(labels_dir.glob("*.txt")) if labels_dir.exists() else []
        records.append(
            {
                "split": split,
                "num_label_files": len(label_files),
                "num_images": len(list(images_dir.glob("*")) if images_dir.exists() else []),
            }
        )
    return pd.DataFrame(records)


def iter_images(dataset_root: Path, split: str) -> Iterable[Path]:
    images_dir = dataset_root / split / "images"
    for ext in IMG_EXTENSIONS:
        yield from images_dir.glob(f"*{ext}")
