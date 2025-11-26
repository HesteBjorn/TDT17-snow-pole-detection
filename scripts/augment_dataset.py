"""
Create an augmented copy of the Poles2025 dataset with extra train samples.

Augmentations:
- Horizontal flip (labels are adjusted).
- Snow-like veil (photometric; labels unchanged).

Output structure mirrors the original:
- Poles2025_Aug/roadpoles_v1/{train,valid,test}/{images,labels}

Val/test are copied unchanged; train contains originals plus augmented variants
(*_flip.*, *_snow.*).
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def apply_snow(
    img: np.ndarray,
    density: float = 0.0004,
    min_r: int = 1,
    max_r: int = 4,
    prob_large: float = 0.05,
    max_large_r: int = 10,
) -> np.ndarray:
    """
    Draw sparse, soft snow blobs with varying shapes/opacities.

    density: approximate fraction of pixels that become flake centers (very low for spacing).
    radii: range of flake sizes in pixels.
    """
    h, w = img.shape[:2]
    num_flakes = max(1, int(h * w * density))
    mask = np.zeros((h, w), dtype=np.float32)
    for _ in range(num_flakes):
        # Mostly tiny flakes; rare larger ones
        if np.random.rand() < prob_large:
            r = np.random.randint(max_r + 1, max_large_r + 1)
        else:
            r = np.random.randint(min_r, max_r + 1)
        cx = np.random.randint(0, w)
        cy = np.random.randint(0, h)
        axes = (r, int(r * np.random.uniform(0.6, 1.4)))
        angle = np.random.uniform(0, 180)
        cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, 1.0, -1)

    # Soften edges and reduce brightness impact (allow some darker shades via slight inversion)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=2.5)
    alpha = np.clip(mask * 0.2, 0.0, 0.25)  # lighter veil
    alpha_3c = np.repeat(alpha[:, :, None], 3, axis=2)

    # Blend: mostly brighten, but mix in slight dark noise to vary shade
    dark_noise = np.random.uniform(0.0, 0.1, size=alpha_3c.shape).astype(np.float32)
    brighten = img.astype(np.float32) * (1.0 - alpha_3c) + 255.0 * alpha_3c
    darkened = img.astype(np.float32) * (1.0 - dark_noise)
    blended = brighten * 0.9 + darkened * 0.1
    return np.clip(blended, 0, 255).astype(np.uint8)


def flip_labels(label_path: Path) -> str:
    """Return flipped label content (YOLO format); x-center mirrored."""
    lines_out = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, cx, cy, w, h = map(float, parts)
            cx = 1.0 - cx
            lines_out.append(f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return "\n".join(lines_out)


def copy_split(src_root: Path, dst_root: Path, split: str):
    """Copy non-train splits unchanged (only existing subfolders)."""
    for sub in ["images", "labels"]:
        src_dir = src_root / split / sub
        if not src_dir.exists():
            continue
        dst_dir = dst_root / split / sub
        dst_dir.mkdir(parents=True, exist_ok=True)
        for path in src_dir.iterdir():
            if path.is_file():
                shutil.copy2(path, dst_dir / path.name)


def augment_train(src_root: Path, dst_root: Path):
    src_images = src_root / "train" / "images"
    src_labels = src_root / "train" / "labels"
    dst_images = dst_root / "train" / "images"
    dst_labels = dst_root / "train" / "labels"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    for img_path in src_images.iterdir():
        if img_path.suffix not in IMG_EXTS:
            continue
        stem = img_path.stem
        label_path = src_labels / f"{stem}.txt"
        if not label_path.exists():
            continue

        # Copy original
        shutil.copy2(img_path, dst_images / img_path.name)
        shutil.copy2(label_path, dst_labels / label_path.name)

        # Snow
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        snow_img = apply_snow(img)
        snow_name = f"{stem}_snow{img_path.suffix}"
        cv2.imwrite(str(dst_images / snow_name), snow_img)
        shutil.copy2(label_path, dst_labels / f"{stem}_snow.txt")

        # Horizontal flip
        flip_img = cv2.flip(img, 1)
        flip_name = f"{stem}_flip{img_path.suffix}"
        cv2.imwrite(str(dst_images / flip_name), flip_img)
        flipped_labels = flip_labels(label_path)
        with open(dst_labels / f"{stem}_flip.txt", "w", encoding="utf-8") as f:
            f.write(flipped_labels)


def main():
    parser = argparse.ArgumentParser(description="Create augmented copy of Poles2025 with snow+flip.")
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("Poles2025/roadpoles_v1"),
        help="Source dataset root (YOLO format).",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("Poles2025_Aug/roadpoles_v1"),
        help="Destination root for augmented dataset.",
    )
    parser.add_argument("--force", action="store_true", help="Allow writing into existing destination.")
    args = parser.parse_args()

    if args.dst.exists() and not args.force:
        raise SystemExit(f"Destination {args.dst} exists. Use --force to write into it.")

    # Copy data.yaml as-is
    args.dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.src / "data.yaml", args.dst / "data.yaml")

    # Copy val/test unchanged (no augmentation)
    for split in ["valid", "test"]:
        copy_split(args.src, args.dst, split)

    # Augment train
    augment_train(args.src, args.dst)
    print(f"Augmented dataset written to {args.dst}")


if __name__ == "__main__":
    main()
