"""
Create an augmented copy of the Poles2025 datasets with extra train samples.

Supports:
- roadpoles_v1 (YOLO folder split)
- Road_poles_iPhone (filelists + images/Train/train layout)

Augmentations:
- Horizontal flip (labels adjusted)
- Snow-like veil (photometric; labels unchanged)

Val/test are copied unchanged. Train contains originals plus *_flip and *_snow.
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def label_path_for_image(image_path: Path, root: Path, images_dirname: str = "images", labels_dirname: str = "labels") -> Path:
    """Infer label path by swapping images_dirname with labels_dirname and .txt suffix."""
    try:
        rel = image_path.relative_to(root)
    except ValueError:
        rel = image_path
    parts = list(rel.parts)
    if parts and parts[0] == images_dirname:
        parts[0] = labels_dirname
    return root / Path(*parts).with_suffix(".txt")


def normalize_rel(path_str: str) -> Path:
    """Strip leading 'data/' if present."""
    p = Path(path_str.strip())
    if p.parts and p.parts[0] == "data":
        p = Path(*p.parts[1:])
    return p


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


# --- YOLO folder-style (roadpoles_v1) helpers ---


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


def augment_train_yolo(src_root: Path, dst_root: Path):
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

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Snow
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


# --- iPhone filelist-style helpers ---


def copy_from_filelist(list_path: Path, src_root: Path, dst_root: Path, do_aug: bool = False) -> None:
    """
    Copy images (and labels if present) from a filelist.
    If do_aug=True, add snow + flip variants with updated labels and add them to the list.
    Writes a new list file in dst_root with absolute paths for the copied (and augmented) images.
    """
    entries = []
    with open(list_path, "r", encoding="utf-8") as f:
        entries = [line.strip() for line in f if line.strip()]

    dst_entries = []
    for line in entries:
        rel = normalize_rel(line)
        src_img = src_root / rel
        if src_img.suffix not in IMG_EXTS or not src_img.exists():
            continue
        dst_img = dst_root / rel
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_img, dst_img)

        src_label = label_path_for_image(src_img, src_root, images_dirname="images", labels_dirname="labels")
        dst_label = label_path_for_image(dst_img, dst_root, images_dirname="images", labels_dirname="labels")
        if src_label.exists():
            dst_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_label, dst_label)

        dst_entries.append(str(dst_img.resolve()))

        if not do_aug:
            continue

        img = cv2.imread(str(src_img))
        if img is None:
            continue
        stem = src_img.stem
        # Snow (slightly larger flakes for iPhone set by bumping max sizes)
        snow_img = apply_snow(img, max_r=6, prob_large=0.08, max_large_r=14)
        snow_name = f"{stem}_snow{src_img.suffix}"
        snow_dst = dst_img.parent / snow_name
        cv2.imwrite(str(snow_dst), snow_img)
        if src_label.exists():
            snow_label = label_path_for_image(snow_dst, dst_root, images_dirname="images", labels_dirname="labels")
            snow_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_label, snow_label)
        dst_entries.append(str(snow_dst.resolve()))

        # Flip
        flip_img = cv2.flip(img, 1)
        flip_name = f"{stem}_flip{src_img.suffix}"
        flip_dst = dst_img.parent / flip_name
        cv2.imwrite(str(flip_dst), flip_img)
        if src_label.exists():
            flipped = flip_labels(src_label)
            flip_label = label_path_for_image(flip_dst, dst_root, images_dirname="images", labels_dirname="labels")
            flip_label.parent.mkdir(parents=True, exist_ok=True)
            with open(flip_label, "w", encoding="utf-8") as f:
                f.write(flipped)
        dst_entries.append(str(flip_dst.resolve()))

    # Write new list file with absolute paths
    dst_list = dst_root / list_path.name
    with open(dst_list, "w", encoding="utf-8") as f:
        f.write("\n".join(dst_entries))


def detect_layout(src_root: Path) -> str:
    if (src_root / "Train.txt").exists():
        return "iphone"
    if (src_root / "train").exists() and (src_root / "train" / "images").exists():
        return "yolo"
    raise SystemExit(f"Unrecognized dataset layout at {src_root}")


def main():
    parser = argparse.ArgumentParser(description="Create augmented copy of Poles2025 with snow+flip.")
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("Poles2025/roadpoles_v1"),
        help="Source dataset root.",
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

    layout = detect_layout(args.src)
    args.dst.mkdir(parents=True, exist_ok=True)
    if (args.src / "data.yaml").exists():
        shutil.copy2(args.src / "data.yaml", args.dst / "data.yaml")

    if layout == "yolo":
        # Copy val/test unchanged
        for split in ["valid", "test"]:
            copy_split(args.src, args.dst, split)
        augment_train_yolo(args.src, args.dst)
    else:
        # iPhone filelist style
        # Copy val/test lists and assets without aug
        for split_file in ["Validation.txt", "Test.txt"]:
            if (args.src / split_file).exists():
                copy_from_filelist(args.src / split_file, args.src, args.dst, do_aug=False)
        # Train with aug
        if (args.src / "Train.txt").exists():
            copy_from_filelist(args.src / "Train.txt", args.src, args.dst, do_aug=True)

    print(f"Augmented dataset written to {args.dst}")


if __name__ == "__main__":
    main()
