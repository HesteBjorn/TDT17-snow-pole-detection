import random
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

from ..config import TorchvisionDetectionConfig
from ..data import _label_path_for_image, _normalize_rel_path, load_data_yaml
from .base import Detector, EvalResult, TrainResult


def _read_yolo_labels(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    labels: List[Tuple[int, float, float, float, float]] = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, cx, cy, w, h = map(float, parts)
            labels.append((int(cls), cx, cy, w, h))
    return labels


def _yolo_to_xyxy(
    labels: List[Tuple[int, float, float, float, float]], width: int, height: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    boxes = []
    classes = []
    for cls, cx, cy, w, h in labels:
        cx_abs, cy_abs = cx * width, cy * height
        bw, bh = w * width, h * height
        x1 = cx_abs - bw / 2.0
        y1 = cy_abs - bh / 2.0
        x2 = cx_abs + bw / 2.0
        y2 = cy_abs + bh / 2.0
        boxes.append([x1, y1, x2, y2])
        classes.append(cls + 1)  # shift to make background class 0
    if not boxes:
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)
    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(classes, dtype=torch.int64)


class YoloListDataset(Dataset):
    """Dataset that reads YOLO-format filelists (as in Road_poles_iPhone)."""

    def __init__(
        self,
        list_file: Path,
        dataset_root: Path,
        transforms: Optional[Callable] = None,
        images_dirname: str = "images",
        labels_dirname: str = "labels",
    ):
        self.dataset_root = dataset_root
        self.transforms = transforms
        with open(list_file, "r", encoding="utf-8") as f:
            self.image_paths = [
                dataset_root / _normalize_rel_path(line) for line in f if line.strip()
            ]
        self.images_dirname = images_dirname
        self.labels_dirname = labels_dirname
        self.image_info: List[Tuple[int, Path, int, int]] = []
        for idx, img_path in enumerate(self.image_paths):
            with Image.open(img_path) as im:
                w, h = im.size
            self.image_info.append((idx, img_path, w, h))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_id, img_path, w, h = self.image_info[idx]
        label_path = _label_path_for_image(
            img_path,
            self.dataset_root,
            images_dirname=self.images_dirname,
            labels_dirname=self.labels_dirname,
        )
        img = Image.open(img_path).convert("RGB")
        labels = _read_yolo_labels(label_path)
        boxes, classes = _yolo_to_xyxy(labels, w, h)

        target: Dict[str, torch.Tensor] = {
            "boxes": boxes,
            "labels": classes,
            "image_id": torch.tensor([img_id]),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if len(boxes) else torch.tensor([]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
            "orig_size": torch.tensor([h, w]),
            "image_path": str(img_path),
        }

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target


def default_transforms(train: bool = True) -> Callable:
    """Horizontal flip + tensor conversion; keep poles upright."""

    def _apply(img, target):
        if train and random.random() < 0.5:
            img = F.hflip(img)
            if "boxes" in target and len(target["boxes"]):
                w, _ = img.size
                boxes = target["boxes"]
                x1 = w - boxes[:, 2]
                x2 = w - boxes[:, 0]
                boxes = torch.stack((x1, boxes[:, 1], x2, boxes[:, 3]), dim=1)
                target["boxes"] = boxes
        img = F.to_tensor(img)
        return img, target

    return _apply


def _resolve_weight_enum(weights: str):
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

    if weights == "DEFAULT":
        return FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    return getattr(FasterRCNN_ResNet50_FPN_Weights, weights)


def build_fasterrcnn(num_classes: int = 2, weights: str = "DEFAULT") -> FasterRCNN:
    from torchvision.models.detection import fasterrcnn_resnet50_fpn

    weight_enum = _resolve_weight_enum(weights)
    model = fasterrcnn_resnet50_fpn(weights=weight_enum)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def _collate(batch):
    return tuple(zip(*batch))


class FasterRCNNDetector(Detector):
    """Torchvision Faster R-CNN fine-tuning with COCO pretrain."""

    name = "fasterrcnn"

    def __init__(self, paths=None, train_cfg: Optional[TorchvisionDetectionConfig] = None, overrides=None):
        super().__init__(paths=paths)
        self.cfg = train_cfg or TorchvisionDetectionConfig()
        self.overrides = overrides or {}

    def _make_dataloaders(self):
        data_cfg = load_data_yaml(self.paths.data_config)
        dataset_root = self.paths.dataset_root
        train_list = dataset_root / data_cfg["Train"]
        val_list = dataset_root / data_cfg["Validation"]

        train_ds = YoloListDataset(train_list, dataset_root, transforms=default_transforms(train=True))
        val_ds = YoloListDataset(val_list, dataset_root, transforms=default_transforms(train=False))

        train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            collate_fn=_collate,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            collate_fn=_collate,
        )
        return train_loader, val_loader, val_ds

    def train(self) -> TrainResult:
        device = torch.device(self.cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = build_fasterrcnn(num_classes=2, weights=self.cfg.weights).to(device)

        train_loader, val_loader, _ = self._make_dataloaders()
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params,
            lr=self.cfg.lr,
            momentum=self.cfg.momentum,
            weight_decay=self.cfg.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        start = time.perf_counter()
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.cfg.epochs):
            model.train()
            running_loss = 0.0
            for i, (images, targets) in enumerate(train_loader):
                images = [img.to(device) for img in images]
                targets = [
                    {k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets
                ]
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                running_loss += losses.item()
                if self.cfg.print_every and i % self.cfg.print_every == 0:
                    print(
                        f"[fasterrcnn][epoch {epoch+1}/{self.cfg.epochs}][{i}/{len(train_loader)}] "
                        f"loss: {losses.item():.4f}"
                    )

            lr_scheduler.step()
            epoch_loss = running_loss / max(1, len(train_loader))
            history["train_loss"].append(epoch_loss)

        # For validation loss, keep model in train mode but disable grad so we get loss dicts.
        val_loss = 0.0
        with torch.no_grad():
            model.train()
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [
                    {k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets
                ]
                loss_dict = model(images, targets)
                if isinstance(loss_dict, dict):
                    losses = sum(loss for loss in loss_dict.values())
                    val_loss += losses.item()
        val_loss = val_loss / max(1, len(val_loader)) if len(val_loader) else 0.0
        history["val_loss"].append(val_loss)
        print(f"[fasterrcnn][epoch {epoch+1}] train_loss={epoch_loss:.4f} val_loss={val_loss:.4f}")

        duration = time.perf_counter() - start
        last_ckpt = None
        if self.cfg.save_last:
            self.paths.runs_dir.mkdir(parents=True, exist_ok=True)
            last_ckpt = self.paths.runs_dir / "fasterrcnn_last.pth"
            torch.save(model.state_dict(), last_ckpt)
            print(f"[fasterrcnn][save] checkpoint -> {last_ckpt}")

        return TrainResult(last_checkpoint=last_ckpt, train_time_seconds=duration, history=history, raw=None)

    def evaluate(
        self,
        weights: Path,
        imgsz: int = 0,
        batch: int = 0,
        conf: float = 0.0,
        device: Optional[str] = None,
        **_,
    ) -> EvalResult:
        """Evaluation with val loss + COCO-style metrics (mAP@50, mAP@50-95) and simple P/R."""
        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
        except ImportError as e:
            raise ImportError("pycocotools is required for Faster R-CNN evaluation metrics.") from e

        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = build_fasterrcnn(num_classes=2, weights=self.cfg.weights).to(device)
        state = torch.load(weights, map_location=device)
        model.load_state_dict(state)

        _, val_loader, val_ds = self._make_dataloaders()

        # Build COCO-style ground truth
        images_gt = []
        annotations_gt = []
        ann_id = 1
        for idx, img_path, w, h in val_ds.image_info:
            images_gt.append({"id": idx, "file_name": str(img_path), "width": w, "height": h})
            label_path = _label_path_for_image(
                img_path, val_ds.dataset_root, images_dirname=val_ds.images_dirname, labels_dirname=val_ds.labels_dirname
            )
            labels = _read_yolo_labels(label_path)
            for cls, cx, cy, bw, bh in labels:
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                annotations_gt.append(
                    {
                        "id": ann_id,
                        "image_id": idx,
                        "category_id": 1,
                        "bbox": [x1, y1, bw * w, bh * h],
                        "area": bw * w * bh * h,
                        "iscrowd": 0,
                    }
                )
                ann_id += 1
        coco_gt = COCO()
        coco_gt.dataset = {
            "images": images_gt,
            "annotations": annotations_gt,
            "categories": [{"id": 1, "name": "pole"}],
            "info": {"description": "road_poles_iPhone val"},
        }
        coco_gt.createIndex()

        det_results = []
        tp = fp = 0
        total_gt = len(annotations_gt)

        start = time.perf_counter()
        val_loss = 0.0
        with torch.no_grad():
            model.eval()
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [
                    {k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets
                ]
                outputs = model(images)

                # Validation loss using train-mode pass
                model.train()
                loss_dict = model(images, targets)
                model.eval()
                if isinstance(loss_dict, dict):
                    losses = sum(loss for loss in loss_dict.values())
                    val_loss += losses.item()

                for out, tgt in zip(outputs, targets):
                    image_id = int(tgt["image_id"].item())
                    boxes = out["boxes"].cpu()
                    scores = out["scores"].cpu()
                    for box, score in zip(boxes, scores):
                        x1, y1, x2, y2 = box.tolist()
                        det_results.append(
                            {
                                "image_id": image_id,
                                "category_id": 1,
                                "bbox": [x1, y1, x2 - x1, y2 - y1],
                                "score": float(score.item()),
                            }
                        )
                    # Simple P/R at IoU>=0.5 using greedy matching
                    if conf and len(boxes):
                        keep = scores >= conf
                        boxes = boxes[keep]
                        scores_keep = scores[keep]
                    else:
                        scores_keep = scores
                    if len(boxes):
                        gt_boxes = tgt["boxes"].cpu()
                        matched = set()
                        order = torch.argsort(scores_keep, descending=True)
                        for idx_det in order:
                            b = boxes[idx_det]
                            best_iou = 0.0
                            best_gt = -1
                            for gi, g in enumerate(gt_boxes):
                                if gi in matched:
                                    continue
                                # IoU
                                xa = max(b[0], g[0])
                                ya = max(b[1], g[1])
                                xb = min(b[2], g[2])
                                yb = min(b[3], g[3])
                                inter = max(0, xb - xa) * max(0, yb - ya)
                                area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
                                area_g = max(0, g[2] - g[0]) * max(0, g[3] - g[1])
                                union = area_b + area_g - inter + 1e-9
                                iou_val = inter / union
                                if iou_val > best_iou:
                                    best_iou = iou_val
                                    best_gt = gi
                            if best_iou >= 0.5 and best_gt >= 0:
                                tp += 1
                                matched.add(best_gt)
                            else:
                                fp += 1
                    else:
                        fp += 0

        val_loss = val_loss / max(1, len(val_loader)) if len(val_loader) else 0.0
        coco_dt = coco_gt.loadRes(det_results) if det_results else coco_gt.loadRes([])
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        duration = time.perf_counter() - start
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / total_gt if total_gt > 0 else 0.0

        results = {
            "metrics/precision(B)": precision,
            "metrics/recall(B)": recall,
            "metrics/mAP50(B)": coco_eval.stats[1] if coco_eval.stats is not None else None,
            "metrics/mAP50-95(B)": coco_eval.stats[0] if coco_eval.stats is not None else None,
            "val_loss": val_loss,
        }

        map50_val = results["metrics/mAP50(B)"]
        print(
            f"[fasterrcnn][eval] val_loss={val_loss:.4f} "
            f"precision={precision:.3f} recall={recall:.3f} "
            f"mAP50={(map50_val if map50_val is not None else float('nan')):.3f}"
        )

        return EvalResult(results_dict={k: v for k, v in results.items() if v is not None}, eval_time_seconds=duration, raw=coco_eval)
