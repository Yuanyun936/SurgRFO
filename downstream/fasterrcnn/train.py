#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.transforms import functional as TF

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# --------------------------
# Config / constants
# --------------------------
# Detection classes: background(0) + sponge(1)
CLASS_TO_ID = {"sponge": 1}
ID_TO_CLASS = {1: "sponge"}

IMG_EXTS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")


# --------------------------
# Utils
# --------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    # Return (images: list[Tensor], targets: list[dict])
    return tuple(zip(*batch))


def read_manifest(manifest_csv: Path, split: str) -> List[Dict[str, str]]:

    rows: List[Dict[str, str]] = []
    with open(manifest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split") == split:
                rows.append(row)

    if not rows:
        raise RuntimeError(f"No rows found for split='{split}' in {manifest_csv}")
    return rows


def pil_load_rgb(path: Path) -> Image.Image:
    return Image.open(str(path)).convert("RGB")


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between boxes a and b in [x1,y1,x2,y2] format.

    a: (Na, 4), b: (Nb, 4) -> (Na, Nb)
    """
    if a.size == 0 or b.size == 0:
        if a.ndim == 2:
            return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
        return np.zeros((0, 0), dtype=np.float32)

    a = a.astype(np.float32)
    b = b.astype(np.float32)

    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)

    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by2 + (by2 * 0))
    area_b = (bx2 - bx1) * (by2 - by1)

    union = area_a + area_b - inter
    return inter / np.clip(union, 1e-6, None)


# --------------------------
# Metrics
# --------------------------

def ap_iou50_single_class(preds, gts, class_id: int, score_thr: float = 0.5, iou_thr: float = 0.5):
    """Compute AP@0.5 for one detection class; also return P/R/F1.

    preds, gts: lists of dicts from Faster R-CNN output / targets.
    """
    all_scores: List[float] = []
    all_matches: List[int] = []
    total_gt = 0
    tp = fp = fn = 0

    for pred, gt in zip(preds, gts):
        gt_boxes = gt["boxes"]
        gt_labels = gt["labels"]
        gt_mask = (gt_labels == class_id)
        gt_boxes = gt_boxes[gt_mask]
        total_gt += len(gt_boxes)

        p_boxes = pred["boxes"]
        p_scores = pred["scores"]
        p_labels = pred["labels"]
        keep = (p_labels == class_id) & (p_scores >= score_thr)
        p_boxes = p_boxes[keep]
        p_scores = p_scores[keep]

        order = np.argsort(-p_scores)
        p_boxes = p_boxes[order]
        p_scores = p_scores[order]

        matched_gt = np.zeros(len(gt_boxes), dtype=bool)
        for i in range(len(p_boxes)):
            if len(gt_boxes) == 0:
                fp += 1
                all_scores.append(float(p_scores[i]))
                all_matches.append(0)
                continue

            ious = iou_xyxy(p_boxes[i : i + 1].cpu().numpy(), gt_boxes.cpu().numpy())[0]
            j = int(np.argmax(ious))
            if ious[j] >= iou_thr and not matched_gt[j]:
                matched_gt[j] = True
                tp += 1
                all_scores.append(float(p_scores[i]))
                all_matches.append(1)
            else:
                fp += 1
                all_scores.append(float(p_scores[i]))
                all_matches.append(0)

        fn += int((~matched_gt).sum())

    if len(all_scores) == 0:
        precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
        recall = 0.0 if total_gt == 0 else tp / total_gt
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        return 0.0, precision, recall, f1

    order = np.argsort(-np.asarray(all_scores))
    matches = np.asarray(all_matches)[order]
    cum_tp = np.cumsum(matches)
    cum_fp = np.cumsum(1 - matches)
    recalls = cum_tp / max(total_gt, 1)
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-6)

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

    precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 0.0 if total_gt == 0 else tp / total_gt
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return ap, precision, recall, f1


def bin_preds_from_dets(preds, score_thr: float = 0.5) -> np.ndarray:
    """Convert detection outputs to binary image-level predictions.

    Positive (1) if any detection of class 1 with score >= thr; else 0.
    """
    bin_preds: List[int] = []
    for pred in preds:
        pl = pred["labels"]
        ps = pred["scores"]
        keep = (ps >= score_thr) & (pl == 1)
        bin_preds.append(1 if torch.any(keep) else 0)
    return np.asarray(bin_preds, dtype=np.int64)


def classification_metrics_binary(y_true, y_pred) -> Dict[str, float]:
    """Binary metrics (classes={0,1}): accuracy + macro P/R/F1.

    Class 0: normal; Class 1: positive (sponge present).
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    acc = float((y_true == y_pred).mean()) if len(y_true) > 0 else 0.0

    precisions, recalls, f1s = [], [], []
    for c in (0, 1):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    return {
        "acc": float(acc),
        "macro_P": float(np.mean(precisions)),
        "macro_R": float(np.mean(recalls)),
        "macro_F1": float(np.mean(f1s)),
    }


# --------------------------
# Dataset
# --------------------------


class LabelmeSpongeDataset(data.Dataset):
    """Dataset reading LabelMe JSON + images for sponge detection.

    Each sample returns (image_tensor, target_dict):
    - target["boxes"]: FloatTensor[N,4]
    - target["labels"]: LongTensor[N]
    - target["image_id"]: Tensor[1]
    - target["img_label_bin"]: LongTensor[1], 0=normal, 1=positive
    """

    def __init__(
        self,
        root: Path,
        split: str,
        manifest_csv: Optional[Path] = None,
        strict_missing: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.split_dir = self.root / split
        self.manifest = manifest_csv
        self.strict_missing = strict_missing

        if self.manifest is not None:
            rows = read_manifest(self.manifest, split)
        else:
            rows = []

        self.items: List[Dict[str, Path]] = []
        missing_pairs = 0

        if rows:
            for r in rows:
                stem = (r.get("stem") or r.get("file_stem") or "").strip()
                if not stem:
                    continue

                json_name = (r.get("json_name") or f"{stem}.json").strip()
                jp = self.split_dir / json_name

                if not jp.exists():
                    msg = f"[WARN] JSON not found: {jp} (split={split})"
                    if strict_missing:
                        raise FileNotFoundError(msg)
                    missing_pairs += 1
                    continue

                # Resolve image path
                img_path = None
                img_name_col = (r.get("img_name") or "").strip()
                if img_name_col:
                    cand = self.split_dir / img_name_col
                    if cand.exists():
                        img_path = cand

                if img_path is None:

                    with open(jp, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    img_name = meta.get("imagePath")
                    if isinstance(img_name, str) and img_name:
                        cand = self.split_dir / img_name
                        if cand.exists():
                            img_path = cand

                if img_path is None:

                    for ext in IMG_EXTS:
                        cand = self.split_dir / f"{stem}{ext}"
                        if cand.exists():
                            img_path = cand
                            break

                if img_path is None:
                    msg = f"[WARN] Image not found for JSON {jp} (split={split})"
                    if strict_missing:
                        raise FileNotFoundError(msg)
                    missing_pairs += 1
                    continue

                self.items.append({"json": jp, "img": img_path})
        else:
            # No manifest: use all JSONs in the split directory
            for jp in sorted(self.split_dir.glob("*.json")):
                with open(jp, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                img_name = meta.get("imagePath")
                img_path = None
                if isinstance(img_name, str) and img_name:
                    cand = self.split_dir / img_name
                    if cand.exists():
                        img_path = cand
                if img_path is None:
                    stem = jp.stem
                    for ext in IMG_EXTS:
                        cand = self.split_dir / f"{stem}{ext}"
                        if cand.exists():
                            img_path = cand
                            break
                if img_path is None:
                    if strict_missing:
                        raise FileNotFoundError(f"Image not found for JSON {jp}")
                    missing_pairs += 1
                    continue
                self.items.append({"json": jp, "img": img_path})

        if missing_pairs > 0:
            print(f"[INFO] {missing_pairs} JSON/image pairs missing in split '{split}' (skipped)")

        if not self.items:
            raise RuntimeError(f"No valid samples found in split '{split}'")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        rec = self.items[idx]
        with open(rec["json"], "r", encoding="utf-8") as f:
            meta = json.load(f)

        img = pil_load_rgb(rec["img"])

        # Parse LabelMe shapes
        shapes = meta.get("shapes", []) or []
        boxes_list: List[List[float]] = []
        labels_list: List[int] = []

        for sh in shapes:
            raw_label = str(sh.get("label", "")).strip().lower()
            if raw_label not in CLASS_TO_ID:
                continue
            pts = sh.get("points") or []
            if len(pts) < 2:
                continue
            (x1, y1), (x2, y2) = pts[0], pts[1]
            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
            x_min, y_min = min(x1, x2), min(y1, y2)
            x_max, y_max = max(x1, x2), max(y1, y2)
            boxes_list.append([x_min, y_min, x_max, y_max])
            labels_list.append(CLASS_TO_ID[raw_label])

        # Binary image label: 0 normal, 1 positive
        img_label_bin = 1 if len(labels_list) > 0 else 0

        if len(boxes_list) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes_list, dtype=torch.float32)
            labels = torch.as_tensor(labels_list, dtype=torch.int64)

        image_id = torch.tensor([idx])
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "img_label_bin": torch.tensor([img_label_bin], dtype=torch.int64),
        }

        img_t = TF.to_tensor(img)  # [0,1] float, CHW
        return img_t, target


# --------------------------
# Model
# --------------------------


def create_model(num_classes: int = 2, pretrained: bool = True) -> FasterRCNN:
    """Create Faster R-CNN with a ResNet50-FPN backbone.

    num_classes = 1 (background) + K (foreground classes). Here K=1 (sponge).
    """
    backbone = resnet_fpn_backbone("resnet50", pretrained=pretrained, trainable_layers=3)
    model = FasterRCNN(backbone, num_classes=num_classes)
    return model


# --------------------------
# Train / Eval
# --------------------------


@torch.no_grad()
def evaluate(model, loader, device, score_thr: float = 0.5, iou_thr: float = 0.5) -> Dict[str, float]:
    model.eval()
    preds = []
    gts = []
    y_bin_true: List[int] = []

    for images, targets in loader:
        images = [img.to(device) for img in images]
        outs = model(images)

        for o, t in zip(outs, targets):
            preds.append({
                "boxes": o["boxes"].detach().cpu(),
                "scores": o["scores"].detach().cpu(),
                "labels": o["labels"].detach().cpu(),
            })
            gts.append({
                "boxes": t["boxes"].detach().cpu(),
                "labels": t["labels"].detach().cpu(),
            })
            y_bin_true.append(int(t["img_label_bin"][0].item()))

    # Detection AP@0.5 for sponge (class 1)
    ap1, p1, r1, f1_1 = ap_iou50_single_class(preds, gts, class_id=1, score_thr=score_thr, iou_thr=iou_thr)
    mAP = ap1  # only one class

    # Binary image-level metrics
    y_bin_pred = bin_preds_from_dets(preds, score_thr=score_thr)
    bin_metrics = classification_metrics_binary(np.asarray(y_bin_true, dtype=np.int64), y_bin_pred)

    return {
        "AP50_sponge": ap1,
        "DetP_sponge": p1,
        "DetR_sponge": r1,
        "DetF1_sponge": f1_1,
        "mAP50": mAP,
        "BIN_acc": bin_metrics["acc"],
        "BIN_macroP": bin_metrics["macro_P"],
        "BIN_macroR": bin_metrics["macro_R"],
        "BIN_macroF1": bin_metrics["macro_F1"],
    }


def train_one_epoch(model, loader, optimizer, device, scheduler=None):
    model.train()
    loss_sums = {"loss": 0.0, "cls": 0.0, "box_reg": 0.0, "rpn_cls": 0.0, "rpn_bbox": 0.0}
    n = 0

    for step, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        # Faster R-CNN ignores image-level label; do not pass it to the model
        targets = [{k: v.to(device) for k, v in t.items() if k != "img_label_bin"} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad(set_to_none=True)
        losses.backward()
        optimizer.step()

        loss_sums["loss"] += float(losses.detach().cpu())
        loss_sums["cls"] += float(loss_dict["loss_classifier"].detach().cpu())
        loss_sums["box_reg"] += float(loss_dict["loss_box_reg"].detach().cpu())
        loss_sums["rpn_cls"] += float(loss_dict["loss_objectness"].detach().cpu())
        loss_sums["rpn_bbox"] += float(loss_dict["loss_rpn_box_reg"].detach().cpu())
        n += 1

    for k in loss_sums:
        loss_sums[k] /= max(n, 1)
    return loss_sums


# --------------------------
# Plot & Save curves
# --------------------------


def save_curves_npz_pdf(out_dir: Path, history: Dict[str, List[float]], pdf_name: str = "curves.pdf") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # npz
    np.savez_compressed(out_dir / "curves_hist.npz", **{k: np.asarray(v, dtype=np.float32) for k, v in history.items()})
    # pdf
    with PdfPages(out_dir / pdf_name) as pdf:
        plt.figure(figsize=(11, 10))

        plt.subplot(2, 2, 1)
        for k in ["train_loss", "train_cls", "train_box", "train_rpn_cls", "train_rpn_box"]:
            if k in history:
                plt.plot(history[k], label=k)
        plt.title("Training losses"); plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 2)
        for k in ["AP50_sponge", "mAP50"]:
            if k in history:
                plt.plot(history[k], label=k)
        plt.title("Detection AP@0.5"); plt.xlabel("epoch"); plt.ylabel("AP"); plt.legend(); plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 3)
        for k in ["BIN_acc", "BIN_macroF1"]:
            if k in history:
                plt.plot(history[k], label=k)
        plt.title("Binary classification"); plt.xlabel("epoch"); plt.ylabel("score"); plt.legend(); plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 4)
        if "lr" in history:
            plt.plot(history["lr"], label="lr")
        plt.title("Learning Rate"); plt.xlabel("epoch"); plt.ylabel("lr"); plt.legend(); plt.grid(True, alpha=0.3)

        plt.tight_layout(); pdf.savefig(); plt.close()


# --------------------------
# Main
# --------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Faster R-CNN on RFO sponge dataset using LabelMe JSON. "
            "Detect sponge (vs background) and report binary metrics (normal vs positive)."
        )
    )

    # Paths
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="Root containing train/ val/ test/ with LabelMe JSON and images.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./frcnn_sponge_runs",
        help="Output directory for checkpoints & curves.",
    )

    # Training setup
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--score-thr", type=float, default=0.5)
    parser.add_argument("--iou-thr", type=float, default=0.5)

    # Optimizer & scheduler
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adam", "adamw", "sgd"],
        help="Optimizer type (default: adamw).",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "plateau", "none"],
        help="LR scheduler (default: cosine).",
    )
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum LR for cosine/plateau.")
    parser.add_argument("--t0", type=int, default=10, help="T_0 (epochs) for CosineAnnealingWarmRestarts.")
    parser.add_argument("--t-mult", type=int, default=2, help="T_mult for CosineAnnealingWarmRestarts.")
    parser.add_argument("--plateau-factor", type=float, default=0.5, help="ReduceLROnPlateau factor.")
    parser.add_argument("--plateau-patience", type=int, default=3, help="ReduceLROnPlateau patience in epochs.")

    # Manifest selection & strict mode
    parser.add_argument(
        "--manifest-file",
        type=str,
        default=None,
        help="Full path to a manifest CSV. If set, overrides --manifest-name.",
    )
    parser.add_argument(
        "--manifest-name",
        type=str,
        default=None,
        help=(
            "CSV filename under <data-root>, e.g. 'split_manifest_500x3.csv'. "
            "If omitted and --manifest-file is also None, all JSONs in the split are used."
        ),
    )
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="If set, raise error when a CSV row's JSON or image is missing.",
    )

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    data_root = Path(args.data_root)

    # Resolve manifest path (file > name > none)
    manifest: Optional[Path]
    if args.manifest_file:
        manifest = Path(args.manifest_file)
    elif args.manifest_name:
        manifest = data_root / args.manifest_name
    else:
        manifest = None

    if manifest is not None and not manifest.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Datasets
    train_set = LabelmeSpongeDataset(data_root, "train", manifest_csv=manifest, strict_missing=args.strict_missing)
    val_set = LabelmeSpongeDataset(data_root, "val", manifest_csv=manifest, strict_missing=args.strict_missing)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    # Model: background + 1 foreground class -> num_classes=2
    model = create_model(num_classes=2, pretrained=True).to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    else:  # sgd
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    # LR Scheduler
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.t0, T_mult=args.t_mult, eta_min=args.min_lr
        )
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=args.plateau_factor, patience=args.plateau_patience, min_lr=args.min_lr
        )
    # else: none

    # History
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_cls": [],
        "train_box": [],
        "train_rpn_cls": [],
        "train_rpn_box": [],
        "AP50_sponge": [],
        "mAP50": [],
        "BIN_acc": [],
        "BIN_macroF1": [],
        "lr": [],
    }

    best_map = -1.0

    for epoch in range(1, args.epochs + 1):
        import time

        t0 = time.time()
        loss_dict = train_one_epoch(model, train_loader, optimizer, device, scheduler=scheduler)
        dt = time.time() - t0

        metrics = evaluate(model, val_loader, device, score_thr=args.score_thr, iou_thr=args.iou_thr)

        # Log history
        history["train_loss"].append(loss_dict["loss"])
        history["train_cls"].append(loss_dict["cls"])
        history["train_box"].append(loss_dict["box_reg"])
        history["train_rpn_cls"].append(loss_dict["rpn_cls"])
        history["train_rpn_box"].append(loss_dict["rpn_bbox"])

        history["AP50_sponge"].append(metrics["AP50_sponge"])
        history["mAP50"].append(metrics["mAP50"])
        history["BIN_acc"].append(metrics["BIN_acc"])
        history["BIN_macroF1"].append(metrics["BIN_macroF1"])

        lr_now = optimizer.param_groups[0]["lr"]
        history["lr"].append(lr_now)

        # Step LR scheduler after validation metrics are available
        if scheduler is not None:
            if args.scheduler == "cosine":
                # CosineAnnealingWarmRestarts stepped once per epoch
                scheduler.step()
            elif args.scheduler == "plateau":
                # ReduceLROnPlateau uses mAP as monitoring metric
                scheduler.step(metrics["mAP50"])

        # Save curves and history
        save_curves_npz_pdf(out_dir, history, pdf_name="curves.pdf")

        # Save checkpoints
        ckpt_last = out_dir / "last.pt"
        torch.save(
            {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "history": history},
            ckpt_last,
        )

        if metrics["mAP50"] > best_map:
            best_map = metrics["mAP50"]
            ckpt_best = out_dir / "best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "history": history,
                },
                ckpt_best,
            )

        print(
            f"[Epoch {epoch:03d}/{args.epochs}] "
            f"loss={loss_dict['loss']:.4f} (cls {loss_dict['cls']:.4f}, box {loss_dict['box_reg']:.4f}, "
            f"rpn_cls {loss_dict['rpn_cls']:.4f}, rpn_box {loss_dict['rpn_bbox']:.4f}) | "
            f"DetAP50: sponge={metrics['AP50_sponge']:.4f}, mAP={metrics['mAP50']:.4f} | "
            f"BIN acc={metrics['BIN_acc']:.4f}, BIN macroF1={metrics['BIN_macroF1']:.4f} | "
            f"lr={lr_now:.2e} | time={dt:.1f}s"
        )

    print(f"Done. Outputs are in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
