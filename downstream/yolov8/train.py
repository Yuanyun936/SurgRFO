#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch

# Hint to wandb/Ultralytics: disable WandB logging (best to also uninstall
# wandb from this environment if you don't need it).
os.environ.setdefault("WANDB_DISABLED", "true")

# Ultralytics YOLO
try:
    from ultralytics import YOLO
except ImportError as e:  # pragma: no cover - explicit message for missing dep
    raise ImportError(
        "ultralytics package is required. Install via 'pip install ultralytics'."
    ) from e


# --------------------------
# Config / constants
# --------------------------

CLASS_NAME = "sponge"  # single foreground class
IMG_EXTS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")


# --------------------------
# Utils
# --------------------------


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_manifest(manifest_csv: Path, split: str) -> List[Dict[str, str]]:
    """Read manifest CSV and keep only rows with the given split.

    Expected columns (minimal): split, stem
    Optional: json_name, img_name
    """
    rows: List[Dict[str, str]] = []
    with open(manifest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split") == split:
                rows.append(row)
    if not rows:
        raise RuntimeError(f"No rows found for split='{split}' in {manifest_csv}")
    return rows


def resolve_items_for_split(
    data_root: Path,
    split: str,
    manifest_csv: Optional[Path],
    strict_missing: bool = False,
) -> List[Dict[str, Path]]:
    """Resolve list of {"json": json_path, "img": img_path} for a split."""

    split_dir = data_root / split

    if manifest_csv is not None:
        rows = read_manifest(manifest_csv, split)
    else:
        rows = []

    items: List[Dict[str, Path]] = []
    missing_pairs = 0

    if rows:
        for r in rows:
            stem = (r.get("stem") or r.get("file_stem") or "").strip()
            if not stem:
                continue

            json_name = (r.get("json_name") or f"{stem}.json").strip()
            jp = split_dir / json_name

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
                cand = split_dir / img_name_col
                if cand.exists():
                    img_path = cand

            if img_path is None:
                with open(jp, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                img_name = meta.get("imagePath")
                if isinstance(img_name, str) and img_name:
                    cand = split_dir / img_name
                    if cand.exists():
                        img_path = cand

            if img_path is None:
                stem2 = stem
                for ext in IMG_EXTS:
                    cand = split_dir / f"{stem2}{ext}"
                    if cand.exists():
                        img_path = cand
                        break

            if img_path is None:
                msg = f"[WARN] Image not found for JSON {jp} (split={split})"
                if strict_missing:
                    raise FileNotFoundError(msg)
                missing_pairs += 1
                continue

            items.append({"json": jp, "img": img_path})
    else:
        for jp in sorted((data_root / split).glob("*.json")):
            with open(jp, "r", encoding="utf-8") as f:
                meta = json.load(f)
            img_name = meta.get("imagePath")
            img_path = None
            if isinstance(img_name, str) and img_name:
                cand = split_dir / img_name
                if cand.exists():
                    img_path = cand
            if img_path is None:
                stem = jp.stem
                for ext in IMG_EXTS:
                    cand = split_dir / f"{stem}{ext}"
                    if cand.exists():
                        img_path = cand
                        break
            if img_path is None:
                if strict_missing:
                    raise FileNotFoundError(f"Image not found for JSON {jp}")
                missing_pairs += 1
                continue
            items.append({"json": jp, "img": img_path})

    if missing_pairs > 0:
        print(f"[INFO] {missing_pairs} JSON/image pairs missing in split '{split}' (skipped)")

    if not items:
        raise RuntimeError(f"No valid samples found in split '{split}'")

    return items


# --------------------------
# LabelMe -> YOLO conversion
# --------------------------


def load_labelme_boxes(json_path: Path) -> Tuple[List[List[float]], List[int]]:
    """Return (boxes, labels) from a LabelMe JSON.

    YOLO class id is fixed to 0 for `sponge`.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    shapes = meta.get("shapes", []) or []
    boxes: List[List[float]] = []
    labels: List[int] = []

    for sh in shapes:
        raw_label = str(sh.get("label", "")).strip().lower()
        if raw_label != CLASS_NAME:
            continue
        pts = sh.get("points") or []
        if len(pts) < 2:
            continue
        (x1, y1), (x2, y2) = pts[0], pts[1]
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        x_min, y_min = min(x1, x2), min(y1, y2)
        x_max, y_max = max(x1, x2), max(y1, y2)
        boxes.append([x_min, y_min, x_max, y_max])
        labels.append(0)  # YOLO class id 0 -> sponge

    return boxes, labels


def to_yolo_xywh_norm(
    boxes_xyxy: List[List[float]],
    img_w: int,
    img_h: int,
) -> List[Tuple[float, float, float, float]]:
    """Convert [x1,y1,x2,y2] (pixels) to normalized [cx,cy,w,h] in [0,1]."""
    out: List[Tuple[float, float, float, float]] = []
    for x1, y1, x2, y2 in boxes_xyxy:
        w = max(x2 - x1, 1e-6)
        h = max(y2 - y1, 1e-6)
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        out.append((cx / img_w, cy / img_h, w / img_w, h / img_h))
    return out


def symlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare_yolo_split(
    items: List[Dict[str, Path]],
    split: str,
    yolo_root: Path,
) -> None:
    """Create YOLO-style `images/` and `labels/` for a split under yolo_root."""

    img_out_dir = yolo_root / split / "images"
    lbl_out_dir = yolo_root / split / "labels"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    for rec in items:
        img_path = rec["img"]
        json_path = rec["json"]
        stem = img_path.stem

        # symlink / copy image
        img_dst = img_out_dir / img_path.name
        symlink_or_copy(img_path, img_dst)

        # read boxes from LabelMe
        boxes, labels = load_labelme_boxes(json_path)

        # image size
        with Image.open(str(img_path)) as im:
            w, h = im.size

        yolo_boxes = to_yolo_xywh_norm(boxes, w, h)

        # write label file
        lbl_path = lbl_out_dir / f"{stem}.txt"
        with open(lbl_path, "w", encoding="utf-8") as f:
            for (cx, cy, bw, bh), cls_id in zip(yolo_boxes, labels):
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        # if no boxes: leave empty file (or no file); here we create empty file


def prepare_yolo_dataset(
    data_root: Path,
    manifest_csv: Optional[Path],
    strict_missing: bool,
    yolo_root: Path,
    include_test: bool = False,
) -> Path:
    """Convert LabelMe dataset (train/val[/test]) to YOLO format under yolo_root.

    Returns path to generated data.yaml.
    """

    yolo_root.mkdir(parents=True, exist_ok=True)

    splits = ["train", "val"]
    if include_test:
        splits.append("test")

    for split in splits:
        print(f"[YOLO] Preparing split '{split}' ...")
        items = resolve_items_for_split(data_root, split, manifest_csv, strict_missing)
        prepare_yolo_split(items, split, yolo_root)

    # Write data.yaml
    data_yaml = yolo_root / "data.yaml"
    with open(data_yaml, "w", encoding="utf-8") as f:
        f.write(
            "# Auto-generated YOLO dataset config for RFO sponge (single-class)\n"
        )
        f.write(f"path: {yolo_root}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        if include_test:
            f.write("test: test/images\n")
        f.write("nc: 1\n")
        f.write("names: ['sponge']\n")

    return data_yaml


# --------------------------
# Curve plotting (PDF)
# --------------------------


def save_training_curves_pdf(run_dir: Path) -> None:
    """Save simple loss / mAP curves to a PDF, similar to other models.

    Expects Ultralytics-generated results.csv under run_dir.
    """

    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        print(f"[YOLO] results.csv not found at {csv_path}, skip PDF curves.")
        return

    epochs: List[int] = []
    train_box: List[float] = []
    train_cls: List[float] = []
    val_box: List[float] = []
    val_cls: List[float] = []
    map50: List[float] = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Ultralytics results.csv 有时在列名前面有空格，这里统一 strip
            key_map: Dict[str, str] = {}
            for k in row.keys():
                if isinstance(k, str):
                    key_map[k.strip()] = k

            def _raw(name: str) -> Optional[str]:
                col = key_map.get(name)
                if not col:
                    return None
                return row.get(col)

            # epoch 列
            raw_epoch = _raw("epoch")
            try:
                ep = int(raw_epoch) if raw_epoch not in (None, "") else None
            except ValueError:
                ep = None
            if ep is None:
                continue
            epochs.append(ep)

            def _get(name: str) -> Optional[float]:
                v = _raw(name)
                if v is None or v == "":
                    return None
                try:
                    return float(v)
                except ValueError:
                    return None

            train_box.append(_get("train/box_loss") or 0.0)
            train_cls.append(_get("train/cls_loss") or 0.0)
            val_box.append(_get("val/box_loss") or 0.0)
            val_cls.append(_get("val/cls_loss") or 0.0)
            # YOLOv8 metrics column names
            m = _get("metrics/mAP50(B)") or _get("metrics/mAP50") or 0.0
            map50.append(m)

    if not epochs:
        print("[YOLO] No epochs found in results.csv, skip PDF curves.")
        return

    pdf_path = run_dir / "curves_yolo.pdf"

    plt.figure(figsize=(10, 8))

    # Loss curves
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_box, label="train/box_loss")
    plt.plot(epochs, train_cls, label="train/cls_loss")
    plt.plot(epochs, val_box, label="val/box_loss")
    plt.plot(epochs, val_cls, label="val/cls_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("YOLOv8 loss curves")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # mAP@0.5 curve
    plt.subplot(2, 1, 2)
    plt.plot(epochs, map50, label="mAP@0.5 (YOLO metrics/mAP50(B))")
    plt.xlabel("epoch")
    plt.ylabel("mAP@0.5")
    plt.title("YOLOv8 validation mAP@0.5")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"[YOLO] Saved training curves PDF to {pdf_path}")


# --------------------------
# Main (training)
# --------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "YOLOv8 on RFO sponge dataset using LabelMe JSON (single-class detection)."
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
        default="./yolo_sponge_runs",
        help="Output directory for YOLO training runs (Ultralytics 'project').",
    )
    parser.add_argument(
        "--yolo-ds-root",
        type=str,
        default=None,
        help=(
            "Optional path to store converted YOLO dataset. "
            "If omitted, defaults to <out-dir>/yolo_dataset."
        ),
    )

    # Training setup
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    # Optimizer / scheduler & dataloader settings (mapped to Ultralytics args)
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Base learning rate (maps to Ultralytics lr0)",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=None,
        help=(
            "Final learning rate at the end of training. "
            "If set, lrf = min_lr / lr; if scheduler='none', lrf=1.0 (no decay)."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of dataloader workers (Ultralytics 'workers')",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="SGD",
        help="Ultralytics optimizer name, e.g. 'SGD', 'Adam', 'AdamW'",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cos",
        choices=["cos", "none"],
        help=(
            "LR scheduler: 'cos' uses Ultralytics cosine schedule (default), "
            "'none' keeps LR constant (lrf=1.0). t0/t-mult from RetinaNet are not used here."
        ),
    )

    # Eval thresholds (used by Ultralytics during validation metrics)
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.25,
        help="Confidence threshold for metrics (Ultralytics 'conf')",
    )
    parser.add_argument(
        "--iou-thr",
        type=float,
        default=0.5,
        help="IoU threshold for metrics (Ultralytics 'iou')",
    )

    # Optimizer / scheduler etc. are controlled via Ultralytics args
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8s.pt",
        help=(
            "YOLOv8 model name or path, e.g. 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', "
            "or a custom .pt checkpoint."
        ),
    )

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
            "CSV filename under <data-root>, e.g. 'split_manifest_5000x2.csv'. "
            "If omitted and --manifest-file is also None, all JSONs in each split are used."
        ),
    )
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="If set, raise error when a CSV row's JSON or image is missing.",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default="yolo_sponge",
        help="Run name under out-dir/project (Ultralytics 'name').",
    )

    args = parser.parse_args()

    set_seed(args.seed)

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

    if args.yolo_ds_root is not None:
        yolo_ds_root = Path(args.yolo_ds_root)
    else:
        yolo_ds_root = out_dir / "yolo_dataset"

    # Prepare YOLO dataset (train/val only for training)
    data_yaml = prepare_yolo_dataset(
        data_root=data_root,
        manifest_csv=manifest,
        strict_missing=args.strict_missing,
        yolo_root=yolo_ds_root,
        include_test=False,
    )

    # Train YOLOv8
    print(f"[YOLO] Using dataset YAML: {data_yaml}")
    print(f"[YOLO] Loading model: {args.model}")

    model = YOLO(args.model)

    # Map custom CLI args to Ultralytics training parameters
    if args.scheduler == "none":
        # keep LR constant
        lrf = 1.0
    else:
        if args.min_lr is not None and args.min_lr > 0:
            lrf = max(args.min_lr / args.lr, 0.0)
        else:
            # fall back to Ultralytics default
            lrf = 0.01

    # Ultralytics will create out_dir / args.run_name
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        device=args.device,
        project=str(out_dir),
        name=args.run_name,
        exist_ok=True,
        # optimizer & scheduler
        lr0=args.lr,
        lrf=lrf,
        optimizer=args.optimizer,
        cos_lr=(args.scheduler == "cos"),
        # dataloader
        workers=args.num_workers,
        # metric thresholds
        conf=args.score_thr,
        iou=args.iou_thr,
    )

    print("[YOLO] Training finished.")
    print(f"Results project dir: {out_dir.resolve()}")
    print(f"Run name: {args.run_name}")

    # Save curves PDF similar to Faster R-CNN / RetinaNet
    run_dir = out_dir / args.run_name
    save_training_curves_pdf(run_dir)


if __name__ == "__main__":
    main()
