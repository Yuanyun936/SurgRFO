#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLOv8 testing / evaluation on RFO sponge dataset (LabelMe JSON).
Requires `ultralytics` and `matplotlib`.
"""

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

try:
    from ultralytics import YOLO
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "ultralytics package is required. Install via 'pip install ultralytics'."
    ) from e

# Import helpers from train.py in the same folder
import sys
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from train import (  # type: ignore
    CLASS_NAME,
    prepare_yolo_dataset,
    resolve_items_for_split,
)


# --------------------------
# Metrics helpers (mirroring fasterrcnn/test.py style)
# --------------------------

MIN_BOX_W = 1.0
MIN_BOX_H = 1.0


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
    area_b = (bx2 - bx1) * (by2 - by1)

    union = area_a + area_b - inter
    return inter / np.clip(union, 1e-6, None)


def ap_iou50_single_class(preds, gts, score_thr: float = 0.5, iou_thr: float = 0.5):
    """Compute AP@0.5 for one detection class; also return P/R/F1.

    preds, gts: lists of dicts {"boxes","scores","labels"} / {"boxes","labels"}.
    Single foreground class is assumed (label id = 1 in metrics space).
    """
    all_scores: List[float] = []
    all_matches: List[int] = []
    total_gt = 0
    tp = fp = fn = 0

    for pred, gt in zip(preds, gts):
        gt_boxes = gt["boxes"]
        gt_labels = gt["labels"]
        gt_mask = (gt_labels == 1)
        gt_boxes = gt_boxes[gt_mask]
        total_gt += len(gt_boxes)

        p_boxes = pred["boxes"]
        p_scores = pred["scores"]
        p_labels = pred["labels"]
        keep = (p_labels == 1) & (p_scores >= score_thr)
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

            ious = iou_xyxy(p_boxes[i : i + 1], gt_boxes)[0]
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


def classification_metrics_binary(y_true, y_pred) -> Dict[str, float]:
    """Binary metrics (classes={0,1}): accuracy + macro P/R/F1."""
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
# GT helpers & extra evaluation utilities
# --------------------------


def gt_boxes_from_items(items: List[Dict[str, Path]]) -> Tuple[List[Dict[str, np.ndarray]], np.ndarray]:
    """Return per-image GT boxes/labels and binary image labels.

    - labels use id 1 for sponge (to match metrics).
    - binary label 1 if any sponge box exists, else 0.
    """
    gts: List[Dict[str, np.ndarray]] = []
    y_true_bin: List[int] = []

    for rec in items:
        jp = rec["json"]
        with open(jp, "r", encoding="utf-8") as f:
            meta = json.load(f)
        shapes = meta.get("shapes", []) or []
        boxes_list: List[List[float]] = []
        labels_list: List[int] = []
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
            boxes_list.append([x_min, y_min, x_max, y_max])
            labels_list.append(1)  # metrics space: 1 = sponge

        if boxes_list:
            boxes = np.asarray(boxes_list, dtype=np.float32)
            labels = np.asarray(labels_list, dtype=np.int64)
            y_true_bin.append(1)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
            y_true_bin.append(0)

        gts.append({"boxes": boxes, "labels": labels})

    return gts, np.asarray(y_true_bin, dtype=np.int64)


def image_level_labels_from_items(items: List[Dict[str, Path]]) -> np.ndarray:
    """Binary GT per image: 1 if any sponge box exists in JSON, else 0."""
    _gts, y_true_bin = gt_boxes_from_items(items)
    return y_true_bin


def save_jsonl(preds: List[Dict], metas: List[Dict], out_jsonl: Path) -> None:
    """Save raw detections as JSONL (one line per image)."""

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for p, m in zip(preds, metas):
            row = {
                "stem": m["stem"],
                "img_path": m["img_path"],
                "boxes": p["boxes"].tolist(),
                "scores": p["scores"].tolist(),
                "labels": p["labels"].astype(int).tolist(),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_flat_csv(preds: List[Dict], metas: List[Dict], out_csv: Path, score_thr: float) -> None:
    """Flattened CSV of detections (one row per kept box)."""

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stem", "img_path", "x1", "y1", "x2", "y2", "score", "label_id", "label_name"])
        for p, m in zip(preds, metas):
            boxes = p["boxes"]
            scores = p["scores"]
            labels = p["labels"]
            for b, s, l in zip(boxes, scores, labels):
                if s < score_thr:
                    continue
                x1, y1, x2, y2 = b.tolist()
                lid = int(l)
                lname = CLASS_NAME if lid == 1 else f"cls{lid}"
                w.writerow([m["stem"], m["img_path"], x1, y1, x2, y2, float(s), lid, lname])


# --------------------------
# Inference & evaluation
# --------------------------


def run_inference_yolo(
    model: YOLO,
    items: List[Dict[str, Path]],
    device: str,
    score_thr: float,
) -> Tuple[List[Dict], List[Dict]]:
    """Run YOLO on image paths and collect predictions + metadata.

    Returns:
      preds: list of {"boxes","scores","labels"} (numpy arrays)
      metas: list of {"img_path","stem"}
    """

    preds: List[Dict] = []
    metas: List[Dict] = []

    # Ultralytics handles device selection internally via model.to(device)
    model.to(device)

    for rec in items:
        img_path = rec["img"]
        stem = img_path.stem

        # Disable Ultralytics' per-image verbose printing during inference
        results = model(str(img_path), verbose=False)  # list with a single result
        if not results:
            boxes = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
        else:
            r = results[0]
            b = r.boxes
            if b is None or b.xyxy is None or b.conf is None or b.cls is None:
                boxes = np.zeros((0, 4), dtype=np.float32)
                scores = np.zeros((0,), dtype=np.float32)
                labels = np.zeros((0,), dtype=np.int64)
            else:
                boxes = b.xyxy.cpu().numpy().astype(np.float32)
                scores = b.conf.cpu().numpy().astype(np.float32)
                # YOLO class id 0 -> sponge -> metrics label 1
                labels = (b.cls.cpu().numpy().astype(np.int64) + 1)

        # Filter tiny boxes
        if boxes.size > 0:
            ws = boxes[:, 2] - boxes[:, 0]
            hs = boxes[:, 3] - boxes[:, 1]
            keep = (ws >= MIN_BOX_W) & (hs >= MIN_BOX_H)
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

        preds.append({"boxes": boxes, "scores": scores, "labels": labels})
        metas.append({"img_path": str(img_path), "stem": stem})

    return preds, metas


def bin_scores_from_preds(preds: List[Dict], score_thr: float) -> Tuple[np.ndarray, np.ndarray]:
    """From detection preds, derive per-image scores & binary predictions.

    - y_score: max score of sponge (label=1), or 0 if none
    - y_pred : 1 if y_score >= score_thr, else 0
    """

    y_score_list: List[float] = []
    y_pred_list: List[int] = []

    for p in preds:
        scores = p["scores"]
        labels = p["labels"]
        sel = (labels == 1)
        if np.any(sel):
            smax = float(scores[sel].max())
            y_score_list.append(smax)
            y_pred_list.append(1 if smax >= score_thr else 0)
        else:
            y_score_list.append(0.0)
            y_pred_list.append(0)

    return (
        np.asarray(y_score_list, dtype=np.float32),
        np.asarray(y_pred_list, dtype=np.int64),
    )


def confusion_from_preds(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return tp, fp, fn, tn


def medical_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    """Medical-style metrics emphasizing recall (sensitivity)."""
    P = tp + fn
    N = tn + fp
    tpr = tp / max(P, 1)  # Sensitivity / Recall
    tnr = tn / max(N, 1)  # Specificity
    ppv = tp / max(tp + fp, 1)  # Precision / PPV
    npv = tn / max(tn + fn, 1)  # NPV
    f1 = (2 * tp) / max(2 * tp + fp + fn, 1)
    f2 = (5 * tp) / max(5 * tp + 4 * fn + fp, 1)
    f05 = (1.25 * tp) / max(1.25 * tp + 0.25 * fn + fp, 1)
    bal_acc = 0.5 * (tpr + tnr)
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom > 0:
        denom = math.sqrt(denom)
        mcc = ((tp * tn - fp * fn) / denom)
    else:
        mcc = 0.0
    fdr = fp / max(tp + fp, 1)
    _for = fn / max(tn + fn, 1)
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Sensitivity": tpr,
        "Specificity": tnr,
        "PPV": ppv,
        "NPV": npv,
        "F1": f1,
        "F2": f2,
        "F0_5": f05,
        "BalancedAcc": bal_acc,
        "MCC": mcc,
        "FDR": fdr,
        "FOR": _for,
    }


def roc_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """Compute ROC AUC for binary labels."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float32)

    pos = int(np.sum(y_true == 1))
    neg = int(np.sum(y_true == 0))
    if pos == 0 or neg == 0:
        return None

    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]

    tps = np.cumsum(y_true_sorted == 1)
    fps = np.cumsum(y_true_sorted == 0)

    tpr = tps / float(pos)
    fpr = fps / float(neg)

    tpr = np.concatenate(([0.0], tpr))
    fpr = np.concatenate(([0.0], fpr))

    auc = float(np.trapz(tpr, fpr))
    return auc


def froc_curve(
    preds: List[Dict],
    gts: List[Dict],
    iou_thr: float = 0.5,
    class_id: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute FROC curve (FPPI vs sensitivity) for a single detection class.

    Returns (fp_per_image, sensitivity, score_thresholds).
    """

    scores_all: List[float] = []
    is_tp_all: List[int] = []
    total_gt = 0

    for pred, gt in zip(preds, gts):
        gt_boxes = gt["boxes"]
        gt_labels = gt["labels"]
        gt_mask = (gt_labels == class_id)
        gt_boxes = gt_boxes[gt_mask]
        total_gt += len(gt_boxes)

        p_boxes = pred["boxes"]
        p_scores = pred["scores"]
        p_labels = pred["labels"]
        keep = (p_labels == class_id)
        p_boxes = p_boxes[keep]
        p_scores = p_scores[keep]

        if p_scores.size == 0:
            continue

        order = np.argsort(-p_scores)
        p_boxes = p_boxes[order]
        p_scores = p_scores[order]

        matched_gt = np.zeros(len(gt_boxes), dtype=bool)
        for i in range(len(p_boxes)):
            if len(gt_boxes) == 0:
                scores_all.append(float(p_scores[i]))
                is_tp_all.append(0)
                continue
            ious = iou_xyxy(p_boxes[i : i + 1], gt_boxes)[0]
            j = int(np.argmax(ious))
            if ious[j] >= iou_thr and not matched_gt[j]:
                matched_gt[j] = True
                is_tp_all.append(1)
            else:
                is_tp_all.append(0)
            scores_all.append(float(p_scores[i]))

    num_images = len(gts)
    if total_gt == 0 or len(scores_all) == 0 or num_images == 0:
        return (
            np.asarray([], dtype=np.float32),
            np.asarray([], dtype=np.float32),
            np.asarray([], dtype=np.float32),
        )

    scores = np.asarray(scores_all, dtype=np.float32)
    is_tp = np.asarray(is_tp_all, dtype=np.int64)
    order = np.argsort(-scores)
    scores = scores[order]
    is_tp = is_tp[order]

    cum_tp = np.cumsum(is_tp)
    cum_fp = np.cumsum(1 - is_tp)

    sensitivity = cum_tp / float(max(total_gt, 1))
    fp_per_image = cum_fp / float(num_images)

    return fp_per_image, sensitivity, scores


def froc_summary(
    fp_per_image: np.ndarray,
    sensitivity: np.ndarray,
    fp_targets: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0),
) -> Dict[str, float]:
    """Summarize FROC curve as sensitivity at several FPPI targets."""

    summary: Dict[str, float] = {}
    if fp_per_image.size == 0 or sensitivity.size == 0:
        return summary

    for fp_t in fp_targets:
        mask = fp_per_image <= (fp_t + 1e-8)
        if not np.any(mask):
            sens = 0.0
        else:
            sens = float(sensitivity[mask].max())
        summary[f"Sens@{fp_t:g}FPPI"] = sens

    return summary


def sweep_for_sens_100(y_true: np.ndarray, y_score: np.ndarray, num_steps: int = 1001):
    """Sweep threshold to find highest thr with Sensitivity=100% (FN=0)."""

    if np.sum(y_true == 1) == 0:
        return None, (0, 0, 0, 0)

    best_thr: Optional[float] = None
    best_fp = 10 ** 9
    best_tuple: Tuple[int, int, int, int] = (0, 0, 0, 0)

    for t in np.linspace(1.0, 0.0, num_steps):
        y_pred = (y_score >= t).astype(np.int64)
        tp, fp, fn, tn = confusion_from_preds(y_true, y_pred)
        if fn == 0:  # Sensitivity = 100%
            if (fp < best_fp) or (fp == best_fp and (best_thr is None or t > best_thr)):
                best_thr = float(t)
                best_fp = fp
                best_tuple = (tp, fp, fn, tn)

    return best_thr, best_tuple


# --------------------------
# Visualization helpers
# --------------------------


def plot_confusion_matrix(tp: int, fp: int, fn: int, tn: int, out_path: Path, title: str) -> None:
    cm = np.array([[tp, fn], [fp, tn]], dtype=np.int64)
    labels = [["TP", "FN"], ["FP", "TN"]]

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Positive", "Negative"])
    ax.set_yticklabels(["Positive", "Negative"])

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(
            j,
            i,
            f"{labels[i][j]}\n{cm[i, j]}",
            ha="center",
            va="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=12,
            fontweight="bold",
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def draw_detections(img_path: Path, boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray,
                    save_path: Path, score_thr: float) -> None:
    img = Image.open(str(img_path)).convert("RGB")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    ax.axis("off")

    for b, s, l in zip(boxes, scores, labels):
        if s < score_thr:
            continue
        x1, y1, x2, y2 = b.tolist()
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x1,
            y1 - 3,
            f"sponge {s:.2f}",
            color="yellow",
            fontsize=12,
            bbox=dict(facecolor="black", alpha=0.5, pad=2),
        )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def draw_gt_boxes(img_path: Path, gt_boxes: np.ndarray, save_path: Path) -> None:
    """Visualize ground-truth sponge boxes only (for FN cases)."""

    img = Image.open(str(img_path)).convert("RGB")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    ax.axis("off")

    for b in gt_boxes:
        x1, y1, x2, y2 = b.tolist()
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="g",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x1,
            y1 - 3,
            "GT sponge",
            color="yellow",
            fontsize=12,
            bbox=dict(facecolor="black", alpha=0.5, pad=2),
        )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def maybe_save_visualizations(
    preds: List[Dict],
    metas: List[Dict],
    out_dir: Path,
    score_thr: float,
    max_vis: int,
) -> None:
    """General visualizations of detections (similar to Faster R-CNN script)."""

    vis_dir = out_dir / "vis"
    cnt = 0
    for p, m in zip(preds, metas):
        if cnt >= max_vis:
            break
        img_path = Path(m["img_path"])
        stem = m["stem"]
        save_path = vis_dir / f"{stem}.jpg"
        draw_detections(img_path, p["boxes"], p["scores"], p["labels"], save_path, score_thr)
        cnt += 1


def save_fp_fn_visualizations(
    preds: List[Dict],
    metas: List[Dict],
    y_true: np.ndarray,
    y_pred_img: np.ndarray,
    gts: List[Dict],
    out_dir: Path,
    score_thr: float,
    max_vis_fp: int,
    max_vis_fn: int,
) -> None:
    """Save visualizations for misclassified images.

    - False positives (FP): y_true=0, y_pred=1, show predicted sponge boxes (score>=thr)
    - False negatives (FN): y_true=1, y_pred=0, show ground-truth sponge boxes
    """

    fp_dir = out_dir / "vis_fp"
    fn_dir = out_dir / "vis_fn"
    fp_cnt = 0
    fn_cnt = 0

    for idx, (p, m) in enumerate(zip(preds, metas)):
        if idx >= len(y_true) or idx >= len(y_pred_img) or idx >= len(gts):
            break

        yt = int(y_true[idx])
        yp = int(y_pred_img[idx])
        img_path = Path(m["img_path"])
        stem = m["stem"]

        # False positive: predicted positive, GT negative
        if yt == 0 and yp == 1 and fp_cnt < max_vis_fp:
            labels = p["labels"]
            scores = p["scores"]
            boxes = p["boxes"]
            sel = (labels == 1) & (scores >= score_thr)
            if np.any(sel):
                save_path = fp_dir / f"{stem}.jpg"
                draw_detections(img_path, boxes[sel], scores[sel], labels[sel], save_path, score_thr=0.0)
                fp_cnt += 1

        # False negative: predicted negative, GT positive
        if yt == 1 and yp == 0 and fn_cnt < max_vis_fn:
            gt_boxes = gts[idx]["boxes"]
            if gt_boxes.size > 0:
                save_path = fn_dir / f"{stem}.jpg"
                draw_gt_boxes(img_path, gt_boxes, save_path)
                fn_cnt += 1


# --------------------------
# Main
# --------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test YOLOv8 on RFO sponge dataset using LabelMe JSON "
            "(single-class detection + binary image-level metrics)."
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
        "--ckpt",
        type=str,
        required=True,
        help="Path to YOLO checkpoint (.pt) produced by Ultralytics training.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./yolo_sponge_test_out",
        help="Directory to save evaluation metrics & visualizations.",
    )

    # Which split to evaluate
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate (default: test).",
    )

    parser.add_argument("--device", type=str, default="cuda")

    # Metrics thresholds
    parser.add_argument("--score-thr", type=float, default=0.5)
    parser.add_argument("--iou-thr", type=float, default=0.5)

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
            "CSV filename under <data-root>, e.g. 'split_manifest_3000x2.csv'. "
            "If omitted and --manifest-file is also None, all JSONs in the split are used."
        ),
    )
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="If set, raise error when a CSV row's JSON or image is missing.",
    )

    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="If set, save visualization images with detections overlaid.",
    )
    parser.add_argument(
        "--max-vis",
        type=int,
        default=200,
        help="Maximum number of images to visualize if --save-vis is set.",
    )

    args = parser.parse_args()

    device = args.device
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

    # Build list of items for the requested split
    items = resolve_items_for_split(
        data_root=data_root,
        split=args.split,
        manifest_csv=manifest,
        strict_missing=args.strict_missing,
    )

    # GT boxes & binary labels
    gts, y_true_bin = gt_boxes_from_items(items)

    # Load YOLO model
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"[YOLO] Loading checkpoint: {ckpt_path}")

    model = YOLO(str(ckpt_path))

    # Inference
    preds, metas = run_inference_yolo(
        model=model,
        items=items,
        device=device,
        score_thr=args.score_thr,
    )

    # Detection metrics
    ap1, p1, r1, f1_1 = ap_iou50_single_class(
        preds,
        gts,
        score_thr=args.score_thr,
        iou_thr=args.iou_thr,
    )
    mAP = ap1

    # Image-level metrics
    y_score, y_pred_bin = bin_scores_from_preds(preds, score_thr=args.score_thr)
    bin_metrics = classification_metrics_binary(y_true_bin, y_pred_bin)

    print(
        f"[SPLIT={args.split}] AP@IoU={args.iou_thr:.2f}_sponge={ap1:.4f}, "
        f"mAP@IoU={args.iou_thr:.2f}={mAP:.4f} | "
        f"DetP/DetR/DetF1={p1:.3f}/{r1:.3f}/{f1_1:.3f} | "
        f"BIN acc={bin_metrics['acc']:.4f}, BIN macroF1={bin_metrics['macro_F1']:.4f}"
    )

    # Save metrics to JSON
    metrics = {
        "AP50_sponge": float(ap1),
        "DetP_sponge": float(p1),
        "DetR_sponge": float(r1),
        "DetF1_sponge": float(f1_1),
        "mAP50": float(mAP),
        "BIN_acc": float(bin_metrics["acc"]),
        "BIN_macroP": float(bin_metrics["macro_P"]),
        "BIN_macroR": float(bin_metrics["macro_R"]),
        "BIN_macroF1": float(bin_metrics["macro_F1"]),
    }

    metrics_path = out_dir / f"metrics_{args.split}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "split": args.split,
                "ckpt": str(ckpt_path),
                "score_thr": float(args.score_thr),
                "iou_thr": float(args.iou_thr),
                "metrics": metrics,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved metrics JSON to: {metrics_path}")
    # Additional metrics (ROC AUC, confusion, medical-style)
    auc_roc = roc_auc_binary(y_true_bin, y_score)
    tp, fp, fn, tn = confusion_from_preds(y_true_bin, y_pred_bin)
    med_metrics = medical_metrics(tp, fp, fn, tn)

    print(
        f"[IMAGE-LEVEL @ thr={args.score_thr:.2f}] "
        f"Acc={bin_metrics['acc']:.4f}, MacroP={bin_metrics['macro_P']:.4f}, "
        f"MacroR={bin_metrics['macro_R']:.4f}, MacroF1={bin_metrics['macro_F1']:.4f} | "
        f"Med={med_metrics}"
    )

    if auc_roc is not None:
        print(f"[IMAGE-LEVEL ROC] AUC={auc_roc:.4f}")
    else:
        print("[IMAGE-LEVEL ROC] AUC undefined (need both positive and negative samples)")

    # Save raw detections
    save_jsonl(preds, metas, out_dir / "preds.jsonl")
    save_flat_csv(preds, metas, out_dir / "preds_flat.csv", score_thr=args.score_thr)

    # Confusion matrix at fixed threshold
    cm_png = out_dir / f"confusion_matrix_thr{args.score_thr:.2f}.png"
    plot_confusion_matrix(tp, fp, fn, tn, cm_png, title=f"Confusion Matrix (thr={args.score_thr:.2f})")
    print(f"Saved confusion matrix to: {cm_png}")

    # FROC (detection-level, sponge class)
    fp_per_image, sens_froc, thr_froc = froc_curve(preds, gts, iou_thr=args.iou_thr, class_id=1)
    froc_summ = froc_summary(fp_per_image, sens_froc)

    if froc_summ:
        print(f"[FROC] Sensitivity at FPPI targets: {froc_summ}")
    else:
        print("[FROC] Curve undefined (no GT boxes or detections)")

    # Save full FROC curve to CSV (if available)
    if fp_per_image.size > 0:
        froc_csv = out_dir / "froc_curve.csv"
        with open(froc_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["threshold", "FP_per_image", "sensitivity"])
            for thr, fp_i, sens in zip(thr_froc, fp_per_image, sens_froc):
                w.writerow([float(thr), float(fp_i), float(sens)])

    # Threshold sweep for Sensitivity=100%
    best_thr, best_tuple = sweep_for_sens_100(y_true_bin, y_score, num_steps=1001)
    img_metrics_json = out_dir / "image_level_metrics.json"

    if best_thr is not None:
        btp, bfp, bfn, btn = best_tuple
        bmetrics = medical_metrics(btp, bfp, bfn, btn)
        # print(f"[IMAGE-LEVEL @ Sens=100%] thr*={best_thr:.4f} -> {bmetrics}")
        # plot_confusion_matrix(
        #     btp,
        #     bfp,
        #     bfn,
        #     btn,
        #     out_dir / "confusion_matrix_sens100.png",
        #     title=f"Confusion Matrix (Sensitivity=100%, thr={best_thr:.4f})",
        # )
        with open(img_metrics_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "threshold_fixed": float(args.score_thr),
                    "metrics_at_fixed_threshold": {
                        "Acc": bin_metrics["acc"],
                        "MacroP": bin_metrics["macro_P"],
                        "MacroR": bin_metrics["macro_R"],
                        "MacroF1": bin_metrics["macro_F1"],
                        **med_metrics,
                    },
                    "AUC_ROC": auc_roc,
                    "FROC_summary": froc_summ,
                    "threshold_sensitivity_100": float(best_thr),
                    "metrics_at_sensitivity_100": bmetrics,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
    else:
        print("[WARN] No positive samples or cannot achieve Sensitivity=100%. Skipping sweep.")
        with open(img_metrics_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "threshold_fixed": float(args.score_thr),
                    "metrics_at_fixed_threshold": {
                        "Acc": bin_metrics["acc"],
                        "MacroP": bin_metrics["macro_P"],
                        "MacroR": bin_metrics["macro_R"],
                        "MacroF1": bin_metrics["macro_F1"],
                        **med_metrics,
                    },
                    "AUC_ROC": auc_roc,
                    "FROC_summary": froc_summ,
                    "threshold_sensitivity_100": None,
                    "metrics_at_sensitivity_100": None,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    # Optional visualization
    if args.save_vis:
        maybe_save_visualizations(preds, metas, out_dir, score_thr=args.score_thr, max_vis=args.max_vis)

    # Always save error visualizations (FP/FN) into separate folders
    save_fp_fn_visualizations(
        preds,
        metas,
        y_true_bin,
        y_pred_bin,
        gts,
        out_dir,
        score_thr=args.score_thr,
        max_vis_fp=args.max_vis,
        max_vis_fn=args.max_vis,
    )

    # Small README summarizing outputs
    with open(out_dir / "README.txt", "w", encoding="utf-8") as f:
        f.write(
            "Files generated by yolo/test.py\n"
            "- metrics_<split>.json : detection + binary metrics (AP@0.5 etc.)\n"
            "- preds.jsonl          : one line per image (boxes/scores/labels) from YOLO\n"
            "- preds_flat.csv       : flattened rows (one row per kept detection)\n"
            "- confusion_matrix_thr*.png\n"
            "- confusion_matrix_sens100.png (if achievable)\n"
            "- image_level_metrics.json (fixed-threshold + Sens=100% point if available, AUC, FROC summary)\n"
            "- vis/ (optional, if --save-vis: general visualizations)\n"
            "- vis_fp/ (false-positive images; show predicted sponge boxes)\n"
            "- vis_fn/ (false-negative images; show ground-truth sponge boxes)\n"
            f"- Detection AP@IoU={args.iou_thr:.2f}: sponge={metrics['AP50_sponge']:.4f}, mAP={metrics['mAP50']:.4f}\n"
        )

    print(f"Done. Results saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
