#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


import sys
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from train import (
    ID_TO_CLASS,
    LabelmeSpongeDataset,
    classification_metrics_binary,
    collate_fn,
    create_model,
    evaluate,
    iou_xyxy,
    pil_load_rgb,
)

MIN_BOX_W = 1.0
MIN_BOX_H = 1.0


@torch.no_grad()
def run_inference(model, loader, device) -> Tuple[List[Dict], List[Dict]]:
    """Run model on the loader and collect raw detections + metadata.

    Returns:
      preds: list of {"boxes","scores","labels"} (numpy arrays)
      metas: list of {"img_path","stem"}
    """

    model.eval()
    preds: List[Dict] = []
    metas: List[Dict] = []

    ds = loader.dataset
    assert hasattr(ds, "items"), "Dataset must have 'items' list with 'json' and 'img'."

    base_idx = 0
    for images, _targets in loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for out in outputs:
            pb = out["boxes"].detach().cpu().numpy()
            ps = out["scores"].detach().cpu().numpy()
            pl = out["labels"].detach().cpu().numpy()


            if pb.size > 0:
                ws = pb[:, 2] - pb[:, 0]
                hs = pb[:, 3] - pb[:, 1]
                keep = (ws >= MIN_BOX_W) & (hs >= MIN_BOX_H)
                pb = pb[keep]
                ps = ps[keep]
                pl = pl[keep]

            preds.append({"boxes": pb, "scores": ps, "labels": pl})

        bsz = len(outputs)
        for i in range(bsz):
            rec = ds.items[base_idx + i]
            img_path = str(rec["img"])
            stem = Path(rec["json"]).stem
            metas.append({"img_path": img_path, "stem": stem})
        base_idx += bsz

    return preds, metas


def save_jsonl(preds: List[Dict], metas: List[Dict], out_jsonl: Path) -> None:
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
                lname = ID_TO_CLASS.get(lid, f"cls{lid}")
                w.writerow([m["stem"], m["img_path"], x1, y1, x2, y2, float(s), lid, lname])


def image_level_labels_from_dataset_items(ds) -> np.ndarray:
    """Binary GT per image: 1 if any sponge box exists in JSON, else 0."""
    y_true: List[int] = []
    for rec in ds.items:
        with open(rec["json"], "r", encoding="utf-8") as f:
            meta = json.load(f)
        shapes = meta.get("shapes", []) or []
        pos = False
        for sh in shapes:
            raw_label = str(sh.get("label", "")).strip().lower()
            if raw_label == "sponge":
                pos = True
                break
        y_true.append(1 if pos else 0)
    return np.asarray(y_true, dtype=np.int64)


def bin_scores_from_dets(preds: List[Dict], score_thr: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """From detections, derive per-image scores & binary predictions.

    - y_score: max score of class 1 (sponge), or 0 if none
    - y_pred : 1 if y_score >= score_thr, else 0
    - has_pos: 1 if any sponge detection exists (ignoring threshold)
    """

    y_score_list: List[float] = []
    y_pred_list: List[int] = []
    has_pos_list: List[int] = []

    for p in preds:
        scores = p["scores"]
        labels = p["labels"]
        sel = (labels == 1)
        if np.any(sel):
            smax = float(scores[sel].max())
            y_score_list.append(smax)
            y_pred_list.append(1 if smax >= score_thr else 0)
            has_pos_list.append(1)
        else:
            y_score_list.append(0.0)
            y_pred_list.append(0)
            has_pos_list.append(0)

    return (
        np.asarray(y_score_list, dtype=np.float32),
        np.asarray(y_pred_list, dtype=np.int64),
        np.asarray(has_pos_list, dtype=np.int64),
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

    # prepend (0,0) so the curve starts at origin
    tpr = np.concatenate(([0.0], tpr))
    fpr = np.concatenate(([0.0], fpr))

    auc = float(np.trapz(tpr, fpr))
    return auc


def gt_boxes_from_dataset_items(ds) -> List[Dict[str, np.ndarray]]:

    gts: List[Dict[str, np.ndarray]] = []
    for rec in ds.items:
        with open(rec["json"], "r", encoding="utf-8") as f:
            meta = json.load(f)
        shapes = meta.get("shapes", []) or []
        boxes_list: List[List[float]] = []
        labels_list: List[int] = []
        for sh in shapes:
            raw_label = str(sh.get("label", "")).strip().lower()
            if raw_label != "sponge":
                continue
            pts = sh.get("points") or []
            if len(pts) < 2:
                continue
            (x1, y1), (x2, y2) = pts[0], pts[1]
            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
            x_min, y_min = min(x1, x2), min(y1, y2)
            x_max, y_max = max(x1, x2), max(y1, y2)
            boxes_list.append([x_min, y_min, x_max, y_max])
            labels_list.append(1)

        if boxes_list:
            boxes = np.asarray(boxes_list, dtype=np.float32)
            labels = np.asarray(labels_list, dtype=np.int64)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)

        gts.append({"boxes": boxes, "labels": labels})

    return gts


def froc_curve(preds: List[Dict], gts: List[Dict], iou_thr: float = 0.5,
               class_id: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

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


def froc_summary(fp_per_image: np.ndarray, sensitivity: np.ndarray,
                 fp_targets: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)) -> Dict[str, float]:

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


def sweep_for_sens_100(y_true: np.ndarray, y_score: np.ndarray, num_steps: int = 1001):

    if np.sum(y_true == 1) == 0:
        return None, (0, 0, 0, 0)

    best_thr = None
    best_fp = 10 ** 9
    best_tuple = (0, 0, 0, 0)

    for t in np.linspace(1.0, 0.0, num_steps):
        y_pred = (y_score >= t).astype(np.int64)
        tp, fp, fn, tn = confusion_from_preds(y_true, y_pred)
        if fn == 0:  # Sensitivity = 100%
            if (fp < best_fp) or (fp == best_fp and (best_thr is None or t > best_thr)):
                best_thr = float(t)
                best_fp = fp
                best_tuple = (tp, fp, fn, tn)

    return best_thr, best_tuple


def draw_detections(img_path: Path, boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray,
                    save_path: Path, score_thr: float) -> None:
    img = pil_load_rgb(img_path)

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
        cls_name = ID_TO_CLASS.get(int(l), f"cls{int(l)}")
        ax.text(
            x1,
            y1 - 3,
            f"{cls_name} {s:.2f}",
            color="yellow",
            fontsize=12,
            bbox=dict(facecolor="black", alpha=0.5, pad=2),
        )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def maybe_save_visualizations(preds: List[Dict], metas: List[Dict], out_dir: Path, score_thr: float, max_vis: int) -> None:
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


def draw_gt_boxes(img_path: Path, gt_boxes: np.ndarray, save_path: Path) -> None:

    img = pil_load_rgb(img_path)

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test Faster R-CNN on RFO sponge dataset using LabelMe JSON "
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
        help="Path to checkpoint (best.pt or last.pt) produced by train.py.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./frcnn_sponge_test_out",
        help="Directory to save evaluation metrics JSON.",
    )

    # Which split to evaluate
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate (default: test).",
    )

    # Data loading
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    
    # Extra outputs (visualizations etc.)
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="If set, save visualization images with detections overlaid.",
    )
    parser.add_argument(
        "--max-vis",
        type=int,
        default=500,
        help="Maximum number of images to visualize if --save-vis is set.",
    )

    parser.add_argument("--score-thr", type=float, default=0.5)
    parser.add_argument("--iou-thr", type=float, default=0.5)

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
            "CSV filename under <data-root>. "
            "If omitted and --manifest-file is also None, all JSONs in the split are used."
        ),
    )
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="If set, raise error when a CSV row's JSON or image is missing.",
    )

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root)

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

    ds = LabelmeSpongeDataset(
        data_root,
        args.split,
        manifest_csv=manifest,
        strict_missing=args.strict_missing,
    )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    model = create_model(num_classes=2, pretrained=False).to(device)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        sd = state["model"]
    else:
        sd = state


    def _remap_fpn_rpn_keys(sd_in: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        sd_out: Dict[str, torch.Tensor] = {}
        for k, v in sd_in.items():
            new_k = k

            if new_k.startswith("backbone.fpn.inner_blocks.") or new_k.startswith("backbone.fpn.layer_blocks."):

                parts = new_k.split(".")
                # e.g. ["backbone","fpn","inner_blocks","0","0","weight"]
                if len(parts) >= 6 and parts[4] == "0":
                    parts.pop(4)  # remove the extra "0"
                    new_k = ".".join(parts)

            if new_k.startswith("rpn.head.conv."):
                parts = new_k.split(".")

                if len(parts) >= 6 and parts[3] == "0" and parts[4] == "0":

                    new_k = ".".join(parts[:3] + parts[5:])

            sd_out[new_k] = v
        return sd_out

    if isinstance(sd, dict):
        sd = _remap_fpn_rpn_keys(sd)

    model.load_state_dict(sd, strict=True)
    print(f"Loaded checkpoint from: {ckpt_path}")

    # Run evaluation (AP@IoU + binary metrics), reusing train.evaluate
    metrics = evaluate(
        model,
        loader,
        device,
        score_thr=args.score_thr,
        iou_thr=args.iou_thr,
    )

    print(
        f"[SPLIT={args.split}] AP@IoU={args.iou_thr:.2f}_sponge={metrics['AP50_sponge']:.4f}, "
        f"mAP@IoU={args.iou_thr:.2f}={metrics['mAP50']:.4f} | "
        f"DetP/DetR/DetF1={metrics['DetP_sponge']:.3f}/"
        f"{metrics['DetR_sponge']:.3f}/{metrics['DetF1_sponge']:.3f} | "
        f"BIN acc={metrics['BIN_acc']:.4f}, BIN macroF1={metrics['BIN_macroF1']:.4f}"
    )

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

    # --------------------------
    # Extra outputs: raw preds, confusion matrices, image-level metrics
    # --------------------------

    # Run inference once for extra outputs
    preds, metas = run_inference(model, loader, device)


    save_jsonl(preds, metas, out_dir / "preds.jsonl")
    save_flat_csv(preds, metas, out_dir / "preds_flat.csv", score_thr=args.score_thr)


    y_true = image_level_labels_from_dataset_items(ds)
    y_score, y_pred_img, _has_pos = bin_scores_from_dets(preds, score_thr=args.score_thr)

    # ROC AUC (image-level)
    auc_roc = roc_auc_binary(y_true, y_score)

    tp, fp, fn, tn = confusion_from_preds(y_true, y_pred_img)
    med_metrics = medical_metrics(tp, fp, fn, tn)
    cls_metrics = classification_metrics_binary(y_true, y_pred_img)

    print(
        f"[IMAGE-LEVEL @ thr={args.score_thr:.2f}] "
        f"Acc={cls_metrics['acc']:.4f}, MacroP={cls_metrics['macro_P']:.4f}, "
        f"MacroR={cls_metrics['macro_R']:.4f}, MacroF1={cls_metrics['macro_F1']:.4f} | "
        f"Med={med_metrics}"
    )

    if auc_roc is not None:
        print(f"[IMAGE-LEVEL ROC] AUC={auc_roc:.4f}")
    else:
        print("[IMAGE-LEVEL ROC] AUC undefined (need both positive and negative samples)")

    # FROC (detection-level, sponge class)
    gts = gt_boxes_from_dataset_items(ds)
    fp_per_image, sens_froc, thr_froc = froc_curve(preds, gts, iou_thr=args.iou_thr, class_id=1)
    froc_summ = froc_summary(fp_per_image, sens_froc)

    if froc_summ:
        print(f"[FROC] Sensitivity at FPPI targets: {froc_summ}")
    else:
        print("[FROC] Curve undefined (no GT boxes or detections)")


    if fp_per_image.size > 0:
        froc_csv = out_dir / "froc_curve.csv"
        with open(froc_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["threshold", "FP_per_image", "sensitivity"])
            for thr, fp_i, sens in zip(thr_froc, fp_per_image, sens_froc):
                w.writerow([float(thr), float(fp_i), float(sens)])


    cm_png = out_dir / f"confusion_matrix_thr{args.score_thr:.2f}.png"
    plot_confusion_matrix(tp, fp, fn, tn, cm_png, title=f"Confusion Matrix (thr={args.score_thr:.2f})")
    print(f"Saved confusion matrix to: {cm_png}")

    # Threshold sweep for Sensitivity=100%
    best_thr, best_tuple = sweep_for_sens_100(y_true, y_score, num_steps=1001)
    img_metrics_json = out_dir / "image_level_metrics.json"

    if best_thr is not None:
        btp, bfp, bfn, btn = best_tuple
        bmetrics = medical_metrics(btp, bfp, bfn, btn)

        with open(img_metrics_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "threshold_fixed": float(args.score_thr),
                    "metrics_at_fixed_threshold": {
                        "Acc": cls_metrics["acc"],
                        "MacroP": cls_metrics["macro_P"],
                        "MacroR": cls_metrics["macro_R"],
                        "MacroF1": cls_metrics["macro_F1"],
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
                        "Acc": cls_metrics["acc"],
                        "MacroP": cls_metrics["macro_P"],
                        "MacroR": cls_metrics["macro_R"],
                        "MacroF1": cls_metrics["macro_F1"],
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


    if args.save_vis:
        maybe_save_visualizations(preds, metas, out_dir, score_thr=args.score_thr, max_vis=args.max_vis)

    # Always save error visualizations (FP/FN) into separate folders
    save_fp_fn_visualizations(
        preds,
        metas,
        y_true,
        y_pred_img,
        gts,
        out_dir,
        score_thr=args.score_thr,
        max_vis_fp=args.max_vis,
        max_vis_fn=args.max_vis,
    )

    # Small README summarizing outputs
    with open(out_dir / "README.txt", "w", encoding="utf-8") as f:
        f.write(
            "Files generated by test.py\n"
            "- metrics_<split>.json : detection + binary metrics from evaluate()\n"
            "- preds.jsonl          : one line per image (boxes/scores/labels)\n"
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
