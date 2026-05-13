#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RFO patch maker: sponge -> 64x64, curved needle -> 16x16.
- Use mask .npy files for localization and cropping; read images from .jpg/.jpeg.
- Output four folders: sponge_image64, sponge_mask64, needle_image16, needle_mask16.
- Save both .npy files (image as float32 0..1, mask as uint8 0/1) and .png files (8-bit).
- Print crop size and resize ratio for each sample; generate crop_log.csv.
"""

from __future__ import annotations
import os, math, csv, argparse, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

# -------------------- Default paths (override via CLI) --------------------
DEF_MASK_DIR = "../data/critical_RFO_data/images_and_masks/mask512"
DEF_IMG_DIR  = "../data/critical_RFO_data/images_and_masks/image512"
DEF_CSV      = "../data/critical_RFO_data/images_and_masks/rfo_anno.CSV"
DEF_OUTROOT  = "../data/critical_RFO_data/images_and_masks"

# -------------------- Processing hyperparameters (tune as needed) --------------------
CFG = {
    # Target sizes
    # "sponge_target": 64,
    # "needle_target": 16,
    "sponge_target": 256,
    "needle_target": 64,


    # Expansion margin relative to the tight square bounding box
    "sponge_margin_ratio": 0.10,   # Sponge is larger, so a small margin is enough
    "needle_margin_ratio": 0.20,   # Needles are small and benefit from more context

    # Size safety band: constrain the resize ratio r = target / crop_side
    "r_max_up":   1.2,   # Do not upsample by more than 1.2x
    "r_min_down": 0.15,  # Do not downsample below 0.15 (~6.7x max downsampling)

    "dry_run": False,    # True prints only and does not write files
}

# -------------------- Utility functions --------------------
def load_npy(path: Path) -> np.ndarray:
    arr = np.load(str(path))
    if arr.ndim == 3 and arr.shape[0] in (1, 3):  # Support CHW
        arr = np.moveaxis(arr, 0, -1)
    return arr

def find_image_file(img_dir: Path, number_tag: str) -> Path | None:
    """Prefer matching .jpg/.jpeg files for number_Y, case-insensitive."""
    candidates = [
        f"{number_tag}.jpg", f"{number_tag}.jpeg",
        f"{number_tag}.JPG", f"{number_tag}.JPEG",
    ]
    for name in candidates:
        p = img_dir / name
        if p.exists():
            return p
    return None

def load_image_gray_uint8(img_path: Path) -> np.ndarray:
    """Read an image as grayscale uint8 (H, W)."""
    with Image.open(img_path) as im:
        im = im.convert("L")
        return np.array(im, dtype=np.uint8)

def to_uint8_png_mask(mask: np.ndarray) -> np.ndarray:
    """Convert a binary mask to 0/255."""
    m = (mask > 0).astype(np.uint8) * 255
    return m

def get_bbox_from_mask(mask: np.ndarray):
    """Return (ymin, xmin, ymax, xmax), or None if the mask is empty."""
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())

def square_from_bbox(ymin, xmin, ymax, xmax):
    """Expand a rectangle to a square centered on the original box."""
    h = ymax - ymin + 1
    w = xmax - xmin + 1
    side = max(h, w)
    cy = (ymin + ymax) / 2.0
    cx = (xmin + xmax) / 2.0
    return cy, cx, int(side)

def clamp_square_in_image(cy, cx, side, H, W):
    """Clamp a square centered at (cy, cx) into the image and return integer bounds."""
    half = side / 2.0
    y1 = int(round(cy - half))
    x1 = int(round(cx - half))
    y1 = max(0, min(y1, H - side))
    x1 = max(0, min(x1, W - side))
    y2 = y1 + side - 1
    x2 = x1 + side - 1
    return y1, x1, y2, x2

def choose_crop_side(base_side: int, target: int, margin_ratio: float, H: int, W: int,
                     r_max_up: float, r_min_down: float) -> int:
    """Choose the final crop side length from margin and resize constraints."""
    side_min = int(math.ceil(base_side * (1.0 + margin_ratio)))  # expand beyond the tight square
    # Limit upsampling: side >= target / r_max_up
    side_min = max(side_min, int(math.ceil(target / max(r_max_up, 1e-6))))
    # Limit downsampling: side <= target / r_min_down
    side_max_pref = int(math.floor(target / max(r_min_down, 1e-6)))
    side_max = min(side_max_pref, min(H, W))

    if side_min <= side_max:
        side = side_min
    else:
        side = min(side_min, min(H, W))  # If infeasible, prioritize covering the RFO
    side = max(1, min(side, min(H, W)))
    return int(side)

def pil_resize(arr: np.ndarray, size: int, is_mask: bool) -> np.ndarray:
    """Resize a square array. Image: LANCZOS for downsampling, BICUBIC for upsampling; mask: NEAREST."""
    H, W = arr.shape[:2]
    upsample = size > max(H, W)
    resample = Image.NEAREST if is_mask else (Image.BICUBIC if upsample else Image.LANCZOS)
    img = Image.fromarray(arr)
    out = img.resize((size, size), resample=resample)
    return np.array(out)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# -------------------- Main pipeline --------------------
def process(args):
    mask_dir = Path(args.mask_dir)
    img_dir  = Path(args.img_dir)
    csv_path = Path(args.csv)
    outroot  = Path(args.out_root)

    # Output folders
    sponge_mask_out = outroot / "sponge_mask64"
    sponge_img_out  = outroot / "sponge_image64"
    needle_mask_out = outroot / "needle_mask16"
    needle_img_out  = outroot / "needle_image16"
    for p in [sponge_mask_out, sponge_img_out, needle_mask_out, needle_img_out]:
        ensure_dir(p)

    df = pd.read_csv(csv_path)
    cols = {c.strip().lower(): c for c in df.columns}
    num_col = cols.get("number")
    lab_col = cols.get("labeling")
    if not num_col or not lab_col:
        raise ValueError("CSV must contain two columns: number and labeling")

    log_rows = []
    n_total = 0
    n_skipped = 0

    for _, row in df.iterrows():
        number = str(row[num_col]).strip()
        label  = str(row[lab_col]).strip().lower()

        for Y in (1, 2):
            number_tag = f"{number}_{Y}"

            mask_npy = mask_dir / f"{number_tag}.npy"
            img_path = find_image_file(img_dir, number_tag)

            if not mask_npy.exists() or img_path is None:
                warnings.warn(f"[Missing files] {number_tag}: could not find {mask_npy} or the matching .jpg/.jpeg file; skipping.")
                n_skipped += 1
                continue

            # Read mask (npy) and image (jpg -> grayscale uint8)
            mask = load_npy(mask_npy)
            if mask.ndim > 2:
                mask = mask[..., 0]
            img_u8 = load_image_gray_uint8(img_path)

            # If sizes differ, resample the image to match the mask size
            Hm, Wm = mask.shape
            Hi, Wi = img_u8.shape
            if (Hm, Wm) != (Hi, Wi):
                warnings.warn(f"[Size mismatch] {number_tag}: mask={mask.shape}, img={img_u8.shape} -> resampling image to mask size.")
                img_u8 = np.array(Image.fromarray(img_u8).resize((Wm, Hm), resample=Image.BICUBIC))

            bbox = get_bbox_from_mask(mask)
            if bbox is None:
                warnings.warn(f"[Empty mask] {number_tag}: mask is all zeros; skipping.")
                n_skipped += 1
                continue

            ymin, xmin, ymax, xmax = bbox
            cy, cx, base_side = square_from_bbox(ymin, xmin, ymax, xmax)

            # Class-specific settings
            if "sponge" in label:
                target = CFG["sponge_target"]
                margin_ratio = CFG["sponge_margin_ratio"]
                out_mask_dir, out_img_dir = sponge_mask_out, sponge_img_out
            elif "needle" in label:  # curved needle
                target = CFG["needle_target"]
                margin_ratio = CFG["needle_margin_ratio"]
                out_mask_dir, out_img_dir = needle_mask_out, needle_img_out
            else:
                warnings.warn(f"[Unknown label] {number_tag}: {label}; skipping.")
                n_skipped += 1
                continue

            side = choose_crop_side(
                base_side=base_side,
                target=target,
                margin_ratio=margin_ratio,
                H=Hm, W=Wm,
                r_max_up=CFG["r_max_up"],
                r_min_down=CFG["r_min_down"],
            )

            y1, x1, y2, x2 = clamp_square_in_image(cy, cx, side, Hm, Wm)
            crop_mask = mask[y1:y2+1, x1:x2+1]
            crop_img  = img_u8[y1:y2+1, x1:x2+1]

            r = target / side
            print(f"{number_tag} [{label}] crop_side={side:3d} -> target={target:2d}  resize_ratio={r:.3f}"
                + ("" if (r <= CFG['r_max_up'] and r >= CFG['r_min_down']) else "  [ratio out of range]"))

            # Resize: mask uses nearest neighbor; image uses LANCZOS/BICUBIC in pil_resize
            mask_bin  = (crop_mask > 0).astype(np.uint8)
            mask_res  = pil_resize(mask_bin * 255, target, is_mask=True) // 255  # 0/1
            img_res_u8 = pil_resize(crop_img, target, is_mask=False)             # uint8 0-255

            if not CFG["dry_run"]:
                # NPY: image stored as 0..1 float32; mask stored as 0/1 uint8
                np.save(out_mask_dir / f"{number_tag}.npy", mask_res.astype(np.uint8))
                np.save(out_img_dir  / f"{number_tag}.npy", (img_res_u8.astype(np.float32) / 255.0))

                # PNG: mask 0/255; image 0..255
                Image.fromarray(to_uint8_png_mask(mask_res)).save(out_mask_dir / f"{number_tag}.png")
                Image.fromarray(img_res_u8.astype(np.uint8)).save(out_img_dir / f"{number_tag}.png")

            log_rows.append({
                "number": number_tag,
                "label": label,
                "crop_side": side,
                "target": target,
                "resize_ratio": round(r, 6),
                "y1": y1, "x1": x1, "y2": y2, "x2": x2,
            })
            n_total += 1

    # Write the CSV log
    if not CFG["dry_run"]:
        log_path = outroot / "crop_log.csv"
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()) if log_rows else
                                    ["number","label","crop_side","target","resize_ratio","y1","x1","y2","x2"])
            writer.writeheader()
            writer.writerows(log_rows)

    print(f"\nDone: processed {n_total} items, skipped {n_skipped}. Output root: {outroot}")

def main():
    parser = argparse.ArgumentParser(description="Make 64×64 sponge & 16×16 needle patches from RFO data (image=.jpg).")
    parser.add_argument("--mask_dir", default=DEF_MASK_DIR, help="Directory containing mask .npy files (512x512).")
    parser.add_argument("--img_dir",  default=DEF_IMG_DIR,  help="Directory containing image .jpg/.jpeg files (512x512).")
    parser.add_argument("--csv",      default=DEF_CSV,      help="Annotation CSV with number and labeling columns.")
    parser.add_argument("--out_root", default=DEF_OUTROOT,  help="Output root directory that will contain four subfolders.")
    parser.add_argument("--dry_run",  action="store_true",  help="Print statistics only; do not write files.")
    args = parser.parse_args()

    if args.dry_run:
        CFG["dry_run"] = True

    process(args)

if __name__ == "__main__":
    main()
