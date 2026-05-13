#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Poisson fusion (OpenCV seamlessClone) for pasting an RFO patch onto a chest X-ray.

Features
--------
- Single-run or batch mode (directories).
- RFO is resized *before* fusion to a controllable square size (e.g., 80/64/128).
- Optional mask: if not provided (or "none"), the whole rectangular RFO patch is used as mask.
- Optional feathering of the mask boundary.
- Local brightness/contrast matching (src->dst) computed on the pasted region only.
- Three fusion modes: mixed (default), normal, monochrome.
- Annotated output (box + (X,Y) center) saved as a separate *_annotated.png.
- NEW: a side-by-side canvas (Before / After / Annotated) saved as *_canvas.png (can be disabled).

Default folders (adapt to your setup)
-------------------------------------
X-rays (512x512): ../stage1/results/results_stage1/surgical_results_gen100
RFO images (256x256): ../data/critical_RFO_data/RFOs_filtered_for_stage2/sponge_image256
RFO masks (256x256):  ../data/critical_RFO_data/RFOs_filtered_for_stage2/sponge_mask256
"""

from __future__ import annotations
from pathlib import Path
import argparse
import sys
from typing import Optional, Tuple, List
import numpy as np
import cv2

# --------- Default directories ----------
DEF_XRAY_DIR = "../stage1/results/results_stage1/surgical_results_gen100"
DEF_RFO_DIR  = "../data/critical_RFO_data/RFOs_filtered_for_stage2/sponge_image256"
DEF_MASK_DIR = "../data/critical_RFO_data/RFOs_filtered_for_stage2/sponge_mask256"

# --------- OpenCV seamlessClone flags ----------
FLAG_MAP = {
    "normal": cv2.NORMAL_CLONE,
    "mixed":  cv2.MIXED_CLONE,
    "mono":   cv2.MONOCHROME_TRANSFER,
    "monochrome": cv2.MONOCHROME_TRANSFER,
}

# ======================= I/O helpers =======================

def imread_any(path: str | Path) -> np.ndarray:
    """
    Read image from png/jpg/jpeg or .npy; return uint8.
    - If npy and CHW, convert to HWC.
    - If value range appears 0..1, scale to 0..255.
    - If alpha present, drop alpha (keep first 3 channels).
    """
    p = str(path)
    if p.lower().endswith(".npy"):
        a = np.load(p)
        if a.ndim == 3 and a.shape[0] in (1, 3, 4):   # CHW -> HWC
            a = np.moveaxis(a, 0, -1)
        a = a.astype(np.float32)
        vmax = float(np.max(a)) if a.size else 1.0
        if vmax <= 1.5:   # assume [0,1]
            a = np.clip(a, 0, 1) * 255.0
        else:             # assume [0,255] (or bigger -> clip)
            a = np.clip(a, 0, 255)
        a = a.astype(np.uint8)
        if a.ndim == 3 and a.shape[2] == 4:
            a = a[:, :, :3]
        return a
    else:
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        return img

def ensure_3ch(u8: np.ndarray) -> np.ndarray:
    """
    Ensure an image is 3-channel BGR uint8.
    - If grayscale, convert to BGR.
    - If 3 channels, return as is.
    """
    if u8.ndim == 2:
        return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    if u8.ndim == 3 and u8.shape[2] == 3:
        return u8
    raise ValueError(f"Unsupported image shape: {u8.shape}")

def imread_mask_optional(path: Optional[str | Path], shape_hw: Tuple[int, int]) -> np.ndarray:
    """
    Read a binary mask if provided:
      - If path is None / "none" / "", return full-ones mask (rectangular patch).
      - If path is given, load (npy/png/jpg), binarize to 0/255, and resize to shape_hw with NEAREST.
    """
    h, w = shape_hw
    if (path is None) or (str(path).lower() in ("none", "null", "")):
        return np.full((h, w), 255, dtype=np.uint8)

    p = str(path)
    if p.lower().endswith(".npy"):
        m = np.load(p)
        if m.ndim > 2:
            m = m[..., 0]
        m = (m > 0).astype(np.uint8) * 255
    else:
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f"Cannot read mask: {path}")
        m = (m > 127).astype(np.uint8) * 255

    if (m.shape[0], m.shape[1]) != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return m

# ======================= Processing utils =======================

def feather_mask(mask_255: np.ndarray, feather: int) -> np.ndarray:
    """
    Light feathering to soften the boundary.
    Returns a binary mask (0/255) after morphology + Gaussian blur + threshold.
    """
    if feather <= 0:
        return mask_255
    k = max(1, int(feather))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(mask_255, cv2.MORPH_CLOSE, kernel, iterations=1)
    m = cv2.GaussianBlur(m, (0, 0), sigmaX=feather/2, sigmaY=feather/2)
    _, m = cv2.threshold(m, 32, 255, cv2.THRESH_BINARY)
    return m

def match_local_stats_color(
    src_bgr: np.ndarray,
    dst_bgr: np.ndarray,
    small_mask_255: np.ndarray,   # mask aligned to src size (0/255)
    y1: int, y2: int, x1: int, x2: int  # ROI in dst where src will be pasted
) -> np.ndarray:
    """
    Match src brightness/contrast to dst over the pasted region.

    We compute grayscale statistics on:
      - src within small_mask_255
      - dst within the *same pixels* of dst ROI
    Then apply a single (gain, bias) to all channels of src.
    """
    # Crop dst ROI corresponding to the pasted area.
    dst_roi = dst_bgr[y1:y2+1, x1:x2+1]
    ph, pw = src_bgr.shape[:2]

    # Defensive alignment (should already match exactly).
    if dst_roi.shape[0] != ph or dst_roi.shape[1] != pw:
        h2, w2 = min(ph, dst_roi.shape[0]), min(pw, dst_roi.shape[1])
        src_bgr = src_bgr[:h2, :w2]
        small_mask_255 = small_mask_255[:h2, :w2]
        dst_roi = dst_roi[:h2, :w2]
        ph, pw = h2, w2

    m_small = (small_mask_255 > 0)
    if not np.any(m_small):
        return src_bgr

    s_gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    d_gray = cv2.cvtColor(dst_roi,  cv2.COLOR_BGR2GRAY).astype(np.float32)

    s_sel = s_gray[m_small]
    d_sel = d_gray[m_small]

    s_std = float(s_sel.std())
    d_std = float(d_sel.std())
    s_mean = float(s_sel.mean())
    d_mean = float(d_sel.mean())

    eps = 1e-6
    if s_std < eps:
        gain, bias = 1.0, (d_mean - s_mean)  # nearly constant src
    else:
        gain = d_std / s_std
        bias = d_mean - gain * s_mean

    out = src_bgr.astype(np.float32) * gain + bias
    return np.clip(out, 0, 255).astype(np.uint8)

def parse_center(center_str: Optional[str], H: int, W: int, ph: int, pw: int) -> Tuple[int, int]:
    """
    Parse "x,y" and clamp so the pasted rectangle fits inside the destination image.
    If not provided, default to a chest-like lower center.
    """
    half_h, half_w = ph // 2, pw // 2
    if center_str:
        xs, ys = center_str.split(",")
        cx, cy = int(float(xs)), int(float(ys))
    else:
        cx, cy = W // 2, int(H * 0.58)
    cx = max(half_w, min(W - half_w - 1, cx))
    cy = max(half_h, min(H - half_h - 1, cy))
    return cx, cy

def draw_annotation(img_bgr: np.ndarray, rect_xyxy: Tuple[int, int, int, int], cx: int, cy: int) -> np.ndarray:
    """
    Draw the bounding box and the pasted center "(X,Y)" on a copy of the fused image.
    """
    x1, y1, x2, y2 = rect_xyxy
    vis = img_bgr.copy()
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
    label = f"RFO center=({cx},{cy})"
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    tx, ty = x1, max(0, y1 - th - 6)
    cv2.rectangle(vis, (tx, ty), (tx + tw + 6, ty + th + baseline + 6), (0, 0, 0), -1)
    cv2.putText(vis, label, (tx + 3, ty + th), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return vis

# ---------- NEW: canvas helpers ----------

def _put_title(img_bgr: np.ndarray, text: str, bar_h: int = 34) -> np.ndarray:
    """
    Add a dark title bar on top with white text. Returns a new image with extra bar.
    """
    h, w = img_bgr.shape[:2]
    bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
    bar[:] = (30, 30, 30)
    vis = np.vstack([bar, img_bgr])
    # put text centered-ish
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    tx = max(8, (w - tw) // 2)
    ty = (bar_h + th) // 2 + 4
    cv2.putText(vis, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    return vis

def save_canvas(before_bgr: np.ndarray, after_bgr: np.ndarray, annotated_bgr: np.ndarray, out_path: Path):
    """
    Create and save a horizontal canvas that shows:
      [ Before | After | Annotated ]
    The canvas file is saved as <out_stem>_canvas.png next to the fused image.
    """
    # Ensure same size (use 'after' as reference)
    H, W = after_bgr.shape[:2]
    def resize_to(img):
        if img.shape[:2] != (H, W):
            return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        return img

    b = resize_to(before_bgr)
    a = resize_to(after_bgr)
    an = resize_to(annotated_bgr)

    b = _put_title(b, "Before (X-ray)")
    a = _put_title(a, "After (Fused)")
    an = _put_title(an, "Annotated (RFO)")

    pad = 8
    pad_col = np.full((b.shape[0], pad, 3), 230, dtype=np.uint8)
    canvas = np.hstack([b, pad_col, a, pad_col, an])

    canvas_path = out_path.with_name(out_path.stem + "_canvas.png")
    cv2.imwrite(str(canvas_path), canvas)
    return canvas_path

def list_images(dir_path: Path) -> List[Path]:
    """List common image files in a directory."""
    exts = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")
    return sorted([p for p in Path(dir_path).glob("*") if p.suffix in exts])

# ======================= Core: one fusion =======================

def fuse_once(
    xray_path: Path,
    rfo_img_path: Path,
    rfo_mask_path: Optional[Path],
    out_path: Path,
    mode: str = "mixed",
    rfo_target_size: int = 64,
    center_str: Optional[str] = None,
    feather: int = 2,
    match_stats: bool = True,
    annotate: bool = True,
    save_canvas_flag: bool = True,
):
    """
    Perform one Poisson fusion:
      1) Read X-ray and RFO; force both to 3-ch BGR.
      2) Resize RFO to the target square size (e.g., 64/80/128) BEFORE fusion.
      3) Prepare a mask (given or full-rect) and optionally feather.
      4) Compute paste center and ROI; match local stats if enabled.
      5) Run seamlessClone and save outputs (+ annotated preview if requested).
      6) Save a side-by-side canvas (Before/After/Annotated) if enabled.
    """
    # 1) Read inputs
    xray_raw = imread_any(xray_path)
    rfo_raw  = imread_any(rfo_img_path)

    xray_3c = ensure_3ch(xray_raw)   # "Before"
    rfo_3c  = ensure_3ch(rfo_raw)

    # 2) Controlled resizing (e.g., 256 -> 80 or 64 or 128)
    if rfo_target_size is not None and rfo_target_size > 0:
        orig_h, orig_w = rfo_3c.shape[:2]
        interp = cv2.INTER_CUBIC if rfo_target_size > max(orig_h, orig_w) else cv2.INTER_AREA
        rfo_3c = cv2.resize(rfo_3c, (int(rfo_target_size), int(rfo_target_size)), interpolation=interp)

    # 3) Mask (optional), aligned to the *resized* RFO
    mask = imread_mask_optional(str(rfo_mask_path) if rfo_mask_path else None,
                                shape_hw=(rfo_3c.shape[0], rfo_3c.shape[1]))
    mask = feather_mask(mask, feather)

    # 4) Compute paste location and ROI
    H, W = xray_3c.shape[:2]
    ph, pw = rfo_3c.shape[:2]
    cx, cy = parse_center(center_str, H, W, ph, pw)
    x1, y1 = cx - pw // 2, cy - ph // 2
    x2, y2 = x1 + pw - 1, y1 + ph - 1

    # Optional stats matching using the *small* mask on src and dst ROI
    if match_stats:
        rfo_3c = match_local_stats_color(
            src_bgr=rfo_3c,
            dst_bgr=xray_3c,
            small_mask_255=mask,
            y1=y1, y2=y2, x1=x1, x2=x2
        )

    # 5) Poisson fusion (OpenCV seamlessClone expects src/dst 3-ch, mask single-channel 0/255)
    flag = FLAG_MAP[mode]
    fused_bgr = cv2.seamlessClone(rfo_3c, xray_3c, mask, (cx, cy), flag)  # "After"

    # 6) Save outputs
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), fused_bgr)

    # Create annotated image (box + (X,Y))
    if annotate:
        vis = draw_annotation(fused_bgr, (x1, y1, x2, y2), cx, cy)
        ann_path = out_path.with_name(out_path.stem + "_annotated.png")
        cv2.imwrite(str(ann_path), vis)
    else:
        vis = fused_bgr  # canvas will still work (shows same as "After")

    # Save side-by-side canvas
    canvas_path = None
    if save_canvas_flag:
        canvas_path = save_canvas(xray_3c, fused_bgr, vis, out_path)

    print(f"[OK] {xray_path.name} + {rfo_img_path.name}  "
          f"→ RFO {pw}x{ph}, center=({cx},{cy}), mode={mode}, "
          f"mask={'full-rect' if (rfo_mask_path is None or str(rfo_mask_path).lower()=='none') else rfo_mask_path.name}")
    if save_canvas_flag:
        print(f"Saved canvas: {canvas_path}")
    return out_path

# ======================= CLI / Batch =======================

def list_images(dir_path: Path) -> List[Path]:
    """List common image files in a directory."""
    exts = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")
    return sorted([p for p in Path(dir_path).glob("*") if p.suffix in exts])

def main():
    ap = argparse.ArgumentParser(description="Poisson fusion (seamlessClone) with controllable RFO resizing and canvas export.")
    # Single
    ap.add_argument("--xray",     help="Single X-ray (png/jpg/npy; gray or 3ch)")
    ap.add_argument("--rfo-img",  help="Single RFO image (3ch preferred; png/jpg/npy)")
    ap.add_argument("--rfo-mask", default=None, help="Single RFO mask path or 'none'")
    ap.add_argument("--out",      help="Single fused output image path")

    # Batch
    ap.add_argument("--xray-dir", default=DEF_XRAY_DIR, help="Dir of 512x512 X-rays")
    ap.add_argument("--rfo-dir",  default=DEF_RFO_DIR,  help="Dir of 256x256 RFO images")
    ap.add_argument("--mask-dir", default=DEF_MASK_DIR, help="Dir of RFO masks; or 'none' to disable")
    ap.add_argument("--out-dir",  default=None,         help="Output dir for batch mode")

    # Controls
    ap.add_argument("--rfo-target-size", type=int, default=64,
                    help="Resize RFO to this square size BEFORE fusion (e.g., 80/64/128).")
    ap.add_argument("--mode",   default="mixed", choices=list(FLAG_MAP.keys()), help="seamlessClone mode")
    ap.add_argument("--center", default=None, help="Paste center 'x,y'. If omitted, auto center.")
    ap.add_argument("--feather", type=int, default=2, help="Feather radius for mask (0 disables)")
    ap.add_argument("--match-stats", action="store_true", help="Match RFO brightness/contrast to local dst region")
    ap.add_argument("--no-match-stats", dest="match_stats", action="store_false")
    ap.set_defaults(match_stats=True)
    # ap.set_defaults(match_stats=False)


    ap.add_argument("--annotate", action="store_true", help="Also save an annotated preview with box & coordinates")
    ap.add_argument("--no-annotate", dest="annotate", action="store_false")
    ap.set_defaults(annotate=True)

    # NEW: canvas on/off
    ap.add_argument("--canvas", dest="save_canvas", action="store_true", help="Save a side-by-side canvas (default)")
    ap.add_argument("--no-canvas", dest="save_canvas", action="store_false", help="Do not save the canvas image")
    ap.set_defaults(save_canvas=True)

    ap.add_argument("--pairing", choices=["one_to_one", "cartesian"], default="one_to_one",
                    help="Batch pairing: zip by sorted filename, or full Cartesian product.")

    args = ap.parse_args()

    # Decide single vs batch
    single = all([args.xray, args.rfo_img, args.out])

    if single:
        mask_path = None if (args.rfo_mask is None or str(args.rfo_mask).lower() == "none") else Path(args.rfo_mask)
        fuse_once(
            xray_path=Path(args.xray),
            rfo_img_path=Path(args.rfo_img),
            rfo_mask_path=mask_path,
            out_path=Path(args.out),
            mode=args.mode,
            rfo_target_size=args.rfo_target_size,
            center_str=args.center,
            feather=args.feather,
            match_stats=args.match_stats,
            annotate=args.annotate,
            save_canvas_flag=args.save_canvas,
        )
        return

    # Batch mode
    if not args.out_dir:
        print("Batch mode requires --out-dir", file=sys.stderr)
        sys.exit(2)

    xray_dir = Path(args.xray_dir)
    rfo_dir  = Path(args.rfo_dir)
    mask_dir = None if (args.mask_dir is None or str(args.mask_dir).lower() == "none") else Path(args.mask_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xlist = list_images(xray_dir)
    rlist = list_images(rfo_dir)
    if not xlist:
        print(f("[ERR] No X-ray images in {xray_dir}"), file=sys.stderr); sys.exit(3)
    if not rlist:
        print(f("[ERR] No RFO images in {rfo_dir}"), file=sys.stderr); sys.exit(4)

    def find_mask_for(rfo_path: Path) -> Optional[Path]:
        if mask_dir is None:
            return None
        stem = rfo_path.stem
        for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG", ".npy"):
            p = mask_dir / f"{stem}{ext}"
            if p.exists():
                return p
        return None  # fall back to full-rect

    jobs: List[Tuple[Path, Path, Optional[Path], Path]] = []

    if args.pairing == "one_to_one":
        n = min(len(xlist), len(rlist))
        for i in range(n):
            x, r = xlist[i], rlist[i]
            m = find_mask_for(r)
            out_name = f"{x.stem}__{r.stem}__s{args.rfo_target_size}.png"
            jobs.append((x, r, m, out_dir / out_name))
    else:  # cartesian product
        for x in xlist:
            for r in rlist:
                m = find_mask_for(r)
                out_name = f"{x.stem}__{r.stem}__s{args.rfo_target_size}.png"
                jobs.append((x, r, m, out_dir / out_name))

    print(f"[INFO] Batch jobs: {len(jobs)} (pairing={args.pairing})")
    for (x, r, m, o) in jobs:
        fuse_once(
            xray_path=x, rfo_img_path=r, rfo_mask_path=m, out_path=o,
            mode=args.mode, rfo_target_size=args.rfo_target_size,
            center_str=args.center, feather=args.feather,
            match_stats=args.match_stats, annotate=args.annotate,
            save_canvas_flag=args.save_canvas,
        )

if __name__ == "__main__":
    main()
