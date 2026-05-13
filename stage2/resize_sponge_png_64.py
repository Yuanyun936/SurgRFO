#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from PIL import Image

SRC_DIR = Path(os.environ.get("RFO_STAGE2_SRC_DIR", "./sponge_image256"))
DST_DIR = Path(os.environ.get("RFO_STAGE2_DST_DIR", str(SRC_DIR.parent / "sponge_image64")))
TARGET_SIZE = (64, 64)

def main():
    DST_DIR.mkdir(parents=True, exist_ok=True)
    pngs = sorted(SRC_DIR.glob("*.png"))
    if not pngs:
        print(f"[WARN] No PNG files found in source directory: {SRC_DIR}")
        return

    n_ok, n_err = 0, 0
    for p in pngs:
        try:
            with Image.open(p) as im:
                w, h = im.size
                if (w, h) != (256, 256):
                    print(f"[INFO] {p.name}: original size {w}x{h} -> 64x64")
                im_resized = im.resize(TARGET_SIZE, resample=Image.LANCZOS)
                out_path = DST_DIR / p.name
                im_resized.save(out_path, format="PNG", optimize=True)
                n_ok += 1
        except Exception as e:
            print(f"[ERROR] Failed to process {p.name}: {e}")
            n_err += 1

    print(f"\nDone: {n_ok} succeeded, {n_err} failed. Output: {DST_DIR}")

if __name__ == "__main__":
    main()
