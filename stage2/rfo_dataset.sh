#!/bin/bash
#SBATCH --job-name=sd_inference
#SBATCH --output=logs/rfo_%j.out
#SBATCH --gres=gpu:1           
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# python make_rfo_patches.py \
#   --mask_dir ../data/critical_RFO_data/images_and_masks/mask \
#   --img_dir  ../data/critical_RFO_data/images_and_masks/image \
#   --csv      ../data/critical_RFO_data/images_and_masks/rfo_anno.CSV \
#   --out_root ../data/critical_RFO_data/images_and_masks

python resize_sponge_png_64.py

## SBATCH --partition=h200