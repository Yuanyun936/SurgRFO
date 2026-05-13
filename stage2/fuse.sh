#!/bin/bash
#SBATCH --job-name=fuse
#SBATCH --output=logs/fuse_%j.out
#SBATCH --gres=gpu:1           
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=24:00:00

# # single image
# python poisson_fuse.py \
#   --xray ../stage1/results/results_stage1/surgical_results_gen100/sample_003.png \
#   --rfo-img ./guided-diffusion/samples-64/exp1-model5000/sample_00004.png \
#   --rfo-target-size 64 --mode mixed --feather 3 \
#   --center 450,120 \
#   --out ./stage2_fused/single_001.png

#   --rfo-img ../data/critical_RFO_data/RFOs_filtered_for_stage2/sponge_image256/45_2.png \
#   --rfo-mask ../data/critical_RFO_data/RFOs_filtered_for_stage2/sponge_mask256/45_2.png \

  # --rfo-img ../data/critical_RFO_data/RFOs_filtered_for_stage2/sponge_image256/40_2.png \
  # --rfo-mask ../data/critical_RFO_data/RFOs_filtered_for_stage2/sponge_mask256/40_2.png \

# batch images
python poisson_fuse.py \
  --xray-dir ../stage1/results/results_stage1/surgical_results_gen100 \
  --rfo-dir  ../data/critical_RFO_data/RFOs_filtered_for_stage2/needle_image256 \
  --mask-dir ../data/critical_RFO_data/RFOs_filtered_for_stage2/needle_mask256 \
  --rfo-target-size 64 \
  --mode mixed --feather 3 \
  --center 360,200 \
  --out-dir ./stage2_fused \
  --pairing one_to_one



# python poisson_fuse.py \
#   --xray-dir ../stage1/results/results_stage1/surgical_results_gen100 \
#   --rfo-dir  ./guided-diffusion/samples-64/exp1-model5000 \
#   --rfo-target-size 64 \
#   --mode mixed --feather 3 \
#   --center 90,230 \
#   --out-dir ./stage2_fused \
#   --pairing one_to_one
