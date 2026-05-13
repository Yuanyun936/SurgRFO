#!/bin/bash
#SBATCH --job-name=rfo_adm
#SBATCH --output=logs/rfo_adm_%j.out
#SBATCH --gres=gpu:1         
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# 64
MODEL_FLAGS64="--image_size 64 --channel_mult 1,2,3,4 --num_channels 128 --num_res_blocks 1 --learn_sigma True --class_cond False --num_heads 1 --num_head_channels 64 --use_scale_shift_norm True --dropout 0.0 --resblock_updown True"

# 128
MODEL_FLAGS128="--image_size 128 --channel_mult 1,2,3,4 --num_channels 128 --num_res_blocks 1 --learn_sigma True --class_cond False --num_heads 1 --num_head_channels 64 --use_scale_shift_norm True --dropout 0.0 --resblock_updown True"

# # 256
# MODEL_FLAGS256="--image_size 256 --channel_mult 1,2,3,4 --num_channels 128 --num_res_blocks 1 --learn_sigma True --class_cond False --num_heads 1 --num_head_channels 64 --use_scale_shift_norm True --dropout 0.0 --resblock_updown True"

# 256 nail
MODEL_FLAGS256="--image_size 256 --num_channels 128 --num_res_blocks 1 --learn_sigma True --class_cond False --num_heads 1 --num_head_channels 64 --use_scale_shift_norm True --dropout 0.0 --resblock_updown True"


# 80
MODEL_FLAGS80="--image_size 80 --channel_mult 1,2,3,4 --num_channels 128 --num_res_blocks 1 --learn_sigma True --class_cond False --num_heads 1 --num_head_channels 64 --use_scale_shift_norm True --dropout 0.0 --resblock_updown True"


DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 8 --use_fp16 False" 
# python image_train.py --data_dir ../data/motion-blur-dataset-generator/output/kernels_256x256 $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
python image_train.py --data_dir ../data/critical_RFO_data/RFOs_filtered_for_stage2/sponge_image64_TRAIN $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS

# python image_train.py --data_dir ../data/critical_RFO_data/RFOs_filtered_for_stage2/sponge_image256 $MODEL_FLAGS256 $DIFFUSION_FLAGS $TRAIN_FLAGS

# python image_train.py --data_dir ../data/critical_RFO_data/RFOs_filtered_for_stage2/sponge_image128 $MODEL_FLAGS128 $DIFFUSION_FLAGS $TRAIN_FLAGS

# python image_train.py --data_dir ../data/critical_RFO_data/RFOs_filtered_for_stage2/sponge_image80 $MODEL_FLAGS80 $DIFFUSION_FLAGS $TRAIN_FLAGS

#nail
# python image_train.py --data_dir ../guided-diffusion/datasets/data1 $MODEL_FLAGS256 $DIFFUSION_FLAGS $TRAIN_FLAGS


# #SBATCH --partition=h200