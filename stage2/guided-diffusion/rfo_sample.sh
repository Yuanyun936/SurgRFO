#!/bin/bash
#SBATCH --job-name=adm_inference
#SBATCH --output=logs/rfo_sample_%j.out
#SBATCH --gres=gpu:1           
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00


SAMPLE_FLAGS="--batch_size 4 --num_samples 32 --timestep_respacing 1000"
# MODEL_FLAGS="--diffusion_steps 1000 --noise_schedule linear --image_size 80  --channel_mult 1,2,3,4 --num_channels 128 --num_res_blocks 1 --learn_sigma True --class_cond False --num_heads 1 --num_head_channels 64 --use_scale_shift_norm True --dropout 0.0 --resblock_updown True"
MODEL_FLAGS="--diffusion_steps 1000 --noise_schedule linear --image_size 256  --num_channels 128 --num_res_blocks 1 --learn_sigma True --class_cond False --num_heads 1 --num_head_channels 64 --use_scale_shift_norm True --dropout 0.0 --resblock_updown True"

# python image_sample.py $MODEL_FLAGS --model_path ./model80_ckpt/model007000.pt  $SAMPLE_FLAGS


python image_sample.py $MODEL_FLAGS --model_path ./model_nailfold_ckpt_res1/model014000.pt  $SAMPLE_FLAGS

