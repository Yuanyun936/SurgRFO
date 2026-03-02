#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffusion  

accelerate launch train_finetune_surgical.py

