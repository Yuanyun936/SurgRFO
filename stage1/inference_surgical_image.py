import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import random
import numpy as np


save_dir = "surgical_results_stage1"
os.makedirs(save_dir, exist_ok=True)

# Load Model
pipe = StableDiffusionPipeline.from_pretrained(
    "./RoentGen/roentgen_finetune_stage1/final_model",
    torch_dtype=torch.float32
).to("cuda")


pipe.safety_checker = lambda images, **kwargs: (images, [False])

k=0
# Sample images
for i in range(20):
    seed = 40+i
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    generator = torch.Generator("cuda").manual_seed(seed)
    for j in range(5):
        result = pipe(
            prompt="surgical sample",
            num_inference_steps=75,
            guidance_scale=4,
            generator=generator
        ).images[0]

        result.save(os.path.join(save_dir, f"sample_{k:03d}.png"))
        k=k+1

print(f"Saved 100 images to {save_dir}/")
