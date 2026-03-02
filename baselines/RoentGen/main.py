
import torch
from diffusers import StableDiffusionPipeline
# import matplotlib.pyplot as plt
import numpy as np
import random
import os

model_path = "./RoentGen/roentgen"
device='cuda'  
pipe = StableDiffusionPipeline.from_pretrained(model_path, use_auth_token=True).to(torch.float32).to(device)
# pipe.safety_checker = lambda imgs, _: (imgs, False)
pipe.safety_checker = lambda images, **kwargs: (images, [False])

save_dir = "res"
os.makedirs(save_dir, exist_ok=True)

# prompt = "big right-sided pleural effusion"
prompt = ""
# prompt = "surgical sample"

k=0
for i in range(10):
    seed = 40+i
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    generator = torch.Generator("cuda").manual_seed(seed)
    for j in range(5):
        result = pipe(
            prompt="",
            num_inference_steps=75,
            guidance_scale=4,
            generator=generator
        ).images[0]

        result.save(os.path.join(save_dir, f"sample_{k:03d}.png"))
        k=k+1