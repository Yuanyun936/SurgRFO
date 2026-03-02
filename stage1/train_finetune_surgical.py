import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPTokenizer
from torchvision import transforms
from accelerate import Accelerator
from accelerate.utils import DistributedType
from itertools import cycle
# from tqdm import tqdm
from tqdm.auto import tqdm

class SurgicalDataset(Dataset):
    def __init__(self, image_dir, prompt="surgical sample"):
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".png")]    # or ".png"
        self.prompt = prompt
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return {
            "pixel_values": self.transform(image),
            "prompt": self.prompt
        }

image_dir = "./datasets/NO_RFO_dataset_png_train"
model_path = "./RoentGen/roentgen"
output_dir = "./RoentGen/roentgen_finetune_stage1"
os.makedirs(output_dir, exist_ok=True)

batch_size_per_gpu = 64
# max_train_steps    = 2000          
max_train_steps    = 5000          
save_every_steps   = 1000 

# Load model and tokenizer
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float32)

# freeze VAE and Text Encoder
pipe.vae.requires_grad_(False)
pipe.text_encoder.requires_grad_(False)


pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
# tokenizer = CLIPTokenizer.from_pretrained(pipe.text_encoder.config._name_or_path)
tokenizer = CLIPTokenizer.from_pretrained(os.path.join(model_path, "tokenizer"))

# dataset & dataloader
dataset = SurgicalDataset(image_dir)
dataloader = DataLoader(dataset, batch_size=batch_size_per_gpu, shuffle=True)
# optimizer
optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=5e-6)

# accelerator for device placement
accelerator = Accelerator(mixed_precision="fp16")
pipe.unet, optimizer, dataloader = accelerator.prepare(pipe.unet, optimizer, dataloader)
pipe.text_encoder.to(accelerator.device)
pipe.vae.to(accelerator.device)

# Enable training mode
pipe.unet.train()

print("Trainable parameters in UNet:")
for name, param in pipe.unet.named_parameters():
    if param.requires_grad:
        print(f" - {name}")


progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_main_process)
global_step  = 0
data_iter    = cycle(dataloader)

for step in progress_bar:
    batch = next(data_iter)

    with accelerator.accumulate(pipe.unet):
        # ------ Forward ------
        input_ids = tokenizer(batch["prompt"],
                              padding="max_length",
                              max_length=77,
                              return_tensors="pt").input_ids.to(accelerator.device)
        enc_hidden = pipe.text_encoder(input_ids)[0]

        latents = pipe.vae.encode(batch["pixel_values"].to(accelerator.device)).latent_dist.sample()
        latents = latents * 0.18215

        noise      = torch.randn_like(latents)
        timesteps  = torch.randint(0,
                                   pipe.scheduler.config.num_train_timesteps,
                                   (latents.shape[0],),
                                   device=latents.device).long()
        noisy_lat  = pipe.scheduler.add_noise(latents, noise, timesteps)
        pred_noise = pipe.unet(noisy_lat, timesteps, enc_hidden).sample

        loss = torch.nn.functional.mse_loss(pred_noise, noise)
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

    global_step += 1
    progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    # --------- save checkpoint ---------
    if global_step % save_every_steps == 0:
        if accelerator.is_main_process:
            unet_to_save = accelerator.unwrap_model(pipe.unet) 
            ckpt_dir = os.path.join(output_dir,
                                    f"checkpoint_step_{global_step}")
            unet_to_save.save_pretrained(ckpt_dir)
            print(f"\n[Main] Saved checkpoint to: {ckpt_dir}")

        accelerator.wait_for_everyone() 



# ========= Final save =========
if accelerator.is_main_process:
    pipe.unet = accelerator.unwrap_model(pipe.unet)
    final_dir = os.path.join(output_dir, "final_model")
    pipe.save_pretrained(final_dir)
    print(f"\n[Main] Training finished. Final model saved to: {final_dir}")

accelerator.wait_for_everyone()  


