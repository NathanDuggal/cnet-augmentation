import random
import torch
from diffusers import StableDiffusionPipeline

output_dir = "intel-image-classification/seg_train/seg_train/buildings"

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt_list = [
    "City building",
    "Religious building",
    "Rural building",
    "Modern building",
    "Historic building",
    "Industrial building",
    "Residential building",
    "Commercial building",
    "Skyscraper",
    "Townhouse",
]

for i in range(2500):
    prompt = random.choice(prompt_list) + ("s" if random.random() < 0.5 else "")
    image = pipe(prompt).images[0]

    save_dir = f"{output_dir}/{i}.jpg"
    image.save(save_dir)
    print(f"Image with prompt '{prompt}' saved to: {save_dir}")
