import argparse

import cv2
import numpy as np
import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)
from PIL import Image
from tqdm import tqdm
from utils import get_and_create_path, original_path, palette

low_threshold = 100
high_threshold = 200

parser = argparse.ArgumentParser()
parser.add_argument("classes", nargs="*", default=None)
args = parser.parse_args()
classes = set(args.classes)

# initialize models
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float16,
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.set_progress_bar_config(disable=True)

name = "canny"
path = get_and_create_path(name)

print("Canny ControlNet Augmentation")
for folder in original_path.iterdir():
    skip_count = 0
    if classes and folder.name not in classes:
        continue
    for file in tqdm(list(folder.iterdir()), desc=folder.name):

        # print(file)
        new_file = path / file.parent.name / (file.stem + "-" + name + file.suffix)
        # print(new_file)

        if new_file.exists():
            # print("Skipping " + file.parent.name + "/" + file.name)
            skip_count += 1
            continue

        image = str(file)
        image = Image.open(image).convert("RGB")
        image = cv2.resize(np.asarray(image), (512, 512))
        image = cv2.Canny(image, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)
        # image.save(str(new_file)[:-4] + "-edges.jpg")

        # Remove if you do not have xformers installed
        # see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
        # for installation instructions
        # pipe.enable_xformers_memory_efficient_attention()

        prompt = folder.name
        image = pipe(prompt, image, num_inference_steps=20).images[0]
        image = cv2.resize(np.asarray(image), (150, 150))
        image = Image.fromarray(image)
        image.save(new_file)

    print(f"Skipped {skip_count} images in {folder.name}")
