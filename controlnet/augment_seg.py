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
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
from utils import get_and_create_path, original_path, palette

parser = argparse.ArgumentParser()
parser.add_argument("classes", nargs="*", default=None)
args = parser.parse_args()
classes = set(args.classes)

# initialize models
image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
image_segmentor = UperNetForSemanticSegmentation.from_pretrained(
    "openmmlab/upernet-convnext-small"
)
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-seg", torch_dtype=torch.float16
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

name = "seg"
path = get_and_create_path(name)

for folder in original_path.iterdir():
    if classes and folder not in classes:
        continue
    for file in tqdm(list(folder.iterdir()), desc=folder.name):

        # print(file)
        new_file = path / file.parent.name / (file.stem + "-" + name + file.suffix)
        # print(new_file)

        if new_file.exists():
            print("Skipping " + file.parent.name + "/" + file.name)
            continue

        image = str(file)
        image = Image.open(image).convert("RGB")
        image = Image.fromarray(cv2.resize(np.asarray(image), (512, 512)))

        pixel_values = image_processor(image, return_tensors="pt").pixel_values

        with torch.no_grad():
            outputs = image_segmentor(pixel_values)

        seg = image_processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]

        color_seg = np.zeros(
            (seg.shape[0], seg.shape[1], 3), dtype=np.uint8
        )  # height, width, 3

        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color

        color_seg = color_seg.astype(np.uint8)
        image = Image.fromarray(color_seg)

        # Remove if you do not have xformers installed
        # see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
        # for installation instructions
        # pipe.enable_xformers_memory_efficient_attention()

        image = pipe(str(folder), image, num_inference_steps=20).images[0]
        image = cv2.resize(np.asarray(image), (150, 150))
        image = Image.fromarray(image)
        image.save(new_file)
