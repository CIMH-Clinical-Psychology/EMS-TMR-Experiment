#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 08:36:09 2025

simple script to make images square

if the image is almost square:
    crop it
else:
    use outpainting AI



@author: simon
"""


from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16
).to("cuda")

# Load image and create mask for outpainting area
image = Image.open("/home/simon/zi_nextcloud/Masterthesis_EMS-TMR/EMS-TMR-Experiment/stimuli/stimuli_1/dog/dog214.jpg")
# Create expanded canvas with original image in center

width, height = image.size

# Define how much to expand the canvas (in pixels)
expand_x = 128  # expand horizontally
expand_y = 128  # expand vertically

# Create expanded canvas
new_width = width + (2 * expand_x)
new_height = height + (2 * expand_y)
expanded_image = Image.new("RGB", (new_width, new_height), (0, 0, 0))
expanded_image.paste(image, (expand_x, expand_y))

# Create mask (white where we want to outpaint, black for original image)
mask = Image.new("RGB", (new_width, new_height), (255, 255, 255))
mask.paste(Image.new("RGB", (width, height), (0, 0, 0)), (expand_x, expand_y))
mask = mask.convert("L")  # Convert to grayscale

# Run outpainting
result = pipe(
    prompt="seamless continuation of the image, high quality, detailed",
    image=expanded_image,
    mask_image=mask,
    num_inference_steps=50,
    guidance_scale=7.5
).images[0]

# Save result
