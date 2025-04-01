#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 13:57:48 2025

@author: simon
"""
import os
import io
import imageio
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
from diffusers import FluxFillPipeline
from diffusers.utils import load_image


folder = './stimuli_1/dogs2.0/'
images = [os.path.join(folder, f)  for f in os.listdir(folder) if f.endswith('.jpg')]
out_folder = folder + '/gemini2-exp/'
os.makedirs(out_folder, exist_ok=True)

overlap = 0.1

os.environ

def buffer_numpy_png(array):
    buffer = io.BytesIO()
    imageio.imwrite(buffer, array, format='png')
    buffer.seek(0)
    return buffer


for i, image in enumerate(tqdm(images)):
    if i>10: break

    img_input = imageio.imread(image)
    h, w = img_input.shape[:2]
    square = max([w, h])

    dw = square - w
    dh = square - h

    if abs(w-h)<3:
        print('image already square')
        continue


    # Create square canvas with 3 channels
    img_expanded = np.ones((square, square, 4), dtype=np.uint8)*255
    img_expanded[:,:,3] = 0  # Set alpha channel to transparent

    # Calculate offsets to center the image
    top = dh // 2
    left = dw // 2

    # Place original image in the center
    img_expanded[top:top+h, left:left+w, :3] = img_input
    img_expanded[top:top+h, left:left+w, 3] = 255  # Set alpha to opaque where

    # Create mask (1 where padding, 0 where original image)
    h_overlap = 0 if h==square else  int(top * 0.1)
    w_overlap = 0 if w==square else  int(left * 0.1)
    mask = np.ones((square, square), dtype=np.uint8)*255
    mask[top+h_overlap:top+h-h_overlap, left+w_overlap:left+w-w_overlap] = False
    # Shrink mask to create 10% overlap with the image
    # overlap_w = int(w * 0.1)

    assert img_expanded.shape[:2] == mask.shape[:2]
    imageio.imwrite('image.png', img_expanded)
    imageio.imwrite('mask.png', mask)


    # image = load_image("https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/cup.png")
    # mask = load_image("https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/cup_mask.png")

    pipe = FluxFillPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Fill-dev",
        # device_map="balanced",  # Automatically optimize placement between GPU and CPU
        offload_folder="offload",  # Folder for offloaded weights
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16
    )

    image = pipe(
        prompt="a white paper cup",
        image=img_expanded[:,:,:3],
        mask_image=mask,
        height=square,
        width=square,
        guidance_scale=30,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]
    image.save(f"flux-fill-dev.png")

    # do some plotting
    fig, axs = plt.subplots(1, 3, figsize=[18, 8])
    axs[0].imshow(imageio.imread('image.png'))
    axs[1].imshow(imageio.imread('mask.png'), cmap='gray')
    axs[2].imshow(imageio.imread('mask.png'), cmap='gray')
    plt.pause(0.1)
