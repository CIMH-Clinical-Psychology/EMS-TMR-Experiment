#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 12:10:23 2025

@author: simon
"""
import os
import io
import imageio
from tqdm import tqdm
import replicate
import numpy as np
import matplotlib.pyplot as plt


folder = './stimuli_1/dogs2.0/'
images = [os.path.join(folder, f)  for f in os.listdir(folder) if f.endswith('.jpg')]
out_folder = folder + '/gemini2-exp/'
os.makedirs(out_folder, exist_ok=True)

overlap = 0.3

out_folder = f'{folder}/replicate/'
os.makedirs(out_folder, exist_ok=True)

def buffer_numpy_png(array):
    buffer = io.BytesIO()
    imageio.imwrite(buffer, array, format='png')
    buffer.seek(0)
    return buffer


for i, image in enumerate(tqdm(images)):
    if i>20: break

    img_input = imageio.imread(image)
    h, w = img_input.shape[:2]
    square = max([w, h])

    dw = square - w
    dh = square - h

    if abs(w-h)<3:
        print('image already square')
        basename = os.path.splitext(os.path.basename(image))[0]
        out_file = f'{out_folder}/{basename}_replicate.jpg'
        imageio.imwrite(out_file, img_input, quality=80)
        continue

    fig, axs = plt.subplots(1, 3, figsize=[18, 8])

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
    h_overlap = 0 if h==square else  int(top * overlap)
    w_overlap = 0 if w==square else int(left * overlap)
    mask = np.ones((square, square), dtype=np.uint8)*255
    mask[top+h_overlap:top+h-h_overlap, left+w_overlap:left+w-w_overlap] = False
    # Shrink mask to create 10% overlap with the image
    # overlap_w = int(w * 0.1)

    assert img_expanded.shape[:2] == mask.shape[:2]

    output = replicate.run(
      "black-forest-labs/flux-fill-pro",
      input={
        "prompt": "seamlessly expand the outer areas naturally with a boring and non-salient texture.",
        "prompt_strength": 0.5,
        # "negative_prompt": "interesting, eye-catching, salient",
        "image": buffer_numpy_png(img_expanded),
        "mask": buffer_numpy_png(mask)
      }
    )
    # Save the outpainted image
    buffer = io.BytesIO(output.read())
    img_outpainted = imageio.imread(output.read())
    axs[0].imshow(img_expanded)
    axs[1].imshow(mask, cmap='gray')
    axs[2].imshow(img_outpainted)
    plt.pause(0.1)

    basename = os.path.splitext(os.path.basename(image))[0]
    out_file = f'{out_folder}/{basename}_replicate.jpg'
    imageio.imwrite(out_file, img_outpainted, quality=80)
