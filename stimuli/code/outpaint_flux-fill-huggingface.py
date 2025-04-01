#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 10:51:14 2025

NOT ENOUGH CREDITS FOR OUR WHOLE DATASET. else works fantastic
using the huggingface API to outpaint an image with the space at
https://huggingface.co/spaces/multimodalart/flux-fill-outpaint

before running this script, start the docker
docker run -it -p 7860:7860 --platform=linux/amd64 --gpus all \
	-e HF_TOKEN="XXXX" \
	registry.hf.space/multimodalart-flux-fill-outpaint:latest python app.py



@author: simon
"""
import time
import random
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


import os
import imageio
import imageio.v3 as imageio
from tqdm import tqdm
from gradio_client import Client, handle_file
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from httpx import ConnectError
# client = Client("http://127.0.0.1:7860/")

folder = './stimuli_1/house2.0/'
images = [os.path.join(folder, f)  for f in os.listdir(folder) if f.endswith('.jpg')]
out_folder = folder + '/square_flux'
os.makedirs(out_folder, exist_ok=True)

def outpaint(image):

    # fig, axs = plt.subplots(1, 2, figsize=[12, 8])
    basename = os.path.basename(image)
    out_file = os.path.join(out_folder, basename[:-4] + '_fluxapi.jpg')

    if os.path.isfile(out_file):
        print(f'{basename} already exists, skip')
        return

    for i in range(10):
        try:
            client = Client("https://8709ec83ebdcf00393.gradio.live", ssl_verify=False)
            break
        except ConnectError as e:
            print(f'connecterror! {e} {repr(e)}')
            time.sleep(random.random()*i)
    else:
        raise Exception('cant connect')

    img_input = imageio.imread(image)
    square = max(img_input.shape)
    h, w = img_input.shape[:2]

    if abs(w-h)<=2:
        imageio.imwrite(out_file, img_input, quality=80)
        print(f'{basename} is already square')
        return

    client.predict(
        api_name="/clear_result"
        )

    expand_h = h<w
    expand_w = h>=w

    res = client.predict(
      image=handle_file(image),
      width=720,
      height=720,
      overlap_percentage=10,
      num_inference_steps=28,
      resize_option="75%",
      custom_resize_percentage=50,
      prompt_input="",
      alignment="Middle",
      overlap_left=expand_w,
      overlap_right=expand_w,
      overlap_top=expand_h,
      overlap_bottom=expand_h,
      api_name="/inpaint"
    )

    # read output image
    img = imageio.imread(res[0])
    # axs[0].imshow(img_input)
    # axs[1].imshow(img)
    # plt.pause(0.1)
    imageio.imwrite(out_file, img, quality=80)

Parallel(4, backend='threading')(delayed(outpaint)(image) for image in tqdm(images[::-1]))
