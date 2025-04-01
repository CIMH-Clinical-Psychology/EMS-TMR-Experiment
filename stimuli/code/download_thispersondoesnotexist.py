#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 08:53:20 2025

download fake face images from thispersondoesnotexist.com
then classify the gender, race and age with deepface.

@author: simon
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
import io
import requests
from tqdm import tqdm
import uuid
from PIL import Image
from deepface import DeepFace
import numpy as np
import hashlib
from joblib import Parallel, delayed
hash_this = lambda bytesx: hashlib.md5(bytesx).hexdigest()



n_images = 10000
out_folder = './thispersondoesnotexist/'

os.makedirs(out_folder, exist_ok=True)

def get_image(out_folder):
    res = requests.get('https://thispersondoesnotexist.com')
    assert res.ok, f'code {res.status_code}'
    uid = hash_this(res.content)[:8]

    # check if we already have this file
    existing_files = os.listdir(out_folder)
    for file in existing_files:
        if uid in file:
            print('file already exists!')
            return

    # convert to image and crop
    image = Image.open(io.BytesIO(res.content))
    image = image.crop([12, 0, 1012, 1000])

    # next, classify the gender, race and age of the fake person
    img_array = np.array(image)[:, :, ::-1]
    try:
        attr = DeepFace.analyze(
          img_path = img_array, actions = ['age', 'gender', 'race'],
          silent=True,
        )
    except ValueError:
        print('no face? error! ')
        return
    if len(attr)>1:
        print('more than one face? error!')
        return
    attr = attr[0]

    if attr['gender'][gender:=attr['dominant_gender']]<70:
       gender = 'unclear'

    if attr['race'][race:=attr['dominant_race']]<30:
       race = 'unclear'

    # if race.lower() in ['latino hispanic', 'asian', 'middle eastern', 'white']:
    #     print(f'enough of {race} already!')
    #     return

    age = attr['age']

    out_file = out_folder + f'/{gender}_{race}_{age}_{uid}.jpg'.lower()
    image.save(out_file, format='jpeg', quality=70)


Parallel(5, backend='loky', verbose=10)(delayed(get_image)(out_folder) for _ in tqdm(range(n_images)))
