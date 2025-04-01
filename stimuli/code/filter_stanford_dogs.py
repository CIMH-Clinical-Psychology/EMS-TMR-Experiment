#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 15:15:58 2025

filter the stanford dogs dataset for images that are larger than 200px wide
and then sort by width

@author: simon
"""
import os
import shutil
from PIL import Image
from tqdm import tqdm
import numpy as np

stanford_dogs_dir = '/home/simon/Desktop/datasets/stanford-dogs-dataset/versions/2/images/Images/'
out_dir = './stanford_dogs_filtered/'

# Create output directory if it doesn't exist
os.makedirs(out_dir, exist_ok=True)

folders = os.listdir(stanford_dogs_dir)


def move_rename_if_large_enough(image_path, out_path, min_area=400):
    # get image dimensions
    try:
        with Image.open(image_path) as img:
            w, h = img.size

        area = int(np.sqrt(w*h))

        orig_name, number = os.path.basename(image_path).split('_', 1)
        new_name = f'{orig_name}_a{area}_{number}'

        # move if large enough
        if area >= min_area:
            shutil.copy2(image_path, os.path.join(out_path, new_name))
            return True
        else:
            return False

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False


for folder in tqdm(folders):
    full_folder_name = os.path.join(stanford_dogs_dir, folder)
    if not os.path.isdir(full_folder_name):
        continue
    # Create corresponding folder in output directory
    folder_out_path = os.path.join(out_dir, folder)
    os.makedirs(folder_out_path, exist_ok=True)

    files = [f for f in os.listdir(full_folder_name) if f.endswith('.jpg')]
    for file in files:
        image_path = os.path.join(full_folder_name, file)
        move_rename_if_large_enough(image_path, out_path=folder_out_path)

print("Processing complete!")
