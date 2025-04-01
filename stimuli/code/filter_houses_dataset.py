#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 11:44:56 2025

@author: simon
"""

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

houses_dir = '/home/simon/Desktop/datasets/images.cv_jmq76c8hhjke3fmzlpuu1d/data/'
out_dir = './houses_filtered/'  # https://images.cv/dataset/house-image-classification-dataset

# Create output directory if it doesn't exist
os.makedirs(out_dir, exist_ok=True)

folders = os.listdir(houses_dir)


def move_rename_if_large_enough(image_path, out_path, min_area=400):
    # get image dimensions
    try:
        with Image.open(image_path) as img:
            w, h = img.size

        area = int(np.sqrt(w*h))

        orig_name = os.path.basename(image_path)[:-4]
        new_name = f'a{area}_{orig_name}.jpg'

        # move if large enough
        if area >= min_area:
            shutil.copy2(image_path, os.path.join(out_path, new_name))
            return True
        else:
            return False

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False


files = [f for f in os.listdir(houses_dir) if f.endswith('.jpg')]

for file in files:
    image_path = os.path.join(houses_dir, file)
    move_rename_if_large_enough(image_path, out_path=out_dir)

print("Processing complete!")
