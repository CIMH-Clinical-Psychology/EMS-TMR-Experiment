# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 11:58:26 2025



@author: Simon.Kern
"""
import os
import random
import shutil
import uuid

random.seed(0)

categories = ['dog', 'flower', 'house', 'face']


shuffle_idx_start = 32    # only shuffle images with id LARGER than than this

# Define folder path

def extract_number(string):
    return int(''.join([x for x in string if x.isdigit()]))

for cat in categories:
    folder = f'../{cat}/'
    # Get all image files above dog031.png
    files = [f for f in os.listdir(folder) if f.startswith(cat) and f.endswith((".jpg", ".png"))]
    files_to_shuffle = [f for f in files if extract_number(f) >= shuffle_idx_start]

    # first rename all files to some random name
    for file in files_to_shuffle:
        assert not '031' in file, 'sanity check failes'

    # Shuffle the files
    random.shuffle(files_to_shuffle)

    # Move files to temp with new random names to prevent collision
    random_names = []
    for file in files_to_shuffle:
        old_path = os.path.join(folder, file)
        random_name = f"{cat}_{uuid.uuid1()}"
        new_random_path = os.path.join(folder, random_name)
        assert not os.path.isfile(new_random_path)
        shutil.move(old_path, new_random_path)
        random_names.append(new_random_path)

    assert len(random_names) == len(files_to_shuffle)

    for i, random_file in enumerate(random_names):
        new_id = i + shuffle_idx_start
        new_path = os.path.join(folder, f'{cat}{new_id:03d}.jpg')
        assert not os.path.isfile(new_path)
        shutil.move(random_file, new_path)

    # Remove temp directory
    print(f"Successfully shuffled and renamed {len(files)} images for {cat}, keep order of first {shuffle_idx_start}")
