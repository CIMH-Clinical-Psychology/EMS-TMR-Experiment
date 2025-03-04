#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 10:06:41 2025

this file creates the stimuli order and pairing for the psychopy experiment.
it will create the following files per participant
 - XX-word-pairings.csv - pairings of images to words for learning
 - XX-tmr-cues.csv      - list of cues that will be replayed for participant
 - XX-localizer.csv     - order of localizer images with attention check

@author: simon kern
"""
import os
import random
import warnings
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
os.makedirs('sequences', exist_ok=True)

#%% SETTINGS
categories = ['car', 'face', 'flower', 'dog']

n_participants = 32 # how many participants should be created
n_localizer = 240   # how many localizer images
n_pairings = 120    # how many word pairings we want to create
n_cued = 60         # how many words should be cued at night
n_categories = len(categories)

# check that we can evenly divide the categories to the trials
assert n_localizer%n_categories==0  # check we can evenly divide n_localizer
assert n_cued%(n_categories//2)==0  # check we can evenly divide cues
assert n_pairings%n_categories==0  # check we can evenly divide pairings

#%% Tracking structures for balanced distributions
# Global counters to ensure balance across participants
image_usage_counter = defaultdict(int)  # Track how often each image is used
word_category_counter = defaultdict(lambda: defaultdict(int))  # Track word-category assignments
word_cue_counter = defaultdict(int)  # Track how often words are used as cues
all_pairings = set()  # Track all word-image pairings used to avoid duplicates

all_words = list(pd.read_csv('stimuli/words.csv').values.squeeze())

# Verify we have exactly the number of words needed
assert len(all_words) == n_pairings, f"Expected exactly {n_pairings} words, got {len(all_words)}"

# Load all images once or create mock data if needed
all_images = {}
for category in categories:
    all_images[category] = os.listdir(f'stimuli/{category}/')

# Create word-category tracker for balancing word assignments across participants
word_category_participant = {word: defaultdict(set) for word in all_words}
counter_localizer = defaultdict(int)


for subj in range(n_participants):
    print(f"Creating sequences for participant {subj}")
    np.random.seed(subj)
    random.seed(subj)

    # Create a copy of available images for this participant
    images = {cat: all_images[cat].copy() for cat in categories}

    #%% localizer
    ### 1. Create localizer sequence
    df_localizer = pd.DataFrame()
    for category in categories:
        n_per_category = n_localizer//n_categories
        random.shuffle(images[category])  # shuffle before sorting

        # Sort images by usage to prioritize less used ones
        sorted_images = sorted(images[category],
                              key=lambda img: counter_localizer.get(f"{img}/{category}", 0))
        imgs = sorted_images[:n_per_category]

        df_tmp = pd.DataFrame({'imgs': imgs, 'category': category})
        df_localizer = pd.concat([df_localizer, df_tmp])

        # Update usage counter and remove chosen images from available set
        for img in imgs:
            counter_localizer[f"{img}/{category}"] += 1
            images[category].remove(img)

    # just shuffle the list
    df_localizer = df_localizer.sample(frac=1).reset_index(drop=True)
    df_localizer.to_csv(f'sequences/{subj:02d}_localizer.csv')

    #%% pairings

    # 2. Create word-image pairings with balanced distribution
    # Clone the words list for this participant
    participant_words = all_words.copy()
    random.shuffle(participant_words)

    # Distribute words evenly across categories for this participant
    words_per_category = n_pairings // n_categories
    word_category_assignments = []

    # Ensure balanced category assignment across participants
    for category in categories:
        # Sort words by how often they've been assigned to this category
        sorted_words = sorted(participant_words,
                             key=lambda w: len(word_category_participant[w][category]))

        # Assign words to this category
        category_words = sorted_words[:words_per_category]

        for word in category_words:
            word_category_assignments.append((word, category))
            participant_words.remove(word)
            word_category_participant[word][category].add(subj)

    random.shuffle(word_category_assignments)

    # Create the actual pairings to images of the chosen category
    df_pairings = pd.DataFrame()
    for i, (word, category) in enumerate(word_category_assignments):
        # Get least used images for this category
        available_cat_images = images[category].copy()
        sorted_images = sorted(available_cat_images,
                              key=lambda img: image_usage_counter.get(f"{img}/{category}", 0))

        # Find an image that hasn't been paired with this word
        img = None
        for potential_img in sorted_images:
            if (potential_img, word) not in all_pairings:
                img = potential_img
                break

        if img is None:
            warnings.warn(f"Cannot find image for {category} for participant {subj}")
            continue

        # Update tracking
        images[category].remove(img)
        image_usage_counter[f"{img}/{category}"] += 1
        all_pairings.add((img, word))

        df_pairings = pd.concat([df_pairings, pd.DataFrame({'image': img,
                                                          'word': word,
                                                          'category': category},
                                                          index=[i])])

    df_pairings.to_csv(f'sequences/{subj:02d}_pairings.csv')

    #%% 3. Select TMR cues with balanced distribution
    cued_categories = np.random.choice(categories, size=2, replace=False)
    df_cues = pd.DataFrame()

    for category in cued_categories:
        df_possible_words = df_pairings[df_pairings.category==category]
        df_possible_words = df_possible_words.sample(frac=1).reset_index(drop=True)

        # Sort words by how often they've been used as cues
        sorted_words = df_possible_words.sort_values(
            by='word',
            key=lambda col: col.map(lambda w: word_cue_counter.get(w, 0))
        )

        # Select the least cued words
        selected_rows = sorted_words.iloc[:n_cued//2]
        df_cues = pd.concat([df_cues, selected_rows], ignore_index=True)

        # Update cue usage counter
        for word in selected_rows['word']:
            word_cue_counter[word] += 1

    # Sanity check for duplicates
    df_cues = df_cues.sample(frac=1).reset_index(drop=True)

    assert len(set(df_cues.word)) == len(df_cues), 'Sanity check failed, some words are doubled'

    # Shuffle dataframe to randomize presentation order
    df_cues.to_csv(f'sequences/{subj:02d}_cues.csv')

#%% Final distribution checks
print("\n=== DISTRIBUTION STATISTICS ===")

# 1. Check image usage balance
print("\nImage usage statistics:")
category_usage = {cat: [] for cat in categories}
for key, count in image_usage_counter.items():
    cat = key.split('/')[1]
    category_usage[cat].append(count)

for cat, counts in category_usage.items():
    if counts:
        print(f"{cat}: min={min(counts)}, max={max(counts)}, mean={np.mean(counts):.2f}")

# 2. Check word-category assignments across participants
print("\nWord-category distribution (number of participants for each assignment):")
word_cat_counts = defaultdict(list)
for word in all_words:
    counts = []
    for cat in categories:
        num_participants = len(word_category_participant[word][cat])
        counts.append(num_participants)
        word_cat_counts[cat].append(num_participants)

    # Print words with significant imbalance
    if max(counts) - min(counts) > 3:
        cats = [f"{cat}:{len(word_category_participant[word][cat])}" for cat in categories]
        print(f"{word}: {', '.join(cats)}")

# Print summary
print("\nWord-category assignment summary:")
for cat in categories:
    counts = word_cat_counts[cat]
    print(f"{cat}: min={min(counts)}, max={max(counts)}, mean={np.mean(counts):.2f}")

# 3. Check cue distribution
print("\nCue distribution:")
cue_counts = [(word, count) for word, count in word_cue_counter.items()]
if cue_counts:
    cue_values = [count for _, count in cue_counts]
    print(f"Min cues: {min(cue_values) if cue_values else 0}, "
          f"Max cues: {max(cue_values) if cue_values else 0}, "
          f"Mean: {np.mean(cue_values) if cue_values else 0:.2f}")

    # Print words with extreme values
    max_cues = max(cue_values) if cue_values else 0
    min_cues = min(cue_values) if cue_values else 0

    print(f"Words never cued: {len([w for w, c in cue_counts if c == 0])}")
    if max_cues - min_cues > 2:
        print(f"Words cued {max_cues} times: {[w for w, c in cue_counts if c == max_cues]}")

print("\nSequence creation complete.")
