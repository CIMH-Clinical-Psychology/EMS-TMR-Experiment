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
import itertools
import networkx as nx
import numpy as np
import pandas as pd
import tools
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from collections import Counter, defaultdict
os.makedirs('sequences', exist_ok=True)


#%% SETTINGS
categories = ['house', 'face', 'flower', 'dog']

n_participants = 32        # how many participants should be created
n_localizer = 256          # how many localizer images
n_pairings = 128           # how many word pairings we want to create
n_cued = n_pairings//2//2  # how many words should be cued at night
n_lures_retrieval = 64     # how many new images to insert into retrieval session
perc_upside_down = 0.125   # how many images should be presented upside down in localizer
n_categories = len(categories)

n_debruijn = 3 #  len of debruijn alphabet (sequence permutation length)

t_iti_localizer = 1.5

# check that we can evenly divide the categories to the trials
assert n_localizer%n_categories==0  # check we can evenly divide n_localizer
assert n_cued%(n_categories//2)==0  # check we can evenly divide cues
assert n_pairings%n_categories==0  # check we can evenly divide pairings

seq_len = n_categories**n_debruijn
n_localizer_blocks = n_localizer // seq_len
n_distractors = int(n_localizer * perc_upside_down)
assert n_distractors%n_categories==0, 'cant split up distractors on categories!'
assert n_distractors%n_localizer_blocks==0, 'cant split up distractors on blocks'
n_dist_block = int(n_distractors / n_localizer_blocks)



#%% Tracking structures for balanced distributions
# Global counters to ensure balance across participants
image_usage_counter = defaultdict(int)  # Track how often each image is used
word_category_counter = defaultdict(lambda: defaultdict(int))  # Track word-category assignments
word_cue_counter = defaultdict(int)  # Track how often words are used as cues
all_pairings = set()  # Track all word-image pairings used to avoid duplicates

all_words = list(pd.read_excel('stimuli/words_de.xlsx').word.values.squeeze())

# Verify we have exactly the number of words needed
assert len(all_words) == n_pairings+(n_lures_retrieval*3), \
    f"Expected exactly {n_pairings+n_lures_retrieval} words, got {len(all_words)}"

# Load all images once or create mock data if needed
# already sort the images into learning/localizer/lures
images_localizer = {}
images_learning = {}
images_lures = {}

for category in categories:
    files = sorted(os.listdir(f'stimuli/{category}/'))
    files = [f'{category}/{f}' for f in files if f.endswith('.png')]

    images_learning[category] = files[:n_pairings//n_categories]

    all_images[category] = files

# Create word-category tracker for balancing word assignments across participants
word_category_participant = {word: defaultdict(set) for word in all_words}
counter_localizer = defaultdict(int)


for subj in range(n_participants):
    print(f"Creating sequences for participant {subj}")
    np.random.seed(subj + np.random.randint(0, 165245))
    random.seed(subj + np.random.randint(0, 165245))

    # Create a copy of available images for this participant
    images = {cat: all_images[cat].copy() for cat in categories}

    words = all_words.copy()
    #%% 1 localizer

    # create sequence with subsequence tuples of length 4
    # each letter stands for a specific category
    alphabet = [chr(65 + i) for i in range(len(categories))]

    df_localizer = pd.DataFrame()
    for block in range(n_localizer_blocks):

        sequence = list(tools.random_debruijn_sequence(alphabet, 3, repeating=True))

        # replace with actual categories
        sequence = list(map(lambda x: categories[ord(x)-65], sequence))

        df_localizer_block = pd.DataFrame({'category': sequence,
                                           'image': 'N/A',
                                           'block': block})

        n_localizer_per_category = seq_len // n_categories

        for char, category in zip(alphabet, categories):
            images_category = images[category][:n_localizer_per_category]
            random.shuffle(images_category)
            df_localizer_block.loc[df_localizer_block.category==category, 'image'] = images_category

            # remove these images so they can't be used later again accidentially
            for img in images_category:
                images[category].remove(img)

        df_localizer_block['distractor'] = tools.place_distractors(df_localizer_block.category,
                                                                   perc_upside_down,
                                                                   block_surrounding=0)
        df_localizer = pd.concat([df_localizer, df_localizer_block])

    # create inter trial interval by sampling from a normal distribution around 1.5
    localizer_iti =  np.random.normal(t_iti_localizer, 0.15, size=n_localizer)
    localizer_iti[localizer_iti<1] = 1
    localizer_iti[localizer_iti>2] = 2
    df_localizer['iti'] = localizer_iti

    assert sum(df_localizer.distractor) == n_distractors

    # write excel
    df_localizer.to_excel(f'sequences/{subj:02d}_localizer.xlsx')
    #%% 2. pairings

    # make sure that participan i and i+16 have the same pairings!
    # we make this to counterbalance that some pairings are cued and learned
    # and in another participant only learned and not cued (Gordons idea!).
    if subj>=16:
        print(f'for {subj=:02d} load pairings of subj{subj-16:02d}')
        df_pairings = pd.read_excel(f'sequences/{(subj-16):02d}_pairings.xlsx')

        has_more_than_three_in_a_row = True
        while has_more_than_three_in_a_row:
           df_pairings = df_pairings.sample(frac=1)  # shuffle
           has_more_than_three_in_a_row = tools.longest_streak(df_pairings.category.values)>3

    else:
        # Clone the words list for this participant
        words_sel = words[:n_pairings]
        cat_vector = categories * (n_pairings//len(categories))
        cat_vector_rolled = tools.shift(cat_vector, -subj)

        word_category_assignments = list(zip(words_sel, cat_vector_rolled, strict=True))

        has_more_than_three_in_a_row = True
        while has_more_than_three_in_a_row:
            random.shuffle(word_category_assignments)
            _cats = [x[1] for x in word_category_assignments]
            has_more_than_three_in_a_row = tools.longest_streak(_cats)>3

        # Create the actual pairings to images of the chosen category
        df_pairings = pd.DataFrame()
        for i, (word, category) in enumerate(word_category_assignments):
            # Get least used images for this category
            n_pairings_per_cat = len(word_category_assignments) // n_categories
            images_cat = images[category][:n_pairings_per_cat]
            random.shuffle(images_cat)

            # Find an image that hasn't been paired with this word
            img = None
            for potential_img in images_cat:
                if (potential_img, word) not in all_pairings:
                    img = potential_img
                    break
            else:
                raise Exception(f"Cannot find image for {category} for participant {subj}")

            # Update tracking
            images[category].remove(img)

            image_usage_counter[f"{img}/{category}"] += 1
            all_pairings.add((img, word))

            df_pairings = pd.concat([df_pairings, pd.DataFrame({'image': img,
                                                              'word': word,
                                                              'category': category},
                                                              index=[i])],
                                    ignore_index=True)

    df_pairings.to_excel(f'sequences/{subj:02d}_pairings.xlsx')

    #%% 3. Select TMR cues with balanced distribution
    cued_categories = [categories[subj%n_categories], categories[(subj+1)% n_categories]]
    if subj>=16:
        subj_match = subj-16
        cued_categories_match = [categories[subj_match%n_categories],
                                 categories[(subj_match+1)% n_categories]]
        cued_categories_new = list(set(categories).difference(set(cued_categories_match)))
        print(f'for {subj=:02d} cue opposite of subj{subj_match:02d} {cued_categories_new}|{cued_categories_match}')
        cued_categories = cued_categories_new

    df_cues = pd.DataFrame()
    for category in cued_categories:
        df_possible_words = df_pairings[df_pairings.category==category]
        df_possible_words = df_possible_words.sample(frac=1).reset_index(drop=True)

        # Sort words by how often they've been used as cues
        sorted_words = df_possible_words.sort_values(
            by='word',
            key=lambda col: col.map(lambda w: word_cue_counter.get(w, 0)) + np.random.rand()
        )

        # Select the least cued words
        selected_rows = sorted_words.iloc[:n_cued//2]
        df_cues = pd.concat([df_cues, selected_rows], ignore_index=True)

        # Update cue usage counter
        for word in selected_rows['word']:
            word_cue_counter[word] += 1

    df_cues = df_cues.sample(frac=1).reset_index(drop=True)

    # Sanity check for duplicates
    assert len(set(df_cues.word)) == len(df_cues), 'Sanity check failed, some words are doubled'

    # Shuffle dataframe to randomize presentation order
    df_cues.to_excel(f'sequences/{subj:02d}_cues.xlsx')
    #%% 4. retrieval cue selection
    # we need three retrieval sessions, one with feedback and two without
    for idx in [1, 2, 3]:
        # lure words -> distribution?
        # category for lures
        df_retrieval = df_pairings.copy()
        df_retrieval = df_retrieval[['word', 'image', 'category']]
        df_retrieval = df_retrieval.rename({'image': 'correct'}, axis=1)
        df_retrieval['is_new'] = False
        df_retrieval['cued'] = df_retrieval.correct.isin(df_cues.image)

        n_retrieval = n_pairings + n_lures_retrieval

        image1 = []
        type1 = []
        image2 = []
        type2 = []
        image3 = []
        type3 = []
        image4 = []
        type4 = []


        images_lure = df_retrieval.correct.values
        images_cat = df_retrieval.category.values
        images_cued = df_retrieval.cued.values

        # this is to help balancing the lure categories, based on the
        # current category. We rotate a list of the other categories
        # for each category individually
        lures_cats = {cat:random.sample([c for c in categories if not c==cat], 3) for cat in categories}
        same_cued = False
        other_cued = False

        for i, row in df_retrieval.iterrows():
            images_trial = [row.correct]
            img_types = ['correct/old/' + ('cued' if row.cued else 'uncued')]

            # first lure: same category, known
            mask_sel = (images_cat==row.category) & (images_lure!=row.correct)

            if row.cued:
                # if this category was cued, select alternating a cued
                # or uncued images as lured to balance
                mask_sel &= images_cued == same_cued
                same_cued = not same_cued  # flip marker to take others

            # from this subselection, choose a random element
            images_trial += [random.choice(np.array(images_lure)[mask_sel])]
            img_types += ['same/old/' + ('cued' if row.cued and same_cued else 'uncued')]

            ### second lure
            # shift the other category vector and choose category for other
            lures_cats[row.category] = tools.shift(lures_cats[row.category], 1)
            other_cat = lures_cats[row.category][0]

            mask_sel = (images_cat==other_cat)
            if other_cat in cued_categories:
                mask_sel &= images_cued == other_cued
                other_cued = not other_cued  # flip marker to take others

            images_trial += [random.choice(np.array(images_lure)[mask_sel])]
            img_types += ['other/old/' + ('cued' if row.cued and other_cued else 'uncued')]

            # # third lure: other category, new image
            image_new = np.random.choice(images[other_cat])
            images[other_cat].remove(image_new)
            images_trial += [image_new]
            img_types += ['other/new/uncued']


            images_trial, img_types = shuffle(images_trial, img_types)
            image1 += [images_trial[0]]
            image2 += [images_trial[1]]
            image3 += [images_trial[2]]
            image4 += [images_trial[3]]
            type1 += [img_types[0]]
            type2 += [img_types[1]]
            type3 += [img_types[2]]
            type4 += [img_types[3]]

        df_retrieval['image1'] = image1
        df_retrieval['type1'] = type1
        df_retrieval['image2'] = image2
        df_retrieval['type2'] = type2
        df_retrieval['image3'] = image3
        df_retrieval['type3'] = type3
        df_retrieval['image4'] = image4
        df_retrieval['type4'] = type4

        lure_words = all_words[n_pairings:]

        image1 = []
        type1 = []
        image2 = []
        type2 = []
        image3 = []
        type3 = []
        image4 = []
        type4 = []
        words = []

        for i in range(n_lures_retrieval):
            lure_cat1 = categories[i%4]
            lures_cats[lure_cat1] = tools.shift(lures_cats[lure_cat1], 1)
            lure_cat2 = lures_cats[lure_cat1][0]

            img1, img2 = np.random.choice(images[lure_cat1], 2, replace=False)
            images[lure_cat1].remove(img1)
            images[lure_cat1].remove(img2)

            img3, img4 = np.random.choice(images[lure_cat2], 2, replace=False)
            images[lure_cat2].remove(img3)
            images[lure_cat2].remove(img4)


            word = np.random.choice(lure_words)
            lure_words.remove(word)

            images_trial = [img1, img2, img3, img4]
            img_types = ['new/cued' if lure_cat1 in cued_categories else 'new/uncued']*2
            img_types += ['new/cued' if lure_cat2 in cued_categories else 'new/uncued']*2


            images_trial, img_types = shuffle(images_trial, img_types)
            image1 += [images_trial[0]]
            image2 += [images_trial[1]]
            image3 += [images_trial[2]]
            image4 += [images_trial[3]]
            type1 += [img_types[0]]
            type2 += [img_types[1]]
            type3 += [img_types[2]]
            type4 += [img_types[3]]
            words += [word]

        df_new = pd.DataFrame({'word': words,
                               'correct': 'N/A',
                               'category': 'N/A',
                               'is_new': True,
                               'cued': 'N/A'})
        df_new['image1'] = image1
        df_new['type1'] = type1
        df_new['image2'] = image2
        df_new['type2'] = type2
        df_new['image3'] = image3
        df_new['type3'] = type3
        df_new['image4'] = image4
        df_new['type4'] = type4

        # last but not least, concatenate old with new words

        df_retrieval = pd.concat([df_retrieval, df_new], ignore_index=True)

        has_more_than_three_in_a_row = True
        while has_more_than_three_in_a_row:
            df_retrieval = df_retrieval.sample(frac=1).reset_index(drop=True)
            has_more_than_three_in_a_row = tools.longest_streak(df_retrieval.category.values)>3


        df_retrieval.to_excel(f'sequences/{subj:02d}_retrieval_{idx}.xlsx')


#%% check localizer

dfs = [pd.read_excel(f'sequences/{subj:02d}_localizer.xlsx') for subj in range(n_participants)]

#### 1. check all participants have the same amount of distractors
assert (np.array([df.distractor.values for df in dfs]).sum(1)==n_distractors).all()
print('All participants have the same number of distractors')

#### 2. all the same number of categories?
assert all([set(np.unique(df.category, return_counts=True)[1])=={n_localizer//n_categories} for df in dfs])


#### 3. visualize distribution of distractor positions
# seems like we cannot balance the position of distractors across participants
# I guess it's something with the debruijn-sequence? weird..
dist_pos = sorted(np.ravel([np.where(df.distractor.values)[0] for df in dfs]))
sns.histplot(dist_pos, binwidth=1.00000001)
plt.title('distractor positions')

#### 4. next check positions of categories
plt.figure()
df_all = pd.concat(dfs)
sns.histplot(df_all, x=df_all.index, hue='category', bins=257)
plt.title('position of categories')

#### 5. check each transition is equally often per participant
allseq = []
for df in dfs:
    for i, df_block in df.groupby('block'):
        seq = list(zip(df_block.category[:-1], df_block.category[1:], strict=True))
        seq = [''.join(x) for x in seq]
        allseq += seq

plt.figure()
sns.histplot(allseq, bins=len(set(seq)))
plt.title('number of transitions equally distributed?')

# NOTE: not perfectly distributed... :-/ probably due to debruin sequence not looping? weird

#%% check pairings
dfs = [pd.read_excel(f'sequences/{subj:02d}_pairings.xlsx') for subj in range(n_participants)]

#### 1. each word is equally paired to each category
cat_pairings = []
for df in dfs:
    cat_pairings += list((df.word + df.category).values)

uniques, counts = np.unique(cat_pairings, return_counts=True)
assert len(set(counts)) == 1, 'not all categories are distributed equally to the words'

#### 2. each word is paired to each category twice
img_pairings = []
for df in dfs:
    img_pairings += list((df.word + df.image).values)

uniques, counts = np.unique(img_pairings, return_counts=True)
assert set(counts)=={2} , 'not all images are paired twice'

#%% check TMR
dfs = [pd.read_excel(f'sequences/{subj:02d}_cues.xlsx') for subj in range(n_participants)]

words_cued = []
images_cued = []
#### 1. each category is cued equally often
for df in dfs:
    uniques, counts = np.unique(df.category, return_counts=True)
    assert len(np.unique(df.word))==len(df), 'some words are cued double'
    assert len(np.unique(df.image))==len(df), 'some images are cued double'
    assert set(counts)=={n_cued//2}, f'not {n_cued=} found {counts=}'
    words_cued.append(df.word)
    images_cued.append(df.image.values)

#### 2. each word is cued equally often
uniques, counts  = np.unique(words_cued, return_counts=True)
assert len(set(counts))==1, f'some words are more often cued than others across all participants! {counts}'

#### 3. each image is equally often cued
uniques, counts  = np.unique(words_cued, return_counts=True)
assert len(set(counts))==1, f'some images are more often cued than others across all participants! {counts}'

#%% check retrievals
dfs = [pd.read_excel(f'sequences/{subj:02d}_retrieval_1.xlsx') for subj in range(n_participants)]
