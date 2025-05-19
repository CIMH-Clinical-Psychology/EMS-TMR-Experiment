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
import numpy as np
import pandas as pd
import tools
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from collections import defaultdict
os.makedirs('sequences', exist_ok=True)

np.random.seed(40)
random.seed(40)


#%% SETTINGS
categories = ['house', 'face', 'flower', 'dog']

n_participants = 32        # how many participants should be created
n_localizer = 256          # how many localizer images
n_pairings = 128           # how many word pairings we want to create
n_cued = n_pairings//2//2  # how many words should be cued during sleep
n_cued_control = n_cued//2 # how many control cues played during sleep
n_lures_retrieval = 64     # how many new images to insert into retrieval session
perc_upside_down = 0.125   # how many images should be presented upside down in localizer
n_categories = len(categories)

nan_value = 'none'  # what to insert into empty slots in the dataframe

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

# these are all the categories that are being cued per participant,
# e.g. list entry 0 is the cued categories for subj 0
cued_cats_all = (list(itertools.combinations(categories, 2))*2) + [['dog', 'house'], ['face', 'flower'], ['dog', 'face'], ['house', 'flower']]

# here we load precomputed and anticlustered distributions of word indices
# to categories and participants, made with R in anticlust by Juli Nagel
df_words_categories = pd.read_csv('./stimuli/category_word_pairings.csv')

#%% Tracking structures for balanced distributions
# Global counters to ensure balance across participants
word_category_counter = defaultdict(lambda: defaultdict(int))  # Track word-category assignments
word_cue_counter = defaultdict(int)  # Track how often words are used as cues
all_pairings = set()  # Track all word-image pairings used to avoid duplicates

all_words = list(pd.read_excel('stimuli/words_de.xlsx').word.values.squeeze())

# Verify we have exactly the number of words needed
assert len(all_words) == n_pairings+(n_lures_retrieval*3) + n_cued_control, \
    f"Expected exactly {n_pairings+n_lures_retrieval} words, got {len(all_words)}"

# Load all images once or create mock data if needed
# already sort the images into learning/localizer/lures
images_localizer = {}
images_learning = {}
images_lures = {}

for category in categories:
    files = sorted(os.listdir(f'stimuli/{category}/'))
    files = [f'{category}/{f}' for f in files if f.endswith('.jpg')]
    n_learn_cat = n_pairings//n_categories
    n_loc_cat = n_localizer//n_categories
    images_learning[category] = files[:n_learn_cat]
    images_localizer[category] = files[n_learn_cat:n_learn_cat+n_loc_cat]
    images_lures[category] = files[n_learn_cat+n_loc_cat:]

    # sanity check
    assert len(set(images_learning[category]).intersection(set(images_localizer[category])))==0
    assert len(set(images_lures[category]).intersection(set(images_localizer[category])))==0
    assert len(set(images_lures[category]).intersection(set(images_learning[category])))==0

# Create word-category tracker for balancing word assignments across participants
word_category_participant = {word: defaultdict(set) for word in all_words}
counter_localizer = defaultdict(int)


for participant in range(1, n_participants+1):
    print(f"Creating sequences for {participant=}")

    # Create a copy of available words for this participant
    words_learning = all_words.copy()[:n_pairings]
    words_lures = all_words.copy()[n_pairings:]
    #%% 1 localizer

    # create sequence with subsequence tuples of length 4
    # each letter stands for a specific category
    alphabet = [chr(65 + i) for i in range(len(categories))]

    df_localizer = pd.DataFrame()

    # Create a copy of available words for this participant for localizer
    images = {cat: images_localizer[cat].copy() for cat in categories}

    for block in range(n_localizer_blocks):

        sequence = list(tools.random_debruijn_sequence(alphabet, 3, repeating=True))

        # replace with actual categories
        sequence = list(map(lambda x: categories[ord(x)-65], sequence))

        df_localizer_block = pd.DataFrame({'category': sequence,
                                           'image': nan_value,
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
    assert all(v==[] for v in images.values()), 'not all localizers used?!'
    del images # sanity check removal of images variable, recreate in next section

    # create inter trial interval by sampling from a normal distribution around 1.5
    localizer_iti =  np.random.normal(t_iti_localizer, 0.15, size=n_localizer)
    localizer_iti[localizer_iti<1] = 1
    localizer_iti[localizer_iti>2] = 2
    df_localizer['iti'] = localizer_iti

    assert sum(df_localizer.distractor) == n_distractors

    # write excel
    df_localizer.fillna(nan_value, inplace=True)
    df_localizer.to_excel(f'sequences/{participant:02d}_localizer.xlsx')
    #%% 2. pairings

    # make sure that participan i and i+16 have the same pairings!
    # we make this to counterbalance that some pairings are cued and learned
    # and in another participant only learned and not cued (Gordons idea!).
    if participant>16:
        print(f'for {participant=:02d} load pairings of subj{participant-16:02d}')
        df_pairings = pd.read_excel(f'sequences/{(participant-16):02d}_pairings.xlsx')

        has_more_than_three_in_a_row = True
        while has_more_than_three_in_a_row:
           df_pairings = df_pairings.sample(frac=1)  # shuffle
           has_more_than_three_in_a_row = tools.longest_streak(df_pairings.category.values)>3

    else:
        # Clone the words list for this participant
        cat_vector = df_words_categories[df_words_categories.participant==participant]
        cat_vector = cat_vector.sort_values('word')
        word_category_assignments = list(zip(words_learning, cat_vector.cat, strict=True))

        has_more_than_three_in_a_row = True
        while has_more_than_three_in_a_row:
            random.shuffle(word_category_assignments)
            _cats = [x[1] for x in word_category_assignments]
            has_more_than_three_in_a_row = tools.longest_streak(_cats)>3

        # Create the actual pairings to images of the chosen category
        for tries in range(5):
            error = False
            df_pairings = pd.DataFrame()
            images = {cat: images_learning[cat].copy() for cat in categories}
            for i, (word, category) in enumerate(word_category_assignments):
                # Get least used images for this category
                n_pairings_per_cat = len(word_category_assignments) // n_categories
                images_cat = images[category]
                random.shuffle(images_cat)

                # Find an image that hasn't been paired with this word
                img = None
                for potential_img in images_cat:
                    if (potential_img, word) not in all_pairings:
                        img = potential_img
                        break
                else:
                    error = True
                    print(f'trying again for {participant=} to find matching (prev {tries=})')
                    break
                    # raise Exception(f"Cannot find image for {category} for participant {subj}")

                # Update tracking
                images[category].remove(img)

                all_pairings.add((img, word))

                df_pairings = pd.concat([df_pairings, pd.DataFrame({'image': img,
                                                                    'word': word,
                                                                    'category': category},
                                                                    index=[i])],
                                        ignore_index=True)
            if not error:
                break

        assert not error

        # remove variable for safety of accidential later reuse
        assert all(v==[] for v in images.values()), 'not all learning images used?!'
        del images

    # replace nan with string NA
    df_pairings.fillna(nan_value, inplace=True)
    df_pairings.to_excel(f'sequences/{participant:02d}_pairings.xlsx', index=False)


    #%% 3. Select TMR cues with balanced distribution

    if participant<=16:
        cued_categories = cued_cats_all[participant-1]
    else:
        part_match = participant-16
        cued_categories_match = [categories[part_match%n_categories],
                                 categories[(part_match+1)% n_categories]]
        cued_categories_new = list(set(categories).difference(set(cued_categories_match)))
        print(f'for {participant=:02d} cue opposite of subj{part_match:02d} {cued_categories_new}|{cued_categories_match}')
        cued_categories = cued_categories_new

    uncued_categories = list(set(categories).difference(set(cued_categories)))


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

    # next add the control cues
    df_cues_control = pd.DataFrame({'word': [words_lures.pop() for i in range(n_cued_control)]})

    # interleave the two datasets, so 2 cues 1 control
    new_rows = []
    idx = 0
    for i, row in df_cues.iterrows():
        if i%2==0:
            new_rows += [df_cues_control.iloc[idx]]
            idx += 1
        new_rows += [row]
    df_cues = pd.concat(new_rows, axis=1).transpose()

    # Sanity check for duplicates
    assert len(set(df_cues.word)) == len(df_cues), 'Sanity check failed, some words are doubled'

    df_cues = df_cues.fillna(nan_value)
    df_cues.to_excel(f'sequences/{participant:02d}_cues_backup.xlsx', index=False)
    #%% 4. retrieval cue selection
    # we need three retrieval sessions, one with feedback and two without
    imgs_lure = {cat: shuffle(images_lures[cat].copy()) for cat in categories}

    for idx in [1, 2, 3]:
        # lure words -> distribution?
        # category for lures
        df_retrieval = df_pairings.copy()
        df_retrieval = df_retrieval[['word', 'image', 'category']]
        df_retrieval = df_retrieval.rename({'image': 'correct'}, axis=1)
        df_retrieval['is_new'] = False
        df_retrieval['cued'] = df_retrieval.correct.isin(df_cues.image)

        n_retrieval = n_pairings + n_lures_retrieval

        images_ordered = []
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

        # randomly sample a category from all other categories
        lures_cats = {cat:random.sample([c for c in categories if not c==cat], 3) for cat in categories}
        same_cued = False   # this will be flipped each time it's used to enforce balancing
        other_cued = False  # this will be flipped each time it's used to enforce balancing

        # we go through the learned list of associations and add lure images
        # additionally we add lure words that have not been learned
        other_cats = []

        ### start of assigning same_old
        # next shuffle the same_old images such that they are balanced
        # wrt the correct image and have no overlap with the correct image
        same_olds = ['' for _ in range(len(df_retrieval))]

        # for uncued category, shuffle images and redistribute
        for cat in uncued_categories:
            trials_ordered = df_retrieval[df_retrieval.category==cat].correct
            trials_shuffled = trials_ordered.values.copy()
            while any(trials_shuffled==trials_ordered):
                trials_shuffled = shuffle(trials_shuffled)
            for i, image in zip(trials_ordered.index, trials_shuffled):
                same_olds[i] = image

        # for cued category, it becomes a bit trickier: here, for the cued
        # correct images, have half of the same_old cued and the other uncued
        for cat in cued_categories:
            sel = (df_retrieval.category==cat)
            trials_cued = df_retrieval[sel & (df_retrieval.cued==True)]
            trials_uncued = df_retrieval[sel & (df_retrieval.cued==False)]
            n = len(trials_cued)//2
            # make two lists to match cued and uncued trials, each with
            # half containing cued and uncued trials themself.
            mixed1 = pd.concat([trials_cued.iloc[:n], trials_uncued.iloc[:n]]).correct.values.copy()
            mixed2 = pd.concat([trials_cued.iloc[n:], trials_uncued.iloc[n:]]).correct.values.copy()
            assert set(mixed2).intersection(set(mixed1))==set(), 'sanity check failed'

            for orig, mix_shuf in zip([trials_cued.correct, trials_uncued.correct],
                                 [mixed1, mixed2]):
                while any(mix_shuf==orig):
                    mix_shuf = shuffle(mix_shuf)
                for i, image in zip(orig.index, mix_shuf):
                    same_olds[i] = image

        ### end of assigning same_olds

        # get the same_olds from julis code
        df_other_olds = tools.assign_retrieval_other_old(df_retrieval)

        # also get new images as lures
        others_new = [imgs_lure[cat].pop() for cat in df_other_olds.other_cat]

        df_retrieval['img_correct'] = df_retrieval.correct
        df_retrieval['img_same_old'] = same_olds
        df_retrieval['img_other_old'] = df_other_olds.other_old
        df_retrieval['img_other_new'] = others_new
        df_retrieval['category_other'] = df_other_olds.other_cat

        for i, row in df_retrieval.iterrows():

            ### first image (correct)
            # collect all images/trial types in lists, then later shuffle
            # to haven them all displayed at different positions
            images_trial = [df_retrieval['img_correct'].iloc[i]]
            img_types = ['correct/old/' + ('cued' if row.cued else 'uncued')]

            # from this subselection, choose a random element
            images_trial += [df_retrieval['img_same_old'].iloc[i]]
            img_types += ['same/old/' + ('cued' if row.cued and same_cued else 'uncued')]

            ### third image (lure): other category, old
            images_trial += [df_retrieval['img_other_old'].iloc[i]]
            img_types += ['other/old/' + ('cued' if row.cued and other_cued else 'uncued')]

            ### fourth image (lure): other category, new image
            images_trial += [df_retrieval['img_other_new'].iloc[i]]
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

        df_retrieval['image_top'] = image1
        df_retrieval['type_top'] = type1
        df_retrieval['image_right'] = image2
        df_retrieval['type_right'] = type2
        df_retrieval['image_bottom'] = image3
        df_retrieval['type_bottom'] = type3
        df_retrieval['image_left'] = image4
        df_retrieval['type_left'] = type4

        ### assign new words
        # next select lure words from the list of remaining words.
        image1 = []
        type1 = []
        image2 = []
        type2 = []
        image3 = []
        type3 = []
        image4 = []
        type4 = []
        words = []
        cat_lures1 = []
        cat_lures2 = []
        words_lures = shuffle(words_lures)
        words = [words_lures.pop() for _ in range(n_lures_retrieval)]
        df_new = pd.DataFrame({'word': words,
                               'is_new': True})

        # in the third retrieval (post sleep) we also show images
        if idx==3:
            categories_new = list(itertools.combinations(categories, 2))*10
            categories_new += [['house', 'face'], ['flower', 'dog'], ['house', 'flower'], ['face', 'dog']]
            categories_new = [shuffle(x) for x in categories_new]
            cats_new1 = [x[0] for x in categories_new]
            cats_new2 = [x[1] for x in categories_new]

            overlap = True
            while overlap:
                imgs = {c:list(df_retrieval[df_retrieval.category==c].correct.values)*2 for c in categories}
                imgs = {c: shuffle(l) for c, l in imgs.items()}
                imgs1 = [(imgs[c]).pop() for c in cats_new1]
                imgs = {c: shuffle(l) for c, l in imgs.items()}
                imgs2 = [(imgs[c]).pop() for c in cats_new1]
                imgs = {c: shuffle(l) for c, l in imgs.items()}
                imgs3 = [(imgs[c]).pop() for c in cats_new2]
                imgs = {c: shuffle(l) for c, l in imgs.items()}
                imgs4 = [(imgs[c]).pop() for c in cats_new2]

                trials = zip(imgs1, imgs2, imgs3, imgs4, strict=True)
                if not any([len(set(x))<4 for x in trials]):
                    overlap=False

            cued = dict(zip(df_retrieval.correct, df_retrieval.cued))
            types1 = [f'cat1/old/{"cued" if cued[img] else "uncued"}' for img in imgs1]
            types2 = [f'cat1/old/{"cued" if cued[img] else "uncued"}' for img in imgs2]
            types3 = [f'cat2/old/{"cued" if cued[img] else "uncued"}' for img in imgs3]
            types4 = [f'cat2/old/{"cued" if cued[img] else "uncued"}' for img in imgs4]

            df_new['category_new1'] = cats_new1
            df_new['category_new2'] = cats_new2
            df_new['image_top'] = imgs1
            df_new['type_top'] = types1
            df_new['image_right'] = imgs2
            df_new['type_right'] = types2
            df_new['image_bottom'] = imgs3
            df_new['type_bottom'] = types3
            df_new['image_left'] = imgs4
            df_new['type_left'] = types4

        if idx>3:
            raise ValueError('more than 3 retrieval sessions??')

        # last but not least, concatenate old with new words
        df_retrieval = pd.concat([df_retrieval, df_new], ignore_index=True)

        # pseudorandomize: we don't want to have too long streaks of any
        # trial property in a row
        x = 0
        while True:
            df_retrieval = df_retrieval.sample(frac=1).reset_index(drop=True)
            x += 1
            assert x<10000, 'no possible shuffle!'

            # has_more_than_three_in_a_row?
            if tools.longest_streak(df_retrieval.category.values)>4:
                # the true category is the most visible
                continue
            if tools.longest_streak(df_retrieval.is_new.values)>6:
                # should not be too many new words in a row
                continue
            if tools.longest_streak(df_retrieval.type_left.values)>4:
                continue
            if tools.longest_streak(df_retrieval.type_top.values)>4:
                continue
            if tools.longest_streak(df_retrieval.type_right.values)>4:
                continue
            if tools.longest_streak(df_retrieval.type_bottom.values)>4:
                continue
            # if all is satisfied, break free from the loop
            break

        # create inter trial interval by sampling from a normal distribution around 1.5
        retrieval_iti =  abs(np.random.normal(0, 0.15, size=len(df_retrieval)))
        retrieval_iti[retrieval_iti>0.5] = 5
        df_retrieval['iti'] = 1 + retrieval_iti

        df_retrieval = df_retrieval.fillna(nan_value)
        df_retrieval.to_excel(f'sequences/{participant:02d}_retrieval_{idx}.xlsx', index=False)
#%% CHECKS
#%% check localizer

dfs = [pd.read_excel(f'sequences/{subj:02d}_localizer.xlsx') for subj in range(1, n_participants+1)]

#### 1. check all participants have the same amount of distractors
assert (np.array([df.distractor.values for df in dfs]).sum(1)==n_distractors).all()
print('All participants have the same number of distractors')

#### 2. all the same number of categories?
assert all([set(np.unique(df.category, return_counts=True)[1])=={n_localizer//n_categories} for df in dfs])


#### 3. visualize distribution of distractor positions
# seems like we cannot balance the position of distractors across participants
# I guess it's something with the debruijn-sequence? weird..
dist_pos = sorted(np.ravel([np.where(df.distractor.values)[0] for df in dfs]))
plt.figure()
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
dfs = [pd.read_excel(f'sequences/{subj:02d}_pairings.xlsx') for subj in range(1, 1+n_participants)]

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

# check the categories are cued equally often and the category pairs as well
assert len(cued_cats_all)==16
cues_sorted = [sorted(x) for x in cued_cats_all]
tuples, counts = np.unique(cues_sorted, axis=0, return_counts=True)
cats, counts = np.unique(cues_sorted, axis=None, return_counts=True)
assert len(set(counts))==1, 'Warning, not all categories are cued equally often'
assert np.ptp(counts)<=1, 'warning, discrepancy in number of cued tuples'

dfs = [pd.read_excel(f'sequences/{subj:02d}_cues_backup.xlsx') for subj in range(1, 1+n_participants)]

words_cued = []
images_cued = []
#### 1. each category is cued equally often
for df in dfs:

    # only select the ones that are cued
    df = df[df.image!=nan_value]
    uniques, counts = np.unique(df.category, return_counts=True)
    assert len(np.unique(df.word))==len(df), 'some words are cued double'
    assert len(np.unique(df.image))==len(df), 'some images are cued double'
    assert set(counts)=={n_cued//2}, f'not {n_cued=} found {counts=}'
    words_cued.append(df.word.values)
    images_cued.append(df.image.values)

#### 2. each word is cued equally often
uniques, counts  = np.unique(words_cued, return_counts=True)
print(f'range counts of words occurring in cueing ranges from {min(counts)} to {max(counts)}')
assert np.ptp(counts)<=3, f'some words are more often cued than others across all participants! {counts}'

#### 3. each image is equally often cued
uniques, counts  = np.unique(words_cued, return_counts=True)
print(f'range counts of images occurring in cueing ranges from {min(counts)} to {max(counts)}')
assert np.ptp(counts)<=3, f'some images are more often cued than others across all participants! {counts}'

#%% check retrievals

for subj in range(1, 1+n_participants):
    new_words = []
    old_words = []
    new_images = []
    new_words_lures = []
    for session in [1,2,3]:
        df = pd.read_excel(f'sequences/{subj:02d}_retrieval_{session}.xlsx')
        df_old = df[~df.is_new]
        df_new = df[df.is_new]
        new_words.extend(df_new.word)
        old_words.extend(df_old.word)
        new_images.extend(df_old.img_other_new)
        ### check no lure comes twice
        assert len(set(df.word))==len(df), 'some words are double within the session'

        # check that the other category is chosen equally often
        cats, counts = np.unique(df_old.category_other, return_counts=True)
        assert np.std(counts)==0, '{counts=}, some categories are more often chosen as "other category" for lures'

        # check that within a session each lure image is shown once in same
        imgs, counts = np.unique(df_old.img_same_old, return_counts=True)
        assert np.std(counts)==0, 'some lure images are shown more often than others'

        # check that within a session each lure image is shown once in other
        imgs, counts = np.unique(df_old.img_other_old, return_counts=True)
        assert np.std(counts)==0, 'some lure images are shown more often than others'

        # check that in combination all lures are occuring twice
        both_lures = [x for x in df_old.img_other_old] + [x for x in df_old.img_same_old]
        imgs, counts = np.unique(df_old.img_other_old, return_counts=True)
        assert np.std(counts)==0, 'some lure images are shown more often than others'

        if session==3:
            # check that categories appear equally often
            tmp_cats = pd.concat([df_new.category_new1, df_new.category_new2])
            cats, counts = np.unique(tmp_cats, return_counts=True)
            assert np.std(counts)==0, 'some categories are shown more often as lure1'

            # check categorie combinations
            cats_combined = [sorted([cat1, cat2]) for cat1, cat2 in zip(df_new.category_new1, df_new.category_new2)]
            cats, counts = np.unique(cats_combined, return_counts=True)
            assert np.std(counts)==0, 'some categories combinations appear more often'

            # also check that new images are not shown anywhere else
            x = pd.concat([df.image_left, df.image_top, df.image_right, df.image_bottom])
            imgs, counts = np.unique(x, return_counts=True)
            assert set(counts)== {1, 5}, 'some images are shown more often than others!'

            # check that all images for new words are actually new
            df_new = df[df.correct=='none']
            new_words_lures.extend(df_new.image_bottom)
            new_words_lures.extend(df_new.image_top)
            new_words_lures.extend(df_new.image_left)
            new_words_lures.extend(df_new.image_right)
            assert len(set(new_words_lures))*2==len(new_words_lures)

    # check each lure word/image is unique

    assert set(np.unique(new_images, return_counts=True)[1])=={1}, 'some new images are double'
    assert set(np.unique(new_words, return_counts=True)[1])=={1}, 'some new words are double'
    assert set(np.unique(old_words, return_counts=True)[1])=={3}, 'some old words are not shown 3 times'
