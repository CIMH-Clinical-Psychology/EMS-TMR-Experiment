#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 13:23:22 2025

1. get list of german verbs by frequency, select the top most 500 verbs
2. filter out 2-syllable verbs
3. select a subset of 160 such that the set has the largest possible
   semantic distance measures by maximum cosine similarity

you need to run "spacy download de_core_news_lg" beforehand

@author: simon kern
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
import random
from tqdm import tqdm
import pandas as pd
import pyphen
import spacy
from functools import lru_cache
cache = lru_cache()
np.random.seed(0)

#%% SETTINGS
n_syllables = 2
n_verbs = 128 + 64*3 + 16  # 128 pairings + 3x new words at retrieval + 16 controls

# exclude some profane and too long words
excluded = ['ficken', 'lutschen', 'furzen', 'poppen', 'zwiebeln', 'fohlen',
            'bumsen', 'haaren', 'modern', 'zweifeln', 'schmerzen', 'erobern',
            'stöhnen', 'blasen', 'aussteigen', 'vögeln', 'piepen', 'sondern',
            'widmen', 'nullen', 'beichten',  'bomben', 'checken', 'daten',
            'dingen', 'flammen', 'frischen',  'grenzen', 'langen', 'lauten',
            'nähern', 'osten', 'polen',   'regen', 'russen', 'schlampen',
            'sieben', 'socken', 'spuren', 'tagen', 'währen', 'stunden',
            'pissen', 'saufen', 'scheißen', 'kugeln', 'erden', 'schelten',
            'kotzen', 'ketten']

wordlist = 'https://gist.githubusercontent.com/wanderingstan/7eaaf0e22461b505c749e268c0b72bc4/raw/12ebe211a929f039791dfeaa1a019b64cadddaf1/top-german-verbs.csv'

#%% get verb list

# Fetches a list of German verbs sorted by frequency of usage.
# Sources to try
verbs = []

# Try to fetch from online frequency lists
df = cache(pd.read_csv)(wordlist)
df = df.drop([c for c in df.columns if not c in ['Rank', 'Freq', 'Infinitiv']], axis=1)

df_words = df.copy()
df_words = df_words[~df.Infinitiv.isin(excluded)]

# throw out words with frequency<1000
df_words = df_words[df_words.Freq>1000]

# count syllables
silbenzähler = pyphen.Pyphen(lang='de_DE')
df_words['Silben'] = df_words.Infinitiv.apply(lambda x: len(silbenzähler.positions(x))+1)
df_words_more_syllables = df_words[df_words['Silben']==n_syllables+1]  # filter for 3 syllables

df_words = df_words[df_words['Silben']==n_syllables]  # filter for 2 syllables
verbs = df_words.Infinitiv.values
#%% next select 168 verbs to maximize distance between them

try:
    nlp = spacy.load("de_core_news_lg")  # You might need to install this with: python -m spacy download de_core_news_md
except OSError:
    print('downloading dictionary embeddings de_core_news_lg')
    spacy.cli.download('de_core_news_lg')
    nlp = spacy.load("de_core_news_lg")  # You might need to install this with: python -m spacy download de_core_news_md

# extract embeddings
embeddings = {}
for verb in tqdm(verbs, desc='fetching embeddings'):
    doc = nlp(verb)
    if doc.has_vector:
        embeddings[verb] = doc.vector

# only take words that have an embedding
valid_verbs = [v for v in verbs if v in embeddings]
embedding_matrix = np.array([embeddings[v] for v in valid_verbs])

# Start with a random verb
selected_indices = [random.randint(0, len(valid_verbs)-1)]
selected_embeddings = [embedding_matrix[selected_indices[0]]]

# Iteratively select the verb farthest from the already selected ones
for i in range(1, n_verbs):
    # get indices that are not part of the set yet
    unselected_indices = list(set(range(len(valid_verbs))) - set(selected_indices))
    unselected_embeddings = embedding_matrix[unselected_indices]

    # For each unselected verb, find its minimum distance to any selected verb
    min_distances = np.min(cosine_distances(unselected_embeddings,
                                            selected_embeddings), axis=1)

    # Select the verb with the maximum minimum distance
    max_min_idx = np.argmax(min_distances)
    selected_indices.append(unselected_indices[max_min_idx])
    selected_embeddings.append(embedding_matrix[selected_indices[-1]])

verbs_selected = verbs[selected_indices]
assert len(verbs_selected)==n_verbs

print(verbs_selected)

#%% save

# filter and shuffle
df_words = df[df.Infinitiv.isin(verbs_selected)]
df_words = df_words.sample(frac=1).reset_index(drop=True)
df_words.rename({'Infinitiv':'word', 'Freq':'freq', 'Rank':'rank'}, axis=1, inplace=True)

df_words.to_excel('../words_de.xlsx')


#%% also create some 3-syllable practice words
wordlist = 'https://gist.githubusercontent.com/wanderingstan/7eaaf0e22461b505c749e268c0b72bc4/raw/12ebe211a929f039791dfeaa1a019b64cadddaf1/top-german-verbs.csv'
df_words_more_syllables.rename({'Infinitiv':'word', 'Freq':'freq', 'Rank':'rank'}, axis=1, inplace=True)
df_words_more_syllables = df_words_more_syllables[['word', 'rank', 'freq']].iloc[:40].reset_index(drop=True)
df_words_more_syllables.to_excel('../words_de_3_silben.xlsx')
