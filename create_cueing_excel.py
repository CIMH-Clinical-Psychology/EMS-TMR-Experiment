# -*- coding: utf-8 -*-
"""
Created on Thu May 15 13:56:54 2025

read previous response log files and randomly choose

@author: Simon.Kern
"""
import os
import tools
import numpy as np
import pandas as pd
import re
from datetime import datetime
from anti_clustering import ExchangeHeuristicAntiClustering

def sort_by_timestamp(list_of_files):
    # Regex to extract the timestamp part: YYYY-MM-DD_HHhMM.SS.mmm
    timestamp_pattern = re.compile(r'(\d{4}-\d{2}-\d{2}_\d{2}h\d{2}\.\d{2}\.\d{3})')

    def extract_timestamp(file_path):
        match = timestamp_pattern.search(file_path)
        if not match:
            return datetime.min  # fallback if no timestamp found
        ts = match.group(1)
        # Convert to datetime object
        # Example: 2025-05-17_09h12.12.810
        return datetime.strptime(ts, "%Y-%m-%d_%Hh%M.%S.%f")
    return sorted(list_of_files, key=extract_timestamp, reverse=True)

subj_id = input('Please enter participant number: ').strip()

assert subj_id.isdigit(), 'please provide an integer'
subj_id = f'{int(subj_id):02d}'

df_backup_cues = pd.read_excel(f'./sequences/{subj_id}_cues_backup.xlsx')

print(f'\nlooking for files starting with {subj_id}_EMS_TMR_feedbackv1 in {os.path.abspath("./data")}')
files1 = tools.list_files('./data', patterns=f'{subj_id}_EMS-TMR_retrievalv1*feedback.csv')
files2 = tools.list_files('./data', patterns=f'{subj_id}_EMS-TMR_retrievalv1*preSleep.csv')

assert files1, f'No feedback file found for {subj_id}!'
assert files2, f'No feedback file found for {subj_id}!'

files1 = sort_by_timestamp(files1)
files2 = sort_by_timestamp(files2)

assert files1[0] == sorted(files1, key=lambda x: os.path.getctime(x))[0], 'newest file is not most recent file?'
assert files2[0] == sorted(files2, key=lambda x: os.path.getctime(x))[0], 'newest file is not most recent file?'

# if more than one file is found, take the newest one
if len(files1)>1:
    print(f'MORE THAN ONE MATCHING FEEDBACK FILE! taking newest: {os.path.basename(files1[0])}')
if len(files2)>1:
    print(f'MORE THAN ONE MATCHING PRESLEEP FILE! taking newest: {os.path.basename(files2[0])}')

df_feedback = pd.read_csv(files1[0])
df_presleep = pd.read_csv(files2[0])

assert len(df_feedback)==320
assert len(df_presleep)==320

if any(df_feedback['trial_type'].isna()) or any(df_presleep['trial_type'].isna()):
    import warnings
    warnings.warn('trial type old/new hack still enabled, should not be the case')
    counts_feedback = np.bincount(df_feedback.trial_num)
    counts_presleep = np.bincount(df_presleep.trial_num)
    def tyial_type_feedback(x):
        return 'old' if counts_feedback[x]==2 else 'new'
    def tyial_type_presleep(x):
        return 'old' if counts_presleep[x]==2 else 'new'

    df_feedback['trial_type'] = df_feedback.trial_num.apply(tyial_type_feedback)
    df_presleep['trial_type'] = df_presleep.trial_num.apply(tyial_type_presleep)

df_feedback.response = df_feedback.response.apply(lambda x: 'false' if x in ['none', np.nan] else x)
df_presleep.response = df_presleep.response.apply(lambda x: 'false' if x in ['none', np.nan] else x)


df_word1 = df_feedback[(df_feedback.session=='word_feedback') & (df_feedback.trial_type=='old')]
df_img1 = df_feedback[df_feedback.session=='image_feedback']
df_word2 = df_presleep[(df_presleep.session=='word_preSleep') & (df_presleep.trial_type=='old')]
df_img2 = df_presleep[df_presleep.session=='image_preSleep']


df1 = pd.DataFrame({'pair': df_word1.stimuli.values + ' - '+ df_img1.stimuli.values,
                    'word': df_word1.stimuli.values,
                    'image': df_img1.stimuli.values,
                    'word1_correct': df_word1.response.values=='correct',
                    'img1_correct': df_img1.response.values=='correct'})

df2 = pd.DataFrame({'pair': df_word2.stimuli.values + ' - '+ df_img2.stimuli.values,
                    'word2_correct': df_word2.response.values=='correct',
                    'img2_correct': df_img2.response.values=='correct'})

df1 = df1.set_index('pair').sort_index()
df2 = df2.set_index('pair').sort_index()

assert sorted(df1.index) == sorted(df2.index), 'index mismatch between feedback and presleep!'

df_joined = df1.join(df2, on = ['pair'])
df_joined['category'] = df_joined.image.apply(lambda x: x.split('/')[0])

#%% anticlust calculations

categories = df_backup_cues[df_backup_cues.category!='none'].category.unique()
df_cues_control = df_backup_cues[df_backup_cues.category=='none']


algo = ExchangeHeuristicAntiClustering()
df_cues = pd.DataFrame()
for category in categories:
    df_category = algo.run(
        df=df_joined[df_joined.category==category],
        numerical_columns=['word1_correct', 'img1_correct', 'word2_correct', 'img2_correct'],
        categorical_columns=None,
        num_groups=2,
        destination_column='Cluster'
    )
    idx = (df_category.Cluster==1).values
    df_cues = pd.concat([df_cues,
                         pd.DataFrame({'word': df_category.word[idx].values,
                                       'image': df_category.image[idx].values,
                                       'category': df_category.category[idx].values})],
                        ignore_index=True)

df_cues = df_cues.sample(frac=1).reset_index(drop=True)
# interleave the two datasets, so 2 cues 1 control
new_rows = []
idx = 0
for i, row in df_cues.iterrows():
    if i%2==0:
        new_rows += [df_cues_control.iloc[idx]]
        idx += 1
    new_rows += [row]

df_cues_final = pd.concat(new_rows, axis=1, ignore_index=True).transpose()
df_cues_final.to_excel(f'./sequences/{subj_id}_cues.xlsx', index=False)
