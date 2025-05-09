#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 11:19:53 2025

synthethize a set of words using TTS

prerequesit
    pip install TTS
    or
    pip install google-cloud-texttospeech

@author: simon
"""
import os
import io
import pandas as pd
from tqdm import tqdm
from google.cloud import texttospeech
from scipy.io import wavfile
from pydub import AudioSegment
from functools import cache

# file needs to be present. JSON file with google credentials.
# see https://developers.google.com/workspace/guides/create-credentials
# needs to have the service active as well.
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '../../../../../Nextcloud/ZI/2020.1 Pilotstudie/google-cloud-creds.json'

@cache
def save_tts_google(word, filename, language_code="de-DE", target_length=1):

    tts_client = texttospeech.TextToSpeechClient()

    # First generate with default speaking rate to get baseline duration
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        effects_profile_id=["headphone-class-device"],
        pitch=0,
        speaking_rate=1.0
    )

    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        name="de-DE-Neural2-H"
    )

    synthesis_input = texttospeech.SynthesisInput(text=word)

    # Generate initial sample
    response = tts_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # Measure initial duration
    with io.BytesIO(response.audio_content) as f:
        # if this fails, you need to have FFMPEG installed
        # conda install -c conda-forge ffmpeg
        audio = AudioSegment.from_mp3(f)
    initial_duration = len(audio) / 1000  # Convert milliseconds to seconds

    # Calculate required speaking rate (limit adjustment to ±25%)
    speaking_rate = min(max(initial_duration / target_length, 0.75), 1.25)

    # Generate final audio with adjusted speaking rate
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        effects_profile_id=["headphone-class-device"],
        pitch=0,
        speaking_rate=speaking_rate
    )

    response = tts_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # Save the file
    with open(filename, "wb") as out:
        out.write(response.audio_content)

    # Return actual duration
    with io.BytesIO(response.audio_content) as f:
        audio = AudioSegment.from_file(f)
    final_duration = len(audio) / 1000

    return final_duration, speaking_rate
asd
#%% make the 2-syllable words

target_length = 0.85

df_words = pd.read_excel('../words_de.xlsx', index_col=0)

os.makedirs('../sounds/', exist_ok=True)

# Perform the text-to-speech request on the text input with the selected
# voice parameters and audio file type
lengths = []
files = []
speaking_rates = []

for word in tqdm(df_words.word, desc='creating audio'):
    file = f"../sounds/de_{word}.mp3"
    seconds, speaking_rate = save_tts_google(word, file, target_length=target_length)
    lengths += [seconds]
    files += [file]
    speaking_rates += [speaking_rate]
save_tts_google("Fehler, Warnung", "../sounds/error.mp3")
print(f'Lengths DE: {", ".join([f"{s:.2f}" for s in lengths])}')
print(f"lengths DE: {min(lengths):.2f}-{max(lengths):.2f} seconds")

#%
df_words['length'] = lengths
df_words['file'] = files
df_words['speaking_rate'] = speaking_rates

df_words.to_excel('../words_de.xlsx')

#%% make the 3-syllable words

target_length = 0.95

df_words = pd.read_excel('../words_de_3_silben.xlsx', index_col=0)

os.makedirs('../sounds/', exist_ok=True)

# Perform the text-to-speech request on the text input with the selected
# voice parameters and audio file type
lengths = []
files = []
speaking_rates = []

for word in tqdm(df_words.word, desc='creating audio'):
    file = f"../sounds/de_{word}.mp3"
    seconds, speaking_rate = save_tts_google(word, file, target_length=target_length)
    lengths += [seconds]
    files += [file]
    speaking_rates += [speaking_rate]
save_tts_google("Fehler, Warnung", "../sounds/error.mp3")
print(f'Lengths DE: {", ".join([f"{s:.2f}" for s in lengths])}')
print(f"lengths DE: {min(lengths):.2f}-{max(lengths):.2f} seconds")

#%
df_words['length'] = lengths
df_words['file'] = files
df_words['speaking_rate'] = speaking_rates

df_words.to_excel('../words_de_3_silben.xlsx')


#%% make the sound check-syllable words

target_length = 0.95

df_words = pd.read_excel('../../sequences/soundcheck.xlsx', index_col=0)

os.makedirs('../sounds/', exist_ok=True)

# Perform the text-to-speech request on the text input with the selected
# voice parameters and audio file type
lengths = []
files = []
speaking_rates = []

for word in tqdm(df_words.word, desc='creating audio'):
    file = f"../sounds/de_{word}.mp3"
    seconds, speaking_rate = save_tts_google(word, file, target_length=target_length)
    lengths += [seconds]
    files += [file]
    speaking_rates += [speaking_rate]
save_tts_google("Fehler, Warnung", "../sounds/error.mp3")
print(f'Lengths DE: {", ".join([f"{s:.2f}" for s in lengths])}')
print(f"lengths DE: {min(lengths):.2f}-{max(lengths):.2f} seconds")

#%
df_words['length'] = lengths
df_words['file'] = files
df_words['speaking_rate'] = speaking_rates

df_words.to_excel('../words_de_3_silben.xlsx')
