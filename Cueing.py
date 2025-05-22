# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 08:56:31 2025

@author: elena
Cueing Session
"""
import pygame
import pandas as pd
import time
import threading
import tools
import meg_triggers
from meg_triggers import send_trigger
meg_triggers.enable_printing()

# Initialisierung von pygame
pygame.init()

def main():
    # Datei-Pfade
    subj_id = input("Bitte geben Sie die Probandennummer ein: ")
    sound_a_volume = float(input ("Bitte gib die ausgewählte Lautstärke für den PinkNoise ein: "))
    sound_b_volume = float(input("Bitte gib die ausgewählte Lautstärke für die Cues ein: "))
    excel_file = f"./sequences/{subj_id}_cues.xlsx"
    sound_a_path = f"./stimuli/PinkNoise-60min.mp3"

    # Lade die Sound-Daten aus der Excel-Datei
    try:
        sounds_data = pd.read_excel(excel_file)
    except FileNotFoundError:
        print (f"Die Datei {excel_file} wurde nicht gefunden.")
        return
 
    print('setting system volume to 0.07')
    tools.set_system_volume(0.07)   
 
    sound_b_list = sounds_data['word'].tolist()
    # Fenster, dass Key Presses erkannt werden
    pygame.display.set_mode((100, 100))

    # Sound A initialisieren
    print('loading noise file...')
    pygame.mixer.init()
    sound_a = pygame.mixer.Sound(sound_a_path)
    sound_a_channel = sound_a.play(-1)  # Sound A kontinuierlich abspielen
    print('ready!\n')

    #trigger Kategorien definieren
    category_mapping = {
    "dog": 1,
    "house": 2,
    "face": 3,
    "flower": 4,
    "none": 5
    }
    sounds_data["cat"] = sounds_data["category"].map(category_mapping)

    print (f"Read.\n\nvolume Noise: {sound_a_volume:0.4f} Cues: {sound_b_volume:0.4f}\n\n")
    volume_increment_a = 0.01
    volume_increment_b = 0.005
    
    # Lautstärken
    sound_a.set_volume(sound_a_volume)
    stop_sound_b = threading.Event()
    
    def play_sound_b():
        nonlocal sound_a_volume, sound_b_volume
        index = 0
        repeat = 0

        while True:
            stop_sound_b.wait()  # Warten, bis der Prozess für Sound B aktiviert wird

            while stop_sound_b.is_set() and index < len(sound_b_list):
                #pygame.mixer.init()
                sound_b = pygame.mixer.Sound (f"./stimuli/sounds/de_" + sound_b_list[index] + ".mp3")
                sound_b.set_volume(sound_b_volume)
                sound_b.play()
                print ("sound played:", f'"{sound_b_list[index]}" category: {sounds_data.loc[index, "cat"]} [{sounds_data.loc[index, "category"]}] {index=}')
                trigger_cat = (int(sounds_data.loc[index, "cat"])-1)*50 + index +1
                send_trigger(trigger_cat, duration = 0.005)
                time.sleep(7)# 7 Sekunden Pause nach Sound B

                index += 1

            if index >= len(sound_b_list):
                index = 0  # Liste von vorne abspielen, wenn das Ende erreicht ist
                repeat += 1
                print ("Liste wird wiederholt", repeat)
                send_trigger(254, duration = 0.005)

    # Thread für Sound B starten
    sound_b_thread = threading.Thread(target=play_sound_b)
    sound_b_thread.daemon = True
    sound_b_thread.start()

    # Steuerung über die Tastatur
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if stop_sound_b.is_set():
                        stop_sound_b.clear()  # Sound B pausieren
                        print ("Cueing beendet")
                        send_trigger (252, duration = 0.005)
                        time.sleep(0.2)

                    else:
                        send_trigger (253, duration = 0.005)
                        time.sleep(0.2)
                        stop_sound_b.set()  # Sound B und Pause starten
                        print ("Cueing started")

                elif event.key == pygame.K_UP:
                    sound_a_volume = min(1.0, sound_a_volume + volume_increment_a)
                    sound_a.set_volume(sound_a_volume)
                    print (f"volume Noise: {sound_a_volume:0.3f}")

                elif event.key == pygame.K_DOWN:
                    sound_a_volume = max(0.0, sound_a_volume - volume_increment_a)
                    sound_a.set_volume(sound_a_volume)
                    print (f"volume Noise: {sound_a_volume:0.3f}")

                elif event.key == pygame.K_RIGHT:
                    sound_b_volume = min(1.0, sound_b_volume + volume_increment_b)
                    print (f"volume Cues: {sound_b_volume:0.3f}")

                elif event.key == pygame.K_LEFT:
                    sound_b_volume = max(0.0, sound_b_volume - volume_increment_b)
                    print (f"volume Cues: {sound_b_volume:0.3f}")

                elif event.key == pygame.K_ESCAPE:
                    running = False
        

    pygame.quit()

if __name__ == "__main__":
    main()

