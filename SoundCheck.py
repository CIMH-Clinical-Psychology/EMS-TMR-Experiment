# -*- coding: utf-8 -*-
"""
Created on Mon May  5 09:54:34 2025

@author: elena
"""

import pygame
import pandas as pd
import time
import threading
import random
import tools

# Initialisierung von pygame
pygame.init()




def main():
    # Datei-Pfade
    excel_file = f"./sequences/soundcheck.xlsx"
    sound_a_path = f"./stimuli/PinkNoise-60min.mp3"
    
    print('setting system volume to 0.07')
    tools.set_system_volume(0.07)
    
    # Lade die Sound-Daten aus der Excel-Datei
    try:
        sounds_data = pd.read_excel(excel_file)
    except FileNotFoundError:
        print (f"Die Datei {excel_file} wurde nicht gefunden.")
        return
    
    sound_b_list = sounds_data['word'].tolist()
    # Fenster, dass Key Presses erkannt werden
    pygame.display.set_mode((100, 100))

    # Sound A initialisieren
    print('loading pink noise file...')
    pygame.mixer.init()
    sound_a = pygame.mixer.Sound(sound_a_path)
    sound_a_channel = sound_a.play(-1)  # Sound A kontinuierlich abspielen
    
    # Lautstärken
    sound_a_volume = 0.5
    sound_b_volume = 0
    sound_a.set_volume(sound_a_volume)
    print (f"Read.\n\nvolume Noise: {sound_a_volume:0.4f} Cues: {sound_b_volume:0.4f}\n\n")
    volume_increment_a = 0.01
    volume_increment_b = 0.005

    stop_sound_b = threading.Event()

    def play_sound_b():
        nonlocal sound_a_volume, sound_b_volume
        index = 0

        while True:
            stop_sound_b.wait()  # Warten, bis der Prozess für Sound B aktiviert wird

            while stop_sound_b.is_set() and index < len(sound_b_list):
                #pygame.mixer.init()
                sound_b = pygame.mixer.Sound (f"./stimuli/sounds/de_" + sound_b_list[index] + ".mp3")
                sound_b.set_volume(sound_b_volume)
                #if sound_b_volume>0:
                #    # reduce sound of noise while cue is playing
                #    sound_a.set_volume(sound_a_volume-sound_b_volume)
                #    sound_b.play()
                #    time.sleep(sound_b.get_length())
                #    sound_a.set_volume(sound_a_volume+sound_b_volume)

                print ("sound played:", sound_b_list[index], f'Volume: {sound_b_volume:0.3f}')
                time.sleep(3)
                index += 1

            if index >= len(sound_b_list):
                index = 0  # Liste von vorne abspielen, wenn das Ende erreicht ist
                random.shuffle(sound_b_list)
                
    # Thread für Sound B starten
    sound_b_thread = threading.Thread(target=play_sound_b)
    sound_b_thread.daemon = True
    sound_b_thread.start()
    stop_sound_b.set()  # Sound B und Pause starten

    # Steuerung über die Tastatur
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if stop_sound_b.is_set():
                        stop_sound_b.clear()  # Sound B pausieren
                        print ("Cueing beendet")
                    else:
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
                #print (f"volume Noise: {sound_a_volume:0.4f} Cues: {sound_b_volume:0.4f} ")


    pygame.quit()

if __name__ == "__main__":
    main()