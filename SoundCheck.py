# -*- coding: utf-8 -*-
"""
Created on Mon May  5 09:54:34 2025

@author: elena
"""

import pygame
import pandas as pd
import time
import threading


# Initialisierung von pygame
pygame.init()

def main():
    # Datei-Pfade
    excel_file = f"./sequences/soundcheck.xlsx"
    sound_a_path = f"./stimuli/PinkNoise-60min.mp3"

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
    pygame.mixer.init()
    sound_a = pygame.mixer.Sound(sound_a_path)
    sound_a_channel = sound_a.play(-1)  # Sound A kontinuierlich abspielen
    
    # Lautstärken
    sound_a_volume = 0.5
    sound_b_volume = 0.5
    sound_a.set_volume(sound_a_volume)

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
                sound_b.play()
                print ("sound played:", sound_b_list[index])

                time.sleep(7)# 7 Sekunden Pause nach Sound B
                #send trigger, dass Pause gestartet hat

                index += 1

            if index >= len(sound_b_list):
                index = 0  # Liste von vorne abspielen, wenn das Ende erreicht ist

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
                    else:
                        stop_sound_b.set()  # Sound B und Pause starten
                        print ("Cueing started")

                elif event.key == pygame.K_UP:
                    if sound_a_volume >= 0.2:
                        sound_a_volume = min(3.0, sound_a_volume + 0.1)
                    elif sound_a_volume < 0.2:
                        sound_a_volume = min(3.0, sound_a_volume + 0.05)
                    sound_a.set_volume(sound_a_volume)
                    print ("volume Pink Noise:", sound_a_volume)

                elif event.key == pygame.K_DOWN:
                    if sound_a_volume >= 0.2:
                        sound_a_volume = max(0.0, sound_a_volume - 0.1)
                    elif sound_a_volume < 0.2:
                        sound_a_volume = max(0.0, sound_a_volume - 0.05)
                    sound_a.set_volume(sound_a_volume)
                    print ("volume Pink Noise:", sound_a_volume)

                elif event.key == pygame.K_RIGHT:
                    if sound_b_volume >= 0.2:
                        sound_b_volume = min(3.0, sound_b_volume + 0.1)
                    elif sound_b_volume < 0.2:
                        sound_b_volume = min(3.0, sound_b_volume + 0.05)
                    print ("volume Cues:", sound_b_volume)

                elif event.key == pygame.K_LEFT:
                    if sound_b_volume >= 0.2:          
                        sound_b_volume = max(0.0, sound_b_volume - 0.1)
                    elif sound_b_volume < 0.2:
                        sound_b_volume = max(0.0, sound_b_volume - 0.05)
                    print ("volume Cues:", sound_b_volume)

                elif event.key == pygame.K_ESCAPE:
                    running = False
        

    pygame.quit()

if __name__ == "__main__":
    main()